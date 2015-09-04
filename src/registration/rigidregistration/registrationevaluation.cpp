#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sstream>
#include <ITK-4.8/itkEuler3DTransform.h>

#include "NnSimilarityMetric.h"
#include "MutualInfoSimilarityMetric.h"


#include "tclap/CmdLine.h"
#include "nlopt.hpp"
#include "nloptfunction.h"

#include "itkMesh.h"
#include "itkTransformMeshFilter.h"
#include "itkEuler3DTransform.h"
typedef itk::ImageFileReader<ImageT> ImageFileReaderT;
typedef itk::Mesh < PixelT, 3> MeshT;

int main(int argc, char *argv[]) {
    std::cout << "Registration Evaluation" << std::endl;

    TCLAP::CmdLine
    cmd("Evaluate the registration");

    TCLAP::ValueArg<std::string>
            fixedFileArg("f", "fixedImage", "Fixed Image", true, " ", "FILE");
    cmd.add(fixedFileArg);

    TCLAP::ValueArg<std::string>
            movingFileArg("m", "movingImage", "Moving Image", true, " ", "FILE");
    cmd.add(movingFileArg);

    TCLAP::ValueArg<std::string>
            networkFileArg("n", "network", "Network", false, " ", "FILE");
    cmd.add(networkFileArg);

    TCLAP::ValueArg<char>
            typeOfSimilarityArg("s", "similarity", "Type of Similarity m = MI c = CNN", true, 't', "CHAR");
    cmd.add(typeOfSimilarityArg);

    TCLAP::UnlabeledMultiArg<double> transformationArg("initialTransformation",
            "Transformation Vector", true, "VECTOR");
    cmd.add(transformationArg);

    cmd.parse(argc,argv);
    const char *netpath = (networkFileArg.getValue()).c_str();

    ImageT::Pointer fixedImage = ImageT::New();
    ImageT::Pointer movingImage = ImageT::New();

    ImageFileReaderT::Pointer fixedImageReader = ImageFileReaderT::New();
    ImageFileReaderT::Pointer movingImageReader = ImageFileReaderT::New();
    
    fixedImageReader->SetFileName(fixedFileArg.getValue());
    movingImageReader->SetFileName(movingFileArg.getValue());

    fixedImage = fixedImageReader->GetOutput();
    movingImage = movingImageReader->GetOutput();
    fixedImage->Update();
    movingImage->Update();

    NloptDataStruct dataStruct;
    dataStruct.m_fixedImage = fixedImage;
    dataStruct.m_movingImage = movingImage;
    
    // TODO : How can I switch between the two??
    MutualInfoSimilarityMetric miSimilarityMetric;
    NnSimilarityMetric nnSimilarityMetric;
    
    if (typeOfSimilarityArg.getValue() == 'm') {
        std::cout << "Running for MI" << std::endl;
        dataStruct.m_similarityMetric = &miSimilarityMetric;
    } else if (typeOfSimilarityArg.getValue() == 'c') {
        
        std::cout << "Running for CNN" << std::endl;

        nnSimilarityMetric.setLuaState();
        nnSimilarityMetric.setFixedImage(fixedImage);
        nnSimilarityMetric.setNetwork(netpath);
        //ImageT::SizeType patchSize = similarityMetric.getPatchSize();

        std::vector<int> gridSpacing(3,0);
        gridSpacing[0] = 32;
        gridSpacing[1] = 32;
        gridSpacing[2] = 16;
        nnSimilarityMetric.setUniformGrid(gridSpacing);
                
        nnSimilarityMetric.initializeTensors();
        dataStruct.m_similarityMetric = &nnSimilarityMetric;
        
    } else 
    {
        std::cerr << "Not a valid similarity metric" << std::endl;
    }

    std::vector<double> transformationVector;
    std::vector<double> initialTransformationVector;
    transformationVector = transformationArg.getValue();
    initialTransformationVector = transformationArg.getValue();
    
    std::vector<double> optimisationLowerBounds(6);
    std::vector<double> optimisationUpperBounds(6);
    std::vector<double> optimisationStepSize(6);
    
    optimisationLowerBounds[0] = transformationVector[0]-1;
    optimisationLowerBounds[1] = transformationVector[1]-1;
    optimisationLowerBounds[2] = transformationVector[2]-1; 
    optimisationLowerBounds[3] = transformationVector[3]-30;
    optimisationLowerBounds[4] = transformationVector[4]-30;
    optimisationLowerBounds[5] = transformationVector[5]-30;
    
    optimisationUpperBounds[0] = transformationVector[0] + 1;
    optimisationUpperBounds[1] = transformationVector[1] + 1;
    optimisationUpperBounds[2] = transformationVector[2] + 1; 
    optimisationUpperBounds[3] = transformationVector[3] + 30;
    optimisationUpperBounds[4] = transformationVector[4] + 30;
    optimisationUpperBounds[5] = transformationVector[5] + 30;
    
    optimisationStepSize[1] = 0.5;
    optimisationStepSize[0] = 0.5;
    optimisationStepSize[2] = 0.5; 
    optimisationStepSize[3] = 15;
    optimisationStepSize[4] = 15;
    optimisationStepSize[5] = 15;
    
    
    
    nlopt::opt optimisation(nlopt::LN_BOBYQA,6);
    optimisation.set_lower_bounds(optimisationLowerBounds);
    optimisation.set_upper_bounds(optimisationUpperBounds);
    optimisation.set_initial_step(optimisationStepSize);
    optimisation.set_max_objective(nloptSimilarityFunction,&dataStruct);
    optimisation.set_xtol_rel(0.01);
    double maximumSimilarity;
    
    std::cout << "Optimisation bounds: " << std::endl;
    for (int i=0;i<6;i++)
    {
     std::cout << optimisationLowerBounds[i] << ", " <<
                  optimisationUpperBounds[i] <<  std::endl;
    }
    
    
    optimisation.optimize(transformationVector,maximumSimilarity);
    
    
    std::cout << "Optimal transformation is: " ;
    for (int i=0; i<6;i++)
        std::cout << -transformationArg.getValue()[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Obtained transformation is: " ;
    for (int i=0; i<6;i++)
        std::cout << transformationVector[i] << " ";
    std::cout << std::endl;

    // Create a Grid for the evaluation of the TRE
    
    MeshT::Pointer treGrid = MeshT::New();
    
    
    ImageT::SizeType imageSize;
    ImageT::SpacingType imageSpacing;
    ImageT::DirectionType imageDirection;
    ImageT::PointType imageOrigin;
    
    imageSize = fixedImage->GetLargestPossibleRegion().GetSize();
    imageSpacing = fixedImage->GetSpacing();
    //imageDirection =  fixedImage->GetDirection();
    imageOrigin = fixedImage->GetOrigin();
    std::vector<int> gridSpacing(3,0);
    std::fill(gridSpacing.begin(),gridSpacing.end(),80);
    
   
        unsigned int x, y, z;
        int iPoint = 0;
        z = 0;

        while (z <= imageSize[2]) {
            y = 0;
            while (y <= imageSize[1]) {
                x = 0;
                while (x <= imageSize[0]) {
                    ImageT::IndexType currentGridIndex;
                    ImageT::PointType currentGridPoint;
                    currentGridIndex[0] = x;
                    currentGridIndex[1] = y;
                    currentGridIndex[2] = z;
                    fixedImage->TransformIndexToPhysicalPoint(currentGridIndex,currentGridPoint);
                    //evaluationGrid.push_back(currentGridPoint);
                    treGrid->SetPoint(iPoint,currentGridPoint);
                    iPoint++;
                    x += gridSpacing[0];
                    //std::cout << currentGridPoint << std::endl;
                }
                y += gridSpacing[1];
            }
            z += gridSpacing[2];
        }
        
       
    MeshT::Pointer treGridOptimal = MeshT::New();
    MeshT::Pointer treGridObtained = MeshT::New();
    typedef itk::Euler3DTransform<double> EulerTransformT;
    typedef itk::TransformMeshFilter<MeshT,MeshT,EulerTransformT> TransformMeshT;
    TransformMeshT::Pointer meshTransformerObtained = TransformMeshT::New();
    TransformMeshT::Pointer meshTransformerOptimal  = TransformMeshT::New();
    EulerTransformT::Pointer obtainedTransform = EulerTransformT::New();
    EulerTransformT::Pointer optimalTransform = EulerTransformT::New();
    ImageT::IndexType centerImageSpace;
    ImageT::PointType centerWorldSpace;
    for (int iDim=0; iDim< 3; iDim++)
       centerImageSpace[iDim] =  imageSize[iDim]/2;
    fixedImage->TransformIndexToPhysicalPoint(centerImageSpace,centerWorldSpace);
    
    meshTransformerObtained->SetInput(treGrid);
    meshTransformerOptimal->SetInput(treGrid);
        
    EulerTransformT::TranslationType translation;
    std::vector<double> rotation(3,0);
    rotation[0] = transformationVector[0];
    rotation[1] = transformationVector[1];
    rotation[2] = transformationVector[2];
    translation[0] = transformationVector[3];
    translation[1] = transformationVector[4];
    translation[2] = transformationVector[5];
    obtainedTransform->SetTranslation(translation);
    obtainedTransform->SetRotation(rotation[0],rotation[1],rotation[2]);
    
    obtainedTransform->SetCenter(centerWorldSpace);
    meshTransformerObtained->SetTransform(obtainedTransform);
    treGridObtained =  meshTransformerObtained->GetOutput();
    treGridObtained->Update();
    
    
    rotation[0] = initialTransformationVector[0];
    rotation[1] = initialTransformationVector[1];
    rotation[2] = initialTransformationVector[2];
    translation[0] = initialTransformationVector[3];
    translation[1] = initialTransformationVector[4];
    translation[2] = initialTransformationVector[5];
    optimalTransform->SetTranslation(translation);
    optimalTransform->SetRotation(rotation[0],rotation[1],rotation[2]);
    optimalTransform->SetCenter(centerWorldSpace);
    meshTransformerOptimal->SetTransform(optimalTransform);
    treGridOptimal =  meshTransformerOptimal->GetOutput();
    treGridOptimal->Update();
    std::vector<float> treVector;
    for (unsigned int i=0; i<treGridObtained->GetNumberOfPoints();i++)
    {
        double thisPointSquaredError;
        thisPointSquaredError =
          sqrt(pow(treGridOptimal->GetPoint(i)[0] - treGridObtained->GetPoint(i)[0],2) +
          pow(treGridOptimal->GetPoint(i)[1] - treGridObtained->GetPoint(i)[1],2) +
          pow(treGridOptimal->GetPoint(i)[2] - treGridObtained->GetPoint(i)[2],2)) ;
          treVector.push_back(thisPointSquaredError);
        
          std::cout << i << " " << treGridOptimal->GetPoint(i) <<
                       " -> " <<   treGridObtained->GetPoint(i) << " : "
                              <<   thisPointSquaredError<< std::endl; 
        
    }
    
    
    
    
    std::cout << std::endl;

    return (0);
}
