#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sstream>

#include "NnSimilarityMetric.h"
#include "MutualInfoSimilarityMetric.h"


#include "tclap/CmdLine.h"
#include "nlopt.hpp"
#include "nloptfunction.h"

typedef itk::ImageFileReader<ImageT> ImageFileReaderT;

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
    transformationVector = transformationArg.getValue();
    
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
        std::cout << transformationVector[i] << " ";
    
    std::cout << std::endl;
    

    return (0);
}
