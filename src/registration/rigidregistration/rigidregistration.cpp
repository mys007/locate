#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sstream>
#include <TH/TH.h>

extern "C"{
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}
#include "luaT.h"
#include "itkImageRegionConstIterator.h"

#include "NnSimilarityMetric.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "typedefinitions.h"

#include "itkEuler3DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"

#include "tclap/CmdLine.h"
#include "nlopt.hpp"
#include "nloptfunction.h"

typedef itk::ImageFileReader<ImageT> ImageFileReaderT;
typedef itk::ExtractImageFilter<ImageT,ImageT> ExtractFilterT;

int main (int argc, char *argv[])
{
    std::cout << "Reading parameters: " <<std::endl;
    //DATAPATH IS HARD CODED FOR SIMPLICITY
    //std::string 
    //dataPath("/home/beckerbe/Multimodal/images/IXI_R/");
    //////////////////////////////////////////////////////////////
    TCLAP::CmdLine 
    cmd("Evaluate the similarity metric on a grid of translations or rotations");
    
    TCLAP::ValueArg<std::string> 
            fixedFileArg ("f", "fixedImage", "Fixed Image", true," ", "FILE"); 
    cmd.add(fixedFileArg);
    
    TCLAP::ValueArg<std::string> 
            movingFileArg ("m", "movingImage", "Moving Image", true," ", "FILE");
    cmd.add(movingFileArg);
    
    TCLAP::ValueArg<std::string> 
            networkFileArg ("n", "network", "Network", true," ", "FILE");
    cmd.add(networkFileArg);

        
    cmd.parse(argc,argv);
    // CHANGE HERE THE IMAGES AND THE NETWORKS.
    const char *netpath =  (networkFileArg.getValue()).c_str();
    std::string 
        filenameFixed (fixedFileArg.getValue());
    std::string
        filenameMoving(movingFileArg.getValue());
    
   
    std::cout << "Reading images " << std::endl ;
    
    ImageT::Pointer fixedImage = ImageT::New();
       
    ImageFileReaderT::Pointer 
    imageFileReaderFixed = ImageFileReaderT::New();
    imageFileReaderFixed->SetFileName(filenameFixed);
    fixedImage = imageFileReaderFixed->GetOutput();
    fixedImage->Update();
       
    ImageT::Pointer movingImage = ImageT::New();
    
    
    ImageFileReaderT::Pointer 
           imageFileReaderMoving = ImageFileReaderT::New();
    imageFileReaderMoving->SetFileName(filenameMoving);
    movingImage = imageFileReaderMoving->GetOutput();
    movingImage->Update();

    ImageT::SizeType  imageSize;
    imageSize = fixedImage->GetLargestPossibleRegion().GetSize();
    
    // Here we setup the similarity metric: 
    NnSimilarityMetric similarityMetric;
    similarityMetric.setLuaState();    
    similarityMetric.setNetwork(netpath);
    ImageT::SizeType patchSize = similarityMetric.getPatchSize();    
    
    
    // Here we define the points where the similarity metric is evaluated. The
    // patches are extracted setting as center each point of the evaluationGrid
    // vector
    
        
    // THIS IS USED TO DEFINE THE GRID USED FOR THE EVALUATION
    // OF THE METRIC
    std::vector<int> gridStep(3,0);
    gridStep[0] = 32;
    gridStep[1] = 32;
    gridStep[2] = 16;    
    
    std::vector<ImageT::IndexType> evaluationGrid;
    std::vector<int> upperGridLimits(3,0);
    std::vector<int> lowerGridLimits(3,0);
    
    upperGridLimits[0] = imageSize[0] - patchSize[0];
    upperGridLimits[1] = imageSize[1] - patchSize[1];
    upperGridLimits[2] = imageSize[2] - patchSize[1];
    
    lowerGridLimits[0] = 0;
    lowerGridLimits[1] = 0;
    lowerGridLimits[2] = 0;
    
    
    int x,y,z;
    
    z = lowerGridLimits[2];
    
    while (z <= upperGridLimits[2])
    {
        y = lowerGridLimits[1];
        while ( y <= upperGridLimits[1])
        {
            x = lowerGridLimits[0];
            while ( x <= upperGridLimits[0])
            {
                ImageT::IndexType currentGridPoint;
                currentGridPoint[0] = x;
                currentGridPoint[1] = y;
                currentGridPoint[2] = z;
                evaluationGrid.push_back(currentGridPoint);
                x += gridStep[0];
                
            }
            y += gridStep[1];
        }
        z += gridStep[2];
    }
    
    
    //ONLY INITIALIZE TENSORS AFTER SETTING UP THE GRID AND LOADING THE NET. THERE
    //ARE NO DEFAULTS!!

    similarityMetric.setGrid(evaluationGrid);
    similarityMetric.setFixedImage(fixedImage);
    similarityMetric.initializeTensors();
    //
    
    std::vector<double> optimisationLowerBounds(6);
    std::vector<double> optimisationUpperBounds(6);
    std::vector<double> optimisationStepSize(6);

    optimisationLowerBounds[0] = -1;
    optimisationLowerBounds[1] = -1;
    optimisationLowerBounds[2] = -1; 
    optimisationLowerBounds[3] = -30;
    optimisationLowerBounds[4] = -30;
    optimisationLowerBounds[5] = -30;
    
    optimisationUpperBounds[0] = 1;
    optimisationUpperBounds[1] = 1;
    optimisationUpperBounds[2] = 1; 
    optimisationUpperBounds[3] = 30;
    optimisationUpperBounds[4] = 30;
    optimisationUpperBounds[5] = 30;
    
    optimisationStepSize[1] = 0.5;
    optimisationStepSize[0] = 0.5;
    optimisationStepSize[2] = 0.5; 
    optimisationStepSize[3] = 15;
    optimisationStepSize[4] = 15;
    optimisationStepSize[5] = 15;
    
    NloptDataStruct dataStructure;
    dataStructure.m_fixedImage = fixedImage;
    dataStructure.m_movingImage = movingImage;
    dataStructure.m_similarityMetric = &similarityMetric;
    
    nlopt::opt optimisation(nlopt::LN_BOBYQA,6);
    optimisation.set_lower_bounds(optimisationLowerBounds);
    optimisation.set_upper_bounds(optimisationUpperBounds);
    optimisation.set_initial_step(optimisationStepSize);
    optimisation.set_max_objective(nloptSimilarityFunction,&dataStructure);
    optimisation.set_xtol_rel(0.01);
    double maximumSimilarity;
    std::vector<double> transformationVector(6);
    std::fill(transformationVector.begin(),
              transformationVector.end(),0);
    
    optimisation.optimize(transformationVector,maximumSimilarity);
    
    std::cout << "Optimal transformation is: " ;
    for (int i=0; i<6;i++)
        std::cout << transformationVector[i] << " ";
    
    std::cout << std::endl;
            
    
    
    
}


