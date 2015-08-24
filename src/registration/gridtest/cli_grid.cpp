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

    TCLAP::ValueArg<std::string> 
            outputFileArg ("o", "output", "Output", true," ", "FILE");
    cmd.add(outputFileArg);
     
        TCLAP::ValueArg<char>
        typeOfTransformationArg("t", "type", "Type of transformation(t,r)",true,'t', "CHAR");
    cmd.add(typeOfTransformationArg);
    
    TCLAP::ValueArg<float>
        rangeArg("r", "range", "Range to evaluate (-r -> r)",true,10, "FLOAT");
    cmd.add(rangeArg);
    
    TCLAP::ValueArg<float>
        stepArg("s", "step", "Step size" ,true,1, "FLOAT");
    cmd.add(stepArg);
    
    TCLAP::ValueArg<int>
        dimArg("d", "dimension", "Dimension of evaluation" ,true,1, "INT");
    cmd.add(dimArg);
    
    cmd.parse(argc,argv);
    // CHANGE HERE THE IMAGES AND THE NETWORKS.
    const char *netpath =  (networkFileArg.getValue()).c_str();
    std::string 
        filenameFixed (fixedFileArg.getValue());
    std::string
        filenameMoving(movingFileArg.getValue());
    
    float range     = rangeArg.getValue();
    float step      = stepArg.getValue();
    char transfType = typeOfTransformationArg.getValue();
    int transfDim   = dimArg.getValue();
    
    std::ofstream outputStream;
    std::string outputFile(outputFileArg.getValue());
    
    outputStream.open(outputFile.c_str(),std::ios::trunc);
    std::cout << outputFile << std::endl;
   
 
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

    
    // Evaluating the similarity on a grid of transformations
    typedef itk::Euler3DTransform<double> EulerTransformT;
    EulerTransformT::Pointer eulerTransform = EulerTransformT::New();

    typedef itk::LinearInterpolateImageFunction<ImageT,double> InterpolatorT;
    InterpolatorT::Pointer interpolator = InterpolatorT::New();
    
    typedef itk::ResampleImageFilter<ImageT,ImageT> ResamplerT;
    ResamplerT::Pointer resampler = ResamplerT::New();
    
    resampler->SetInput(movingImage);
    resampler->SetInterpolator(interpolator);
    
    EulerTransformT::TranslationType translation;
    
    ImageT::IndexType centerImageSpace;
    ImageT::PointType centerWorldSpace;
    ImageT::SizeType imgSize;
    imgSize = fixedImage->GetLargestPossibleRegion().GetSize();
    
    
    for (int iDim=0; iDim<3; iDim++)
    {
        centerImageSpace[iDim] = imgSize[iDim] / 2;
    }
    fixedImage->TransformIndexToPhysicalPoint(centerImageSpace, centerWorldSpace);
    eulerTransform->SetCenter(centerWorldSpace);
    
    similarityMetric.setFixedImage(fixedImage);
    std::cout << "Computing: " << std::endl;
    
    
    for (float idx = -range; idx <=range; idx+=step)
    {
        
        translation[0] = 0;
	    translation[1] = 0;
	    translation[2] = 0;
        std::vector <double> rotation(3,0);
        
        if (transfType == 't')
        {
            translation[transfDim] = idx;
        }
        else if (transfType == 'r')
        {
            rotation[transfDim] = idx * 6.28 /360;
        }
        else
        {
            std::cerr << "invalid transformation type" << std::endl;
        }
        
	eulerTransform->SetTranslation(translation);
        eulerTransform->SetRotation(rotation[0],rotation[1],rotation[2]);
        
        resampler->SetTransform(eulerTransform);
        resampler->UseReferenceImageOn();
        resampler->SetReferenceImage(fixedImage);
        resampler->Update();
        
        similarityMetric.setMovingImage(resampler->GetOutput());
        similarityMetric.compute();
        std::cout << idx << " " << similarityMetric.getSimilarityValue() << std::endl;
        outputStream << idx << " " << similarityMetric.getSimilarityValue() << std::endl;
        
    }
    outputStream.close();
    
}


