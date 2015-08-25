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

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}
#include "luaT.h"
#include "itkImageRegionConstIterator.h"

#include "NnSimilarityMetric.h"
#include "MutualInfoSimilarityMetric.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "typedefinitions.h"

#include "itkEuler3DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"

#include "tclap/CmdLine.h"

typedef itk::ImageFileReader<ImageT> ImageFileReaderT;
typedef itk::ExtractImageFilter<ImageT, ImageT> ExtractFilterT;

int main(int argc, char *argv[]) {
    std::cout << "Reading parameters: " << std::endl;
    //DATAPATH IS HARD CODED FOR SIMPLICITY
    //std::string 
    //dataPath("/home/beckerbe/Multimodal/images/IXI_R/");
    //////////////////////////////////////////////////////////////
    TCLAP::CmdLine
    cmd("Evaluate the similarity metric on a grid of translations or rotations");

    TCLAP::ValueArg<std::string>
            fixedFileArg("f", "fixedImage", "Fixed Image", true, " ", "FILE");
    cmd.add(fixedFileArg);

    TCLAP::ValueArg<std::string>
            movingFileArg("m", "movingImage", "Moving Image", true, " ", "FILE");
    cmd.add(movingFileArg);

    TCLAP::ValueArg<std::string>
            networkFileArg("n", "network", "Network", false, " ", "FILE");
    cmd.add(networkFileArg);

    TCLAP::ValueArg<std::string>
            outputFileArg("o", "outputImage", "Output", true, " ", "FILE");
    cmd.add(outputFileArg);

    TCLAP::ValueArg<char>
            similarityTypeArg("s", "similarityType",
            "Similarity Metric to evaluate (m = Mutual Info) (c = CNN)",
            true, 't', "CHAR");
    cmd.add(similarityTypeArg);

    TCLAP::UnlabeledMultiArg<double>
            initialTransformationArg("initialTransformationVector",
            "Transformation Vector (rot,trans) ", true, "vector");
    cmd.add(initialTransformationArg);

    cmd.parse(argc, argv);
    
     // Read images
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
    
    ImageT::SizeType imageSize;
    imageSize = fixedImage->GetLargestPossibleRegion().GetSize();
    
    if (similarityTypeArg.getValue() == 'm')
        MutualInfoSimilarityMetric similarityMetric;
    else if (similarityTypeArg.getValue() == 'c') {
        
        const char *netpath =  (networkFileArg.getValue()).c_str();
        NnSimilarityMetric similarityMetric;
        similarityMetric.setLuaState();    
        similarityMetric.setNetwork(netpath);
       
        //ONLY INITIALIZE TENSORS AFTER SETTING UP THE GRID AND LOADING THE NET. THERE
        //ARE NO DEFAULTS!!
        similarityMetric.setFixedImage(fixedImage);
        std::vector<int> gridSpacing(3,0);
        gridSpacing[0] = 32;
        gridSpacing[1] = 32;
        gridSpacing[0] = 16;
        similarityMetric.setUniformGrid(gridSpacing);
        similarityMetric.initializeTensors();


    } else
        std::cerr << "Unknown similarity metric type" << std::endl;



}


