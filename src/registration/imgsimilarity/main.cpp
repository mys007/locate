#include "NnSimilarityMetric.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "typedefinitions.h"

//typedef itk::Image < double > ImageT;
typedef itk::ImageFileReader < ImageT > FileReaderT;

#include <TH/TH.h>

extern "C"{
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}
#include "luaT.h"


int main(int argc, char ** argv) 
{
    NnSimilarityMetric similarityMetric;

    FileReaderT::Pointer fileReaderFixed = FileReaderT::New();
    fileReaderFixed->SetFileName("/home/beckerbe/Multimodal/images/IXI_R/R_IXI002-Guys-0828-T1.nii.gz");
    //fileReaderFixed->SetFileName("/home/beckerbe/Multimodal/images/IXI_R/R_IXI012-HH-1211-T1.nii.gz");
    ImageT::Pointer fixedImage = ImageT::New();
    fixedImage = fileReaderFixed->GetOutput();
    fixedImage->Update();
    FileReaderT::Pointer fileReaderMoving = FileReaderT::New();
    fileReaderMoving->SetFileName("/home/beckerbe/Multimodal/images/IXI_R/R_IXI002-Guys-0828-T2.nii.gz");
    ImageT::Pointer movingImage = ImageT::New();
    movingImage = fileReaderMoving->GetOutput();
    movingImage->Update();
    
        
    const char *netpath = "/home/beckerbe/Multimodal/networks/new_network_overlap.net";
    similarityMetric.setLuaState();
    similarityMetric.setNetwork(netpath);
    similarityMetric.setFixedImage(fixedImage);
    similarityMetric.setMovingImage(movingImage);
    
    ImageT::SizeType patchSize = similarityMetric.getPatchSize();
    ImageT::SizeType imageSize = fixedImage->GetLargestPossibleRegion().GetSize();
    
    
    std::vector<ImageT::IndexType> evaluationGrid;
    std::vector<int> upperGridLimits(3,0);
    std::vector<int> lowerGridLimits(3,0);
    std::vector<int> gridStep(3,0);
    
    upperGridLimits[0] = imageSize[0] - patchSize[0];
    upperGridLimits[1] = imageSize[1] - patchSize[1];
    upperGridLimits[2] = imageSize[2] - 20;
    
    lowerGridLimits[0] = patchSize[0];
    lowerGridLimits[1] = patchSize[1];
    lowerGridLimits[2] = 20;
    
    gridStep[0] = patchSize[0];
    gridStep[1] = patchSize[1];
    gridStep[2] = 20;
    
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
    

    similarityMetric.setGrid(evaluationGrid);
    similarityMetric.initializeTensors();
    
    similarityMetric.compute();
    
    std::cout << "Similarity: " <<
            similarityMetric.getSimilarityValue() << std::endl;
    
}
