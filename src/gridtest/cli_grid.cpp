#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <TH/TH.h>

extern "C"{
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}
#include "luaT.h"
#include "itkImageRegionConstIterator.h"

#include "itktotensor.h"

typedef float T;
typedef itk::Image<T,3> ImageT;
typedef itk::ImageFileReader<ImageT> ImageFileReaderT;
typedef itk::ExtractImageFilter<ImageT,ImageT> ExtractFilterT;



int main ()
{
	
    std::cout << "Reading images " << std::endl ;
    std::string filename;

    ImageT::Pointer fixedImage = ImageT::New();
    filename = "/home/beckerbe/Multimodal/images/IXI_R/R_IXI002-Guys-0828-T2.nii.gz";
    
    ImageFileReaderT::Pointer 
           imageFileReaderFixed = ImageFileReaderT::New();
    imageFileReaderFixed->SetFileName(filename);
    fixedImage = imageFileReaderFixed->GetOutput();
    fixedImage->Update();
   

    ImageT::Pointer movingImage = ImageT::New();
    filename = "/home/beckerbe/Multimodal/images/IXI_R/R_IXI002-Guys-0828-T1.nii.gz";
    ImageFileReaderT::Pointer 
           imageFileReaderMoving = ImageFileReaderT::New();
    imageFileReaderMoving->SetFileName(filename);
    movingImage = imageFileReaderMoving->GetOutput();
    movingImage->Update();

    ImageT::SizeType fixedImageSize;
    fixedImageSize = fixedImage->GetLargestPossibleRegion().GetSize();
    std::cout << "Image size: "<< fixedImageSize << std::endl;
    
    ImageT::SizeType patchSize;
    patchSize[0]= 64;
    patchSize[1]= 64;
    patchSize[2] = 1;

    int nbPatches = 11;
    
    THFloatTensor *pairTensor =
       THFloatTensor_newWithSize4d(nbPatches,2,patchSize[1],patchSize[0]);
    
    pairOfImagesToTensor(fixedImage,movingImage,
                         patchSize,pairTensor,nbPatches);
    
    THFloatTensor *similarityTensor = THFloatTensor_newWithSize2d(nbPatches,1);
    
    lua_State * L;
 
    L = luaL_newstate();
    luaL_openlibs(L);
    if (luaL_dofile(L, "../script.lua")) {
      std::cerr << "Could not load file: " << lua_tostring(L, -1) << std::endl; 
        lua_close(L);
        return 1;
    }
    
    const char * netpath = "/home/beckerbe/Multimodal/networks/network_scratch_rot_sca.net";
    
    lua_getglobal(L, "loadNetwork");
    lua_pushstring(L, netpath);
    if (lua_pcall(L, 1, 0, 0) != 0) {
        std::cerr << "Error calling loadNetwork: " << lua_tostring(L, -1) << std::endl;
        lua_close(L);        
        return -1;
    }

    
    lua_getglobal(L, "forward");
    luaT_pushudata(L, pairTensor, "torch.FloatTensor");
    luaT_pushudata(L, similarityTensor, "torch.FloatTensor");    
    if (lua_pcall(L, 2, 0, 0) != 0) {
        std::cerr << "Error calling loadNetwork: " << lua_tostring(L, -1) << std::endl;
        lua_close(L);        
        return -1;
    }
    
    for (int i=0; i<nbPatches; ++i)
      std::cout << "Out: " << THFloatTensor_data(similarityTensor)[i] << std::endl;    
    
    lua_close(L);
    return 0;
        
     THFloatTensor_free(similarityTensor);
     THFloatTensor_free(pairTensor);
    
    
}


