
#include "itktotensor.h"


bool imageToTensor(ImageT::Pointer image, THFloatTensor* tensor)
{
  std::cout << "Passing image to tensor" << std::endl;
  if (image->ImageDimension != 3) {
    std::cerr << "Input image doesn't have 3 dims: " << image->ImageDimension << std::endl;
    return false;
  }
  
  ImageT::SizeType size = image->GetLargestPossibleRegion().GetSize();
  THFloatTensor_resize3d(tensor, size[2], size[1], size[0]);

    memcpy(THFloatTensor_data(tensor), image->GetBufferPointer(),
     THFloatTensor_nElement(tensor)*sizeof(float));
    //TODO: or just sharing? get inspired by http://docs.mitk.org/2014.10/mitkITKImageImport_8txx_source.html
    return true;
}

bool patchToTensor(ImageT::Pointer image, THFloatTensor* tensor,
                   ImageT::IndexType origin,ImageT::SizeType size)

{
    ImageT::RegionType region;
    region.SetSize(size);
    region.SetIndex(origin);
    itk::ImageRegionConstIterator<ImageT> patchIterator(image,region);
    THFloatTensor_resize3d(tensor, size[2], size[1], size[0]);
    //int offset=0;
   /* while(!patchIterator.IsAtEnd())
    
        memcpy(THFloatTensor_data(tensor+offset), patchIterator,
                THFloatTensor_nElement(tensor+offset)*sizeof(float));
       offset ++ ;
       patchIterator++;
    }*/

    return true;
}

bool pairOfImagesToTensor(ImageT::Pointer image1, ImageT::Pointer image2,
                          ImageT::SizeType patchSize, THFloatTensor* tensor,
                          int nbPatches)
{

    ImageT::Pointer     patch = ImageT::New();
    ImageT::Pointer     patch2 = ImageT::New();
    ImageT::IndexType   patchStart;
    ImageT::IndexType   imageCenter;

    for (int iPatch = 0; iPatch<nbPatches; iPatch++)
    {
    patchStart[0] = 150;
    patchStart[1] = 150;
    patchStart[2] = 74;

    ImageT::RegionType  patchRegion(patchStart,patchSize);
    ExtractFilterT::Pointer 
                extractFilter = ExtractFilterT::New();
            
    extractFilter->SetExtractionRegion(patchRegion);
    extractFilter->SetInput(image1);
    patch = extractFilter->GetOutput();
    patch->Update();    
    
    // Copy first patch to the first block of memory
    memcpy(THFloatTensor_data(tensor) +
           (iPatch * patchSize[0] * patchSize[1] * 2),
           patch->GetBufferPointer(),
           patchSize[0] * patchSize[1] *sizeof(float));
   
     ImageT::IndexType patchStart2;
     
     patchStart2 = patchStart;
     patchStart2[1] = patchStart[1]+ iPatch - 5;  
     ImageT::RegionType  patchRegion2(patchStart2,patchSize);
     ExtractFilterT::Pointer 
                extractFilter2 = ExtractFilterT::New();
     
     extractFilter2->SetInput(image2);
     extractFilter2->SetExtractionRegion(patchRegion2);
     patch2 = extractFilter2->GetOutput();
     patch2->Update(); 
     // Copy second patch to the second block of memory
     memcpy(THFloatTensor_data(tensor) + 
            (patchSize[0] * patchSize[1]) +
            (iPatch * patchSize[0] * patchSize[1] * 2) ,
            patch2->GetBufferPointer(), patchSize[0] * patchSize[1] *sizeof(float));
    
    }


    
    return true;
}

