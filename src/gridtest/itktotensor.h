#ifndef ITKTOTENSOR_H_INCLUDED
#define ITKTOTENSOR_H_INCLUDED

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkExtractImageFilter.h"
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <TH/TH.h>

typedef float T;
typedef itk::Image<T,3> ImageT;
typedef itk::ImageFileReader<ImageT> ImageFileReaderT;
typedef itk::ExtractImageFilter<ImageT,ImageT> ExtractFilterT;


bool imageToTensor(ImageT::Pointer image, THFloatTensor* tensor);

bool patchToTensor(ImageT::Pointer image, THFloatTensor* tensor,
                   ImageT::IndexType origin,ImageT::SizeType size);

bool pairOfImagesToTensor(ImageT::Pointer image1, ImageT::Pointer image2,
                          ImageT::SizeType patchSize, THFloatTensor* tensor,
                          int nbPatches);


#endif