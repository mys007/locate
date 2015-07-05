// Extraction of the center slice from 3D images.
// Written by Benjamin Gutierrez Becker. Technische Universitaet Muenchen
// Please do not redistribute without previous consent of the author!
// Contact : gutierrez.becker@gmail.com ingutbecker@gmail.com

#include <stdio.h>
#include <stdlib.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>
#include <itkResampleImageFilter.h>
#include "itkLinearInterpolateImageFunction.h"

#include "boost/filesystem.hpp"
#include <boost/function.hpp>

typedef double PixelType;
typedef itk::Image<PixelType,3> Image3DType;
typedef itk::Image<PixelType,2> Image2DType;
typedef itk::ImageFileReader<Image3DType> ImageFileReaderType;
typedef itk::ImageFileWriter<Image2DType> ImageFileWriterType;
typedef itk::ExtractImageFilter<Image3DType,Image2DType> ExtractImageFilterType;
typedef itk::ResampleImageFilter<Image3DType,Image3DType> ImageResampleFilterType;
typedef itk::LinearInterpolateImageFunction<Image3DType, double>    InterpolatorType;

int main(int argc, char *argv[])
{
	const std::string target_path(argv[1]);
	boost::filesystem::directory_iterator end_itr;
	std::string referenceImageName;
	
	ImageFileReaderType::Pointer  imageReaderReference = ImageFileReaderType::New();
	referenceImageName = target_path + "IXI002-Guys-0828-T1.nii.gz";
	imageReaderReference->SetFileName(referenceImageName);
	Image3DType::Pointer  referenceImage = Image3DType::New();
	referenceImage = imageReaderReference->GetOutput();
	referenceImage->Update();
	referenceImage->DisconnectPipeline();

	for (boost::filesystem::directory_iterator iDir(target_path); iDir!= end_itr; ++iDir)
	{
	if (iDir->path().extension() == ".gz")
		{
			ImageFileReaderType::Pointer  imageReader = ImageFileReaderType::New();
			ImageFileWriterType::Pointer  imageWriter = ImageFileWriterType::New();
			ExtractImageFilterType::Pointer  extractSliceFilter = ExtractImageFilterType::New();

			std::string currentFileName(iDir->path().filename().string()) ;
			std::cout << "Reading from: "<< argv[1]+currentFileName << std::endl;
			imageReader->SetFileName(argv[1]+currentFileName);
			InterpolatorType::Pointer interpolator = InterpolatorType::New();
			ImageResampleFilterType::Pointer imageResampleFilter = ImageResampleFilterType::New();
			imageResampleFilter->SetInput(imageReader->GetOutput());
			imageResampleFilter->SetReferenceImage(referenceImage);
			imageResampleFilter->UseReferenceImageOn();
			imageResampleFilter->SetDefaultPixelValue(0);
			imageResampleFilter->SetInterpolator(interpolator);
			imageResampleFilter->Update();
			//imageReader->Update();
						
			for (int iSlice=0; iSlice<3; ++iSlice)
			{
				Image3DType::RegionType inputRegion = imageResampleFilter->GetOutput()->GetLargestPossibleRegion();
				Image3DType::SizeType   imageSize   = inputRegion.GetSize();
				Image3DType::IndexType  start = inputRegion.GetIndex();
				Image3DType::IndexType   center;
				for (int j=0; j<3; ++j)
				{
				center[j] = imageSize[j]/2;
				}
				start[iSlice] = center[iSlice];
				imageSize[iSlice] = 0;

				Image3DType::RegionType  sliceRegion;
				sliceRegion.SetSize(imageSize);
				sliceRegion.SetIndex(start);

				extractSliceFilter->SetInput(imageResampleFilter->GetOutput());
				extractSliceFilter->SetExtractionRegion(sliceRegion);
				extractSliceFilter->SetDirectionCollapseToIdentity();
				extractSliceFilter->Update();

				std::stringstream outputStream;
				outputStream << argv[2] <<"Slice_" << iSlice << "_"<<currentFileName ;
				std::cout << "Writing to :  " << outputStream.str() << std::endl;
				

				imageWriter->SetFileName(outputStream.str());
				imageWriter->SetInput(extractSliceFilter->GetOutput());
				imageWriter->Update();
			}
		}

    	
	}

}
