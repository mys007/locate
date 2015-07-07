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
#include "itkStatisticsImageFilter.h"

#include "boost/filesystem.hpp"
#include <boost/function.hpp>
#include <vector>
#include <algorithm>

typedef double PixelType;
typedef itk::Image<PixelType,3> Image3DType;
//typedef itk::Image<PixelType,2> Image2DType;
typedef itk::Image<unsigned char,2> Image2DType;
typedef itk::ImageFileReader<Image3DType> ImageFileReaderType;
typedef itk::ImageFileWriter<Image2DType> ImageFileWriterType;
typedef itk::ExtractImageFilter<Image3DType,Image2DType> ExtractImageFilterType;
typedef itk::ResampleImageFilter<Image3DType,Image3DType> ImageResampleFilterType;
typedef itk::LinearInterpolateImageFunction<Image3DType, double>    InterpolatorType;



template <typename ImageType> void stddev(const typename ImageType::Pointer image)
{
	typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
	typename StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
	statisticsImageFilter->SetInput(image);
	statisticsImageFilter->Update();

	std::cout << "Mean: " << (double)statisticsImageFilter->GetMean() << std::endl;
	std::cout << "Std.: " << (double)statisticsImageFilter->GetSigma() << std::endl;
	std::cout << "Min: " << (double)statisticsImageFilter->GetMinimum() << std::endl;
	std::cout << "Max: " << (double)statisticsImageFilter->GetMaximum() << std::endl;
}


//ARGS: srcdir destdir numslices
int main(int argc, char *argv[])
{
	const std::string target_path(argv[1]);
	const int nSlices = argc>3 ? strtol(argv[3],NULL,0) : 1;
	Image3DType::Pointer referenceImage = NULL;
	
    std::vector<boost::filesystem::path> v;
    std::copy(boost::filesystem::directory_iterator(target_path), boost::filesystem::directory_iterator(), std::back_inserter(v));
    std::sort(v.begin(), v.end());

    for (std::vector<boost::filesystem::path>::const_iterator it(v.begin()), it_end(v.end()); it != it_end; ++it)
    {
		if (it->extension() == ".gz")
		{
			ImageFileReaderType::Pointer  imageReader = ImageFileReaderType::New();
			ImageFileWriterType::Pointer  imageWriter = ImageFileWriterType::New();
			ExtractImageFilterType::Pointer  extractSliceFilter = ExtractImageFilterType::New();

			std::string currentFileName(it->filename().string());
			std::cout << "Reading from: "<< argv[1]+currentFileName << std::endl;
			imageReader->SetFileName(argv[1]+currentFileName);

			if (it->filename().string().find("-T1.nii.gz") != std::string::npos)
			{
				std::cout << "Image set as reference" << std::endl;
				referenceImage = Image3DType::New();
				referenceImage = imageReader->GetOutput();
				referenceImage->Update();
				referenceImage->DisconnectPipeline();
			}

			InterpolatorType::Pointer interpolator = InterpolatorType::New();
			ImageResampleFilterType::Pointer imageResampleFilter = ImageResampleFilterType::New();
			imageResampleFilter->SetInput(imageReader->GetOutput());
			imageResampleFilter->SetReferenceImage(referenceImage); //align w.r.t. physical coordinates
			imageResampleFilter->UseReferenceImageOn();
			imageResampleFilter->SetDefaultPixelValue(-1);
			imageResampleFilter->SetInterpolator(interpolator);
			imageResampleFilter->Update();
			//imageReader->Update();

			//Image3DType::PointType origin = imageReader->GetOutput()->GetOrigin();
			//Image3DType::SpacingType spacing = imageReader->GetOutput()->GetSpacing();
			//Image3DType::SizeType size_pxl = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
			//std::cout << size_pxl[0] << " " << size_pxl[1] << " " << spacing[0] << " " << spacing[1] << " " << origin[0] << " " << origin[1] << std::endl;

			stddev<Image3DType>(imageReader->GetOutput());
						
			for (int iDim=0; iDim<3; ++iDim)
			{
				Image3DType::RegionType inputRegion = imageResampleFilter->GetOutput()->GetLargestPossibleRegion();
				Image3DType::SizeType   imageSize   = inputRegion.GetSize();
				Image3DType::IndexType  start = inputRegion.GetIndex();

				int step = imageSize[iDim]/(nSlices+1);
				imageSize[iDim] = 0;

				for (int iSlice=0; iSlice<nSlices; ++iSlice)
				{
					Image3DType::RegionType  sliceRegion;
					sliceRegion.SetSize(imageSize);
					start[iDim] = (iSlice+1) * step;
					sliceRegion.SetIndex(start);

					extractSliceFilter->SetInput(imageResampleFilter->GetOutput());
					extractSliceFilter->SetExtractionRegion(sliceRegion);
					extractSliceFilter->SetDirectionCollapseToIdentity();
					extractSliceFilter->Update();

					std::stringstream outputStream;
					outputStream << argv[2] << "Dim_" << iDim << "Slice_" << iSlice << "_" << currentFileName << ".png";
					std::cout << "Writing to :  " << outputStream.str() << std::endl;

					//stddev<Image2DType>(extractSliceFilter->GetOutput());

					imageWriter->SetFileName(outputStream.str());
					imageWriter->SetInput(extractSliceFilter->GetOutput());
					imageWriter->Update();
				}





				/*

				for (int j=0; j<3; ++j)
				{
				center[j] = imageSize[j]/2;
				}
				start[iDim] = center[iDim];
				imageSize[iDim] = 0;

				Image3DType::RegionType  sliceRegion;
				sliceRegion.SetSize(imageSize);
				sliceRegion.SetIndex(start);

				extractSliceFilter->SetInput(imageResampleFilter->GetOutput());
				extractSliceFilter->SetExtractionRegion(sliceRegion);
				extractSliceFilter->SetDirectionCollapseToIdentity();
				extractSliceFilter->Update();

				std::stringstream outputStream;
				outputStream << argv[2] <<"Slice_" << iDim << "_"<<currentFileName;// << ".png";
				std::cout << "Writing to :  " << outputStream.str() << std::endl;

				stddev<Image2DType>(extractSliceFilter->GetOutput());

				imageWriter->SetFileName(outputStream.str());
				imageWriter->SetInput(extractSliceFilter->GetOutput());
				imageWriter->Update();*/
			}
		}

    	
	}

}
