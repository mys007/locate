#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <TH/TH.h>

extern "C" {
void getSourceBox(float axis1, float axis2, float axis3, float angle, float scale, THFloatTensor* dest);
void transformVolume(float axis1, float axis2, float axis3, float angle, float scale, float center1, float center2, float center3, THFloatTensor* src, THFloatTensor* dst);
}

// General note: all coordinates/sizes/.. need to be flipped between Torch and ITK, they index their dimensions in the opposite way. But memory is indexed the same way:).


#include <itkSimilarity3DTransform.h>
#include <itkResampleImageFilter.h>

typedef float T;
typedef itk::Similarity3DTransform<T> Similarity3DTransformT;
typedef itk::Image<T,3> ImageT;

// compute inverse transformation of a axis-aligned box centered at (0,0,0) with vertices at points like (1,1,1) 
void getSourceBox(float axis1, float axis2, float axis3, float angle, float scale, THFloatTensor* dest)
{
	THFloatTensor_resize2d(dest, 8, 3);
	float* data = THFloatTensor_data(dest);
	
	itk::Vector<T,3> axis;
	axis[0] = axis3;
    axis[1] = axis2;
    axis[2] = axis1;
	
	Similarity3DTransformT::Pointer transform = Similarity3DTransformT::New();
	transform->SetScale(1.f/scale);
	transform->SetRotation (axis, -angle);
	
	itk::Point<T,3> p, pt;
	int pos = 0;
	for (int i=-1; i<=1; i+=2)
		for (int j=-1; j<=1; j+=2)	
			for (int k=-1; k<=1; k+=2)		
			{
			    p[0] = i;
				p[1] = j;
				p[2] = k;
				pt = transform->TransformPoint(p);
				data[pos++] = pt[2];
				data[pos++] = pt[1];
				data[pos++] = pt[0];
			}
}

// set image to the size of tensor and deep-copy the content
bool tensorToImage(THFloatTensor* tensor, ImageT::Pointer& image)
{
 	if (tensor->nDimension != 3) {
 		std::cerr << "Input tensor doesn't have 3 dims: " << tensor->nDimension << std::endl;
 		return false;
	}
	
	image = ImageT::New();
  	ImageT::IndexType start;
  	start.Fill(0);
 	ImageT::SizeType size;
 	size[0] = tensor->size[2];
 	size[1] = tensor->size[1];
 	size[2] = tensor->size[0];	 	
 	
  	image->SetRegions(ImageT::RegionType(start, size));
  	image->Allocate();
  	memcpy(image->GetBufferPointer(), THFloatTensor_data(tensor), THFloatTensor_nElement(tensor)*sizeof(float));
  	//TODO: or just sharing? get inspired by http://docs.mitk.org/2014.10/mitkITKImageImport_8txx_source.html
  	return true;
}

// set tensor to the size of image and deep-copy the content
bool imageToTensor(ImageT::Pointer image, THFloatTensor* tensor)
{
 	if (image->ImageDimension != 3) {
 		std::cerr << "Input image doesn't have 3 dims: " << image->ImageDimension << std::endl;
 		return false;
	}
	
	ImageT::SizeType size = image->GetLargestPossibleRegion().GetSize();
	THFloatTensor_resize3d(tensor, size[2], size[1], size[0]);

  	memcpy(THFloatTensor_data(tensor), image->GetBufferPointer(), THFloatTensor_nElement(tensor)*sizeof(float));
  	//TODO: or just sharing? get inspired by http://docs.mitk.org/2014.10/mitkITKImageImport_8txx_source.html
  	return true;
}

// resamples img by the given transformation
void transformVolume(float axis1, float axis2, float axis3, float angle, float scale, float center1, float center2, float center3, THFloatTensor* src, THFloatTensor* dst)
{
	typedef itk::ResampleImageFilter<ImageT,ImageT,T,T> ResampleImageFilterT;
	typedef itk::LinearInterpolateImageFunction<ImageT, T> InterpolatorT;

	itk::Vector<T,3> axis;
	axis[0] = axis3;
    axis[1] = axis2;
    axis[2] = axis1;	
    
	itk::Point<T,3> center;
	center[0] = center3;
    center[1] = center2;
    center[2] = center1;      
	
	//Transform wants to be given the inverse transform
	Similarity3DTransformT::Pointer transform = Similarity3DTransformT::New();
	transform->SetScale(1.f/scale);
	transform->SetRotation (axis, -angle);
	transform->SetCenter(center);

	ImageT::Pointer srcimg;
	if (!tensorToImage(src, srcimg))
		return;
	
	//Do transform. Input size = result size. Note: this is really slow but no way around it.
	ResampleImageFilterT::Pointer imageResampleFilter = ResampleImageFilterT::New();
	imageResampleFilter->SetInput(srcimg);
    imageResampleFilter->SetSize(srcimg->GetLargestPossibleRegion().GetSize());
	imageResampleFilter->SetTransform(transform);
	imageResampleFilter->SetDefaultPixelValue(-1);
	imageResampleFilter->SetInterpolator(InterpolatorT::New());
	imageResampleFilter->Update();

	if (!imageToTensor(imageResampleFilter->GetOutput(), dst))
		return;
}


