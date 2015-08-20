#ifndef SIMILARITYMETRIC_H_INCLUDED
#define SIMILARITYMETRIC_H_INCLUDED
#include "itkImage.h"
#include "typedefinitions.h"

class SimilarityMetric
{
	public:
		SimilarityMetric();
		virtual void compute() = 0;
		double getSimilarityValue();
		void setFixedImage(ImageT::Pointer fixedImage);
		void setMovingImage(ImageT::Pointer movingImage);
	
	protected:	
		ImageT::Pointer m_fixedImage;
		ImageT::Pointer m_movingImage;
		double m_similarityValue;
};
#endif