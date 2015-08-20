
#include "SimilarityMetric.h"


SimilarityMetric::SimilarityMetric() 
{
	m_similarityValue = 0;

}

void SimilarityMetric::setFixedImage(ImageT::Pointer fixedImage)
{
	m_fixedImage = fixedImage;
}

void SimilarityMetric::setMovingImage(ImageT::Pointer movingImage)
{

	m_movingImage = movingImage;
}

double SimilarityMetric::getSimilarityValue()
{

	return m_similarityValue;
}

