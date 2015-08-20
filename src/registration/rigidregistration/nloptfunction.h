#ifndef SIMILARITYEVALUATION_H_INCLUDED
#define SIMILARITYEVALUATION_H_INCLUDED

#include "itkImage.h"
#include "SimilarityMetric.h"
#include "NnSimilarityMetric.h"

typedef struct
{
    ImageT::Pointer m_fixedImage;
    ImageT::Pointer m_movingImage;
    SimilarityMetric *m_similarityMetric;
    
    
} NloptDataStruct;



double nloptSimilarityFunction( const std::vector<double> &transformationVector,
                                std::vector<double> &grad ,
                                void *dataStructure);



#endif