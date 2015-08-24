#include "itkEuler3DTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include "nloptfunction.h"

typedef itk::Euler3DTransform<double> EulerTransformT;
typedef itk::ResampleImageFilter<ImageT,ImageT> ResamplerT;
typedef itk::LinearInterpolateImageFunction<ImageT,double> InterpolatorT;

double nloptSimilarityFunction( const std::vector<double> &transformationVector,
                                std::vector<double> &grad,
                                void *dataStructure)
{
    NloptDataStruct *m_dataStructure = (NloptDataStruct*)dataStructure;
    ImageT::Pointer fixedImage = m_dataStructure->m_fixedImage;
    ImageT::Pointer movingImage = m_dataStructure->m_movingImage;
    
    ResamplerT::Pointer resampler = ResamplerT::New();
    ImageT::SizeType    imgSize = fixedImage->GetLargestPossibleRegion().GetSize();
    ImageT::Pointer     movedImage = ImageT::New();
    
    EulerTransformT::TranslationType translation;
    std::vector <double> rotation(3,0);
    
    rotation[0] = transformationVector[0];
    rotation[1] = transformationVector[1];
    rotation[2] = transformationVector[2];
    translation[0] = transformationVector[3];
    translation[1] = transformationVector[4];
    translation[2] = transformationVector[5];
    
    EulerTransformT::Pointer eulerTransform = EulerTransformT::New();
    InterpolatorT::Pointer interpolator = InterpolatorT::New();
    eulerTransform->SetTranslation(translation);
    eulerTransform->SetRotation(rotation[0],rotation[1],rotation[2]);
    ImageT::IndexType centerImageSpace;
    ImageT::PointType centerWorldSpace;
    for (int iDim=0; iDim< 3; iDim++)
       centerImageSpace[iDim] =  imgSize[iDim]/2;
    fixedImage->TransformIndexToPhysicalPoint(centerImageSpace,centerWorldSpace);
    
    eulerTransform->SetCenter(centerWorldSpace);
    
    //resampler->SetSize(imgSize);
    resampler->SetInput(movingImage);
    resampler->SetTransform(eulerTransform);
    resampler->SetInterpolator(interpolator);
    resampler->UseReferenceImageOn();
    resampler->SetReferenceImage(fixedImage);
    movedImage = resampler->GetOutput();
    movedImage->Update();
    
    m_dataStructure->m_similarityMetric->setFixedImage(fixedImage);
    m_dataStructure->m_similarityMetric->setMovingImage(movedImage);
    m_dataStructure->m_similarityMetric->compute();
    
    double similarityValue;
    
    similarityValue = m_dataStructure->m_similarityMetric->getSimilarityValue();
    std::cout << transformationVector[0] << " " << transformationVector[1] << " " 
              << transformationVector[2] << " " << transformationVector[3] << " " 
              << transformationVector[4] << " " << transformationVector[5] << " " 
              << ":  " << similarityValue; 
    
    std::cout << std::endl;
        
    
    return similarityValue;
}