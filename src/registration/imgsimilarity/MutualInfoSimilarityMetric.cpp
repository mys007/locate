#include "MutualInfoSimilarityMetric.h"
#include "itkJoinImageFilter.h"
#include "itkImageToHistogramFilter.h"
typedef itk::JoinImageFilter<ImageT, ImageT >  JoinFilterType;


void MutualInfoSimilarityMetric::compute()
{
	JoinFilterType::Pointer joinFilter = JoinFilterType::New();
	joinFilter->SetInput1(m_fixedImage);
	joinFilter->SetInput2(m_movingImage);
	joinFilter->Update();


	typedef JoinFilterType::OutputImageType            VectorImageType;
	typedef itk::Statistics::ImageToHistogramFilter<
		VectorImageType >  HistogramFilterType;
	HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();

	histogramFilter->SetInput(joinFilter->GetOutput());
	histogramFilter->SetMarginalScale(10.0);

	typedef HistogramFilterType::HistogramSizeType   HistogramSizeType;
	HistogramSizeType size(2);
	size[0] = 64;  // number of bins for the first  channel
	size[1] = 64;  // number of bins for the second channel

	histogramFilter->SetHistogramSize(size);
	typedef HistogramFilterType::HistogramMeasurementVectorType
		HistogramMeasurementVectorType;
	HistogramMeasurementVectorType binMinimum(3);
	HistogramMeasurementVectorType binMaximum(3);
	binMinimum[0] = -0.5;
	binMinimum[1] = -0.5;
	binMinimum[2] = -0.5;
	binMaximum[0] = 255.5;
	binMaximum[1] = 255.5;
	binMaximum[2] = 255.5;
	histogramFilter->SetHistogramBinMinimum(binMinimum);
	histogramFilter->SetHistogramBinMaximum(binMaximum);
	histogramFilter->Update();

	typedef HistogramFilterType::HistogramType  HistogramType;
	const HistogramType * histogram = histogramFilter->GetOutput();

	HistogramType::ConstIterator itr = histogram->Begin();
	HistogramType::ConstIterator end = histogram->End();
	const double Sum = histogram->GetTotalFrequency();

	double JointEntropy = 0.0;
	while (itr != end)
	{
		const double count = itr.GetFrequency();
		if (count > 0.0)
		{
			const double probability = count / Sum;
			JointEntropy +=
				-probability * std::log(probability) / std::log(2.0);
		}
		++itr;
	}
	// Software Guide : EndCodeSnippet


	size[0] = 64;  // number of bins for the first  channel
	size[1] = 1;  // number of bins for the second channel
	histogramFilter->SetHistogramSize(size);
	histogramFilter->Update();

	itr = histogram->Begin();
	end = histogram->End();
	double Entropy1 = 0.0;
	while (itr != end)
	{
		const double count = itr.GetFrequency();
		if (count > 0.0)
		{
			const double probability = count / Sum;
			Entropy1 += -probability * std::log(probability) / std::log(2.0);
		}
		++itr;
	}

	size[0] = 1;  // number of bins for the first channel
	size[1] = 64;  // number of bins for the second channel
	histogramFilter->SetHistogramSize(size);
	histogramFilter->Update();

	itr = histogram->Begin();
	end = histogram->End();
	double Entropy2 = 0.0;
	while (itr != end)
	{
		const double count = itr.GetFrequency();
		if (count > 0.0)
		{
			const double probability = count / Sum;
			Entropy2 += -probability * std::log(probability) / std::log(2.0);
		}
		++itr;
	}
	// Software Guide : EndCodeSnippet
	//	std::cout << "Image2 Entropy   = " << Entropy2 << " bits " << std::endl;

	double MutualInformation = Entropy1 + Entropy2 - JointEntropy;
	// Software Guide : EndCodeSnippet
	//std::cout << "Mutual Information = " << MutualInformation << " bits " << std::endl;
	double NormalizedMutualInformation1 =
		2.0 * MutualInformation / (Entropy1 + Entropy2);
	// Software Guide : EndCodeSnippet
	
		
	m_similarityValue = NormalizedMutualInformation1;
	
}