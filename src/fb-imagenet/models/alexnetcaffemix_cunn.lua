require 'cunn'
require 'cudnn'
require 'ccn2'
require 'inn'
package.path = "../?.lua;" .. package.path
require('myutils')
require('SpatialMaxPoolingCaffe')

function createModel(nGPU)

    assert(opt.caffeBiases and opt.winit == 'Gauss')

-- -backend ccn2 -modelName alexnetcaffe -nDonkeys 5 -caffeBiases true -learningRate 0.01 -lrdPolicy step -lrdStep 100000 -lrdGamma 0.1 -weightDecay 0.0005 -numIters 450000 -batchSize 256 -imagenetCaffeInput true -optimization SGDCaffe

    local features = nn.Sequential() -- branch 1
    
    if opt.modelParams['vers'] and opt.modelParams['vers']>=20 then
        -- ccn2 conv, rest nn
       features:add(nn.Transpose({1,4},{1,3},{1,2}))
       features:add(gaussConstInit( ccn2.SpatialConvolution(3, 96, 11, 4, 0, opt.modelParams['gr'] or 1), 0.01))
       features:add(nn.Transpose({4,1},{4,2},{4,3}))
       features:add(nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))
       features:add(nn.Transpose({1,4},{1,3},{1,2}))
       features:add(gaussConstInit( ccn2.SpatialConvolution(96, 256, 5, 1, 2, opt.modelParams['gr'] or 2), 0.01, 0.1))       --  27 -> 27
       features:add(nn.Transpose({4,1},{4,2},{4,3}))
       features:add(nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))
       features:add(nn.Transpose({1,4},{1,3},{1,2}))
       features:add(gaussConstInit( ccn2.SpatialConvolution(256, 384, 3, 1, 1, opt.modelParams['gr'] or 1), 0.01))      --  13 ->  13
       features:add(cudnn.ReLU())
       features:add(gaussConstInit( ccn2.SpatialConvolution(384, 384, 3, 1, 1, opt.modelParams['gr'] or 2), 0.01, 0.1))      --  13 ->  13
       features:add(cudnn.ReLU())
       features:add(gaussConstInit( ccn2.SpatialConvolution(384, 256, 3, 1, 1, opt.modelParams['gr'] or 2), 0.01, 0.1))      --  13 ->  13
       features:add(cudnn.ReLU())
       features:add(nn.Transpose({4,1},{4,2},{4,3}))
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))
    
    elseif opt.modelParams['vers'] and opt.modelParams['vers']>=10 then
       features:add(gaussConstInit( nn.SpatialConvolutionMM(3, 96, 11, 11, 4, 4, 0, 0, 1), 0.01))
       features:add(opt.modelParams['vers']==11 and cudnn.ReLU(true) or nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))
       features:add(gaussConstInit( nn.SpatialConvolutionMM(96, 256, 5, 5, 1, 1, 2, 2, 2), 0.01, 0.1))       --  27 -> 27
       features:add(opt.modelParams['vers']==11 and cudnn.ReLU(true) or nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))
       features:add(gaussConstInit( nn.SpatialConvolutionMM(256, 384, 3, 3, 1, 1, 1, 1, 1), 0.01))      --  13 ->  13
       features:add(opt.modelParams['vers']==11 and cudnn.ReLU(true) or nn.ReLU())
       features:add(gaussConstInit( nn.SpatialConvolutionMM(384, 384, 3, 3, 1, 1, 1, 1, 2), 0.01, 0.1))      --  13 ->  13
       features:add(opt.modelParams['vers']==11 and cudnn.ReLU(true) or nn.ReLU())
       features:add(gaussConstInit( nn.SpatialConvolutionMM(384, 256, 3, 3, 1, 1, 1, 1, 2), 0.01, 0.1))      --  13 ->  13
       features:add(opt.modelParams['vers']==11 and cudnn.ReLU(true) or nn.ReLU())
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))
       -- ---> all these ReLUs are equal
   elseif opt.modelParams['vers'] and opt.modelParams['vers']>0 then
       -- nn conv, rest cudnn (vers=1) 
       features:add(gaussConstInit( nn.SpatialConvolutionMM(3, 96, 11, 11, 4, 4, 0, 0, 1), 0.01))
       features:add(opt.modelParams['vers']==1 and cudnn.ReLU(true) or nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())                   -- 55 ->  27
       features:add(gaussConstInit( nn.SpatialConvolutionMM(96, 256, 5, 5, 1, 1, 2, 2, 2), 0.01, 0.1))       --  27 -> 27
       features:add(opt.modelParams['vers']==1 and cudnn.ReLU(true) or nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())                   --  27 ->  13
       features:add(gaussConstInit( nn.SpatialConvolutionMM(256, 384, 3, 3, 1, 1, 1, 1, 1), 0.01))      --  13 ->  13
       features:add(opt.modelParams['vers']==1 and cudnn.ReLU(true) or nn.ReLU())
       features:add(gaussConstInit( nn.SpatialConvolutionMM(384, 384, 3, 3, 1, 1, 1, 1, 2), 0.01, 0.1))      --  13 ->  13
       features:add(opt.modelParams['vers']==1 and cudnn.ReLU(true) or nn.ReLU())
       features:add(gaussConstInit( nn.SpatialConvolutionMM(384, 256, 3, 3, 1, 1, 1, 1, 2), 0.01, 0.1))      --  13 ->  13
       features:add(opt.modelParams['vers']==1 and cudnn.ReLU(true) or nn.ReLU())
       features:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())                   -- 13 -> 6    
       -- --> cudnn.SpatialMaxPooling is not deterministic
   else
       -- cudnn conv, rest nn
       features:add(gaussConstInit( cudnn.SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0, opt.modelParams['gr'] or 1), 0.01))
       features:add(nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))                   -- 55 ->  27
       features:add(gaussConstInit( cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, opt.modelParams['gr'] or 2), 0.01, 0.1))       --  27 -> 27
       features:add(nn.ReLU())
       features:add(inn.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))                   --  27 ->  13
       features:add(gaussConstInit( cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, opt.modelParams['gr'] or 1), 0.01))      --  13 ->  13
       features:add(nn.ReLU())
       features:add(gaussConstInit( cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, opt.modelParams['gr'] or 2), 0.01, 0.1))      --  13 ->  13
       features:add(nn.ReLU())
       features:add(gaussConstInit( cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, opt.modelParams['gr'] or 2), 0.01, 0.1))      --  13 ->  13
       features:add(nn.ReLU())
       features:add(myrock.SpatialMaxPoolingCaffe(3,3,2,2))                   -- 13 -> 6
       -- --> cudnn.SpatialConvolution is not deterministic
    end
    
   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(-1):setNumInputDims(3))
   classifier:add(gaussConstInit( nn.Linear(256*6*6, 4096), 0.005, 0.1))
   classifier:add(nn.ReLU())
   classifier:add(nn.Dropout(0.5))
   classifier:add(gaussConstInit( nn.Linear(4096, 4096), 0.005, 0.1))
   classifier:add(nn.ReLU())
   classifier:add(nn.Dropout(0.5))
   classifier:add(gaussConstInit( nn.Linear(4096, datasetInfo.nClasses), 0.01))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)
   
 

   return model
end
