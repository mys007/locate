require 'ccn2'
require 'cudnn'
package.path = "../?.lua;" .. package.path
require 'myutils'
require 'JitteringModule'

function createModel(nGPU)

    assert(opt.caffeBiases and opt.winit == 'Gauss')

-- -backend ccn2 -modelName alexnetcaffe -nDonkeys 5 -caffeBiases true -learningRate 0.01 -lrdPolicy step -lrdStep 100000 -lrdGamma 0.1 -weightDecay 0.0005 -numIters 450000 -batchSize 256 -imagenetCaffeInput true -optimization SGDCaffe

   local features = nn.Sequential() -- branch 1
   features:add(nn.Transpose({1,4},{1,3},{1,2}))
   features:add(gaussConstInit( ccn2.SpatialConvolution(3, 96, 11, 4, 0, opt.modelParams['gr'] or 1), 0.01))
   if opt.modelParams['jit'] then features:add(nn.JitteringModuleGNoise(opt.modelParams['jit'])) end
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 55 ->  27
   features:add(gaussConstInit( ccn2.SpatialConvolution(96, 256, 5, 1, 2, opt.modelParams['gr'] or 2), 0.01, 0.1))       --  27 -> 27
   if opt.modelParams['jit'] then features:add(nn.JitteringModuleGNoise(opt.modelParams['jit'])) end
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
   features:add(ccn2.SpatialMaxPooling(3,2))                   --  27 ->  13
   features:add(gaussConstInit( ccn2.SpatialConvolution(256, 384, 3, 1, 1, opt.modelParams['gr'] or 1), 0.01))      --  13 ->  13
   if opt.modelParams['jit'] then features:add(nn.JitteringModuleGNoise(opt.modelParams['jit'])) end
   features:add(cudnn.ReLU(true))
   features:add(gaussConstInit( ccn2.SpatialConvolution(384, 384, 3, 1, 1, opt.modelParams['gr'] or 2), 0.01, 0.1))      --  13 ->  13
   if opt.modelParams['jit'] then features:add(nn.JitteringModuleGNoise(opt.modelParams['jit'])) end
   features:add(cudnn.ReLU(true))
   features:add(gaussConstInit( ccn2.SpatialConvolution(384, 256, 3, 1, 1, opt.modelParams['gr'] or 2), 0.01, 0.1))      --  13 ->  13
   if opt.modelParams['jit'] then features:add(nn.JitteringModuleGNoise(opt.modelParams['jit'])) end
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialMaxPooling(3,2))                   -- 13 -> 6
   features:add(nn.Transpose({4,1},{4,2},{4,3}))

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(-1):setNumInputDims(3))
   classifier:add(gaussConstInit( nn.Linear(256*6*6, 4096), 0.005, 0.1))
   classifier:add(cudnn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(gaussConstInit( nn.Linear(4096, 4096), 0.005, 0.1))
   classifier:add(cudnn.ReLU(true))
   classifier:add(nn.Dropout(0.5))
   classifier:add(gaussConstInit( nn.Linear(4096, datasetInfo.nClasses), 0.01))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model
end
