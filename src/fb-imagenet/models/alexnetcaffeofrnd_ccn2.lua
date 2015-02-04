require 'ccn2'
require 'cudnn'
package.path = "../?.lua;../myrock/?.lua;" .. package.path
require('pyratest')

function createModel(nGPU)

    local osrt = false

    assert(opt.caffeBiases)

-- -backend ccn2 -modelName alexnetcaffe -nDonkeys 5 -caffeBiases true -learningRate 0.01 -lrdPolicy step -lrdStep 100000 -lrdGamma 0.1 -weightDecay 0.0005 -numIters 450000 -batchSize 256 -imagenetCaffeInput true -optimization SGDCaffe

    local function gaussConstInit(module, wstddev, bval)
        assert(module and wstddev)
        module.weight:normal(0, wstddev)  
        module.bias:fill(bval or 0)
        return module        
    end

   local features = nn.Sequential() -- branch 1
   features:add(nn.Transpose({1,4},{1,3},{1,2}))
   features:add(gaussConstInit( ccn2.SpatialConvolution(3, 96, 11, 4, 0, 1), 0.01))
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
features:add(nn.Transpose({4,1},{4,2},{4,3}))   
   --features:add(ccn2.SpatialMaxPooling(3,2))                   -- 55 ->  27
   features:add(osrt and createOverfeatSubsamplingRndTestAvgFast(3,3,2,2) or createOverfeatSubsamplingRnd(3,3,2,2))
features:add(nn.Transpose({1,4},{1,3},{1,2}))   
   features:add(gaussConstInit( ccn2.SpatialConvolution(96, 256, 5, 1, 2, 2), 0.01, 0.1))       --  27 -> 27
   features:add(cudnn.ReLU(true))
   features:add(ccn2.SpatialCrossResponseNormalization(5, 0.000100, 0.7500, 1.000000))
features:add(nn.Transpose({4,1},{4,2},{4,3}))   
   --features:add(ccn2.SpatialMaxPooling(3,2))                   --  27 ->  13
    features:add(osrt and createOverfeatSubsamplingRndTestAvgFast(3,3,2,2) or createOverfeatSubsamplingRnd(3,3,2,2))
features:add(nn.Transpose({1,4},{1,3},{1,2}))   
   features:add(gaussConstInit( ccn2.SpatialConvolution(256, 384, 3, 1, 1, 1), 0.01))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(gaussConstInit( ccn2.SpatialConvolution(384, 384, 3, 1, 1, 2), 0.01, 0.1))      --  13 ->  13
   features:add(cudnn.ReLU(true))
   features:add(gaussConstInit( ccn2.SpatialConvolution(384, 256, 3, 1, 1, 2), 0.01, 0.1))      --  13 ->  13
   features:add(cudnn.ReLU(true))
features:add(nn.Transpose({4,1},{4,2},{4,3}))
   --features:add(ccn2.SpatialMaxPooling(3,2))                   -- 13 -> 6
   features:add(osrt and createOverfeatSubsamplingRndTestAvgFast(3,3,2,2) or createOverfeatSubsamplingRnd(3,3,2,2))
   --features:add(nn.Transpose({4,1},{4,2},{4,3}))

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
