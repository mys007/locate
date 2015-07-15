--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local iter, nIters
local loss
local confusion = optim.ConfusionMatrix(datasetInfo.nClasses)
local timer = torch.Timer()
local _ = nil
__threadid = 1

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local dispatcherRing = DispatcherRing(math.max(1,opt.nDonkeys))


-- TODO: the real test-case will be given patch1 and image2, compute feats full-conv over image2 and output map? need to implement overfeat-like shifts or upsampling? or are they fine just with patch-patch similarity?


-------------------------------------------------------------------------------------------------------------
function test(isTest)
    model:evaluate()

    print("<tester> online epoch # " .. model.epoch .. ', ' .. (isTest and 'test' or 'valid') .. ' data:')
    
    local testBatchSize = opt.dataset=='imagenet' and 16 or 1 --16 for ccn2 (ccn2 backend constraint) and 10 (alexnet test eval avg))
    local nTest = isTest and datasetInfo.nTest or datasetInfo.nValid
    if (opt.nTestSamples>0 and datasetInfo.nTest > opt.nTestSamples and isTest) then nTest = opt.nTestSamples end
        
    nIters = math.floor(nTest/testBatchSize)
    if nTest % testBatchSize ~= 0 then print('Warning, '..(nTest % testBatchSize)/nTest..'% test examples will not be tested due to batch size.') end    

    iter = 0
    cutorch.synchronize()
    timer:reset()
    loss = 0
    
    local semaIds = dispatcherRing:getSemaphoreIds()
    for i=1,nIters do
        local indexStart = (i-1) * testBatchSize + 1
        local indexEnd = (indexStart + testBatchSize - 1)
        donkeys:addjob(
            -- work to be done by donkey thread
            function()
                local inputs, labels = (isTest and testLoader or trainLoader):get(indexStart, indexEnd)
                
                local Threads = require 'threads'
                local s = Threads.Mutex(semaIds[__threadid])
                s:lock(); s:free()                
                return __threadid, sendTensor(inputs), sendTensor(labels)
            end,
            -- callback that is run in the main thread once the work is done
            function(tidx, inputsThread, labelsThread)
                dispatcherRing:receive(tidx, inputsThread, labelsThread)
                dispatcherRing:dispatch(testBatch)
            end
        )
    end

    donkeys:synchronize()
    cutorch.synchronize()
    
    print(confusion)
    loss = loss / nIters    

    if isTest then 
        testLogger:add{
            ['% top1 accuracy (test set)'] = confusion.totalValid ,
            ['avg loss (test set)'] = loss,
            ['epoch'] = model.epoch } 
    end

    print(string.format('Epoch: [%d]['.. (isTest and 'TESTING' or 'VALIDATION') ..' SUMMARY] Total Time(s): %.2f \t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy (%%):\t top-1 %.2f',
        model.epoch, timer:time().real, loss, confusion.totalValid * 100))

    print('\n')

    local ret = {}
    ret['meanAccuracy'] = confusion.totalValid * 100   
    return ret
end

-------------------------------------------------------------------------------------------------------------
function testBatch(inputsCPU, labelsCPU)

    xlua.progress(iter, nIters)
    iter = iter + 1

    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    if opt.criterion == "bsvm" or opt.criterion == "emb" then labelsCPU[torch.eq(labelsCPU,2)] = -1 end
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local err, outputs = 0, nil
    if opt.batchEvalSize ~= 1 then
        outputs = model:forward(inputs)
        err = criterion:forward(outputs, labels)    
    else
        outputs = torch.CudaTensor(inputs:size(1),1000) --TODO
        for i = 1,inputs:size(1) do
            outputs[i] = model:forward(inputs[i])                         
            err = err + criterion:forward(outputs[i], labels[i])
        end
        err = err / inputs:size(1)
    end

    cutorch.synchronize()
    if iter%10 == 0 then collectgarbage() end
    
     --confusion
    outputs = outputs:float()
    if opt.criterion == "bsvm" or opt.criterion == "emb" then 
        outputs = outputs:view(-1)
        for i=1,labelsCPU:nElement() do
            local isPos = outputs[i]>0; if (opt.criterion == "emb") then isPos = outputs[i]<criterion.margin end
            local it, ip = (labelsCPU[i]==1 and 1 or 2), (isPos and 1 or 2)
            confusion.mat[it][ip] = confusion.mat[it][ip] + 1
        end
    else        
        confusion:add(outputs, labelsCPU)
    end   
    
    loss = loss + err
end
