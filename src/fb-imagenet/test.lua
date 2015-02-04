--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local iter, nIters
local top1_center, top5_center, loss
local top1_10crop, top5_10crop
local timer = torch.Timer()
local _ = nil
__threadid = 1

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local dispatcherRing = DispatcherRing(math.max(1,opt.nDonkeys))

-------------------------------------------------------------------------------------------------------------
function test(isTest)
    model:evaluate()

    print("<tester> online epoch # " .. model.epoch .. ', ' .. (isTest and 'test' or 'valid') .. ' data:')
    
    local testBatchSize = 16 --(opt.backend=='ccn2') and 16 or 12 -- =120 (close to 128) or =160 (smallest multiple of 32 (ccn2 backend constraint) and 10 (alexnet test eval avg))
    local nTest = datasetInfo.nTest
    if opt.nTestSamples>0 and datasetInfo.nTest > opt.nTestSamples and isTest then nTest = opt.nTestSamples end
        
    nIters = math.floor(nTest/testBatchSize)
    if nTest % testBatchSize ~= 0 then print('Warning, '..(nTest % testBatchSize)/nTest..'% test examples will not be tested due to batch size.') end    

    iter = 0
    cutorch.synchronize()
    timer:reset()

    top1_center = 0; top5_center = 0
    top1_10crop = 0; top5_10crop = 0
    loss = 0
    
    local semaIds = dispatcherRing:getSemaphoreIds()
    for i=1,nIters do
        local indexStart = (i-1) * testBatchSize + 1
        local indexEnd = (indexStart + testBatchSize - 1)
        donkeys:addjob(
            -- work to be done by donkey thread
            function()
                local inputs, labels = testLoader:get(indexStart, indexEnd)
                
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

    top1_center = top1_center * 100 / nTest
    top5_center = top5_center * 100 / nTest
    top1_10crop = top1_10crop * 100 / nTest
    top5_10crop = top5_10crop * 100 / nTest
    loss = loss / nIters -- because loss is calculated per batch
    testLogger:add{
        ['% top1 accuracy (test set) (center crop)'] = top1_center,
        ['% top5 accuracy (test set) (center crop)'] = top5_center,
        ['% top1 accuracy (test set) (10 crops)'] = top1_10crop,
        ['% top5 accuracy (test set) (10 crops)'] = top5_10crop,
        ['avg loss (test set)'] = loss
    }

    print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy [Center](%%):\t top-1 %.2f\t top-5 %.2f\t'
        .. '[10crop](%%):\t top-1 %.2f\t top-5 %.2f',
        model.epoch, timer:time().real, loss, top1_center, top5_center,
        top1_10crop, top5_10crop))

    print('\n')

    local ret = {}
    ret['meanAccuracy'] = top1_center   
    return ret
end

-------------------------------------------------------------------------------------------------------------
function testBatch(inputsCPU, labelsCPU)

    xlua.progress(iter, nIters)
    iter = iter + 1

    inputs:resize(inputsCPU:size()):copy(inputsCPU)
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
    
    local pred = outputs:float()

    loss = loss + err

    local function topstats(p, g)
        local top1 = 0; local top5 = 0
        _,p = p:sort(1, true)
        if p[1] == g then
            top1 = top1 + 1
            top5 = top5 + 1
        else
            for j=2,5 do
                if p[j] == g then
                    top5 = top5 + 1
                    break
                end
            end
        end
        return top1, top5, p[1]
    end

    for i=1,pred:size(1),10 do
        local p = pred[{{i,i+9},{}}]
        local g = labelsCPU[i]
        for j=0,9 do assert(labelsCPU[i] == labelsCPU[i+j]) end
        -- center
        local center = p[1] + p[2]
        local top1,top5 = topstats(center, g)
        top1_center = top1_center + top1
        top5_center = top5_center + top5
        -- 10crop
        local tencrop = p:sum(1)[1]
        local top1,top5,ans = topstats(tencrop, g)
        top1_10crop = top1_10crop + top1
        top5_10crop = top5_10crop + top5
    end
end
