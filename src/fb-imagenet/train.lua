--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'sgdcaffe'
require 'opthelper'

local iter
local top1_epoch, top5_epoch, loss_epoch
local nIters = math.floor(datasetInfo.nTrain / opt.batchSize)
__threadid = 1

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local learningDebugger = LearningDebugger()
local loggingFreqBatch = 40
local loggingFreqSingle = 100
local optloss, optstep

local config = {}
if (opt.network ~= '' and opt.networkLoadOpt and opt.numEpochs > 0) then
    config = torch.load(opt.network..'.optconfig')     
end

local dispatcherRing = DispatcherRing(math.max(1,opt.nDonkeys))

local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

-------------------------------------------------------------------------------------------------------------
function train()
    model:training()

    -- next epoch
    model.epoch = model.epoch + 1     
    iter = 0
    
    -- set the exact rng state
    if config.rngCuda then cutorch.setRNGState(config.rngCuda) end
    if config.rngCpu then 
        torch.setRNGState(config.rngCpu[0]) 
        donkeys:specific(true)
        for id=1,math.min(opt.nDonkeys, #config.rngCpu) do local s = config.rngCpu[id]; donkeys:addjob(id, function() return torch.setRNGState(s) end) end
        donkeys:specific(false)
    end

    -- do one epoch
    print('<trainer ' .. opt.runName .. '> on training set:')
    print("<trainer> online epoch # " .. model.epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    cutorch.synchronize()

    local tm = torch.Timer()
    top1_epoch = 0
    top5_epoch = 0
    loss_epoch = 0

    --TODO: randperm instead of random samplingthis is a source of nondeterminism, threads may deliver in different order. 
    -- proposal: randperm is parted between threads. main thread has #thread slots, where it stores the results. at each new store, it tries to process the ringbuffer sequentially
    local semaIds = dispatcherRing:getSemaphoreIds()
    for i=1,nIters do
        -- queue jobs to data-workers
        
        donkeys:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, labels = trainLoader:sample(opt.batchSize)
                
                local Threads = require 'threads'
                local s = Threads.Mutex(semaIds[__threadid])
                s:lock(); s:free()
                return __threadid, sendTensor(inputs), sendTensor(labels)
            end,
            -- the end callback (runs in the main thread)
            function(tidx, inputsThread, labelsThread)
                dispatcherRing:receive(tidx, inputsThread, labelsThread)
                dispatcherRing:dispatch(trainBatch)
            end
        )
    end

    donkeys:synchronize()
    cutorch.synchronize()
    
    learningLogger:logWeights(parameters)
    learningLogger:logLoss(optloss)
    learningLogger:logGradUpd(optstep, config.learningRate)    

    learningDebugger:reset()

    top1_epoch = top1_epoch * 100 / (opt.batchSize * nIters)
    top5_epoch = top5_epoch * 100 / (opt.batchSize * nIters)
    loss_epoch = loss_epoch / nIters

    print(string.format('Epoch: [%d/%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f\t top-5 %.2f',
        model.epoch, opt.numEpochs, tm:time().real, loss_epoch, top1_epoch, top5_epoch))
    print('\n')
    
   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['% top5 accuracy (train set)'] = top5_epoch,
      ['avg loss (train set)'] = loss_epoch,
      ['epoch'] = model.epoch
   }    

    -- save the exact rng state
    config.rngCuda = cutorch.getRNGState()
    config.rngCpu = {[0]=torch.getRNGState()}
    donkeys:specific(true)
    for id=1,opt.nDonkeys do donkeys:addjob(id, function() return torch.getRNGState() end, function(c) config.rngCpu[id] = c end) end
    donkeys:specific(false)    

    -- save/log current net (a clean model without any gradients, inputs,... ; it takes less storage)
    cleanModelSave(model, parameters, config, opt, 'network.net')

    collectgarbage()

    local ret = {}
    ret['meanAccuracy'] = top1_epoch --% top1 accuracy (train set)
    ret['meanAccuracy5'] = top5_epoch--% top5 accuracy (train set)
    ret['meanLoss'] = loss_epoch     --avg loss (train set)
    return ret    
end

-------------------------------------------------------------------------------------------------------------
function trainBatch(inputsCPU, labelsCPU)
    cutorch.synchronize()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    labels:resize(labelsCPU:size()):copy(labelsCPU)

    local top1 = 0
    local top5 = 0   
    local loss, outputs = 0, nil

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        --   learningDebugger:visit(model, 1); 
        --   learningDebugger:reset()                     

        if opt.batchEvalSize < 1 then
            outputs = model:forward(inputs)
            loss = criterion:forward(outputs, labels)
            local df_do = criterion:backward(outputs, labels)
            model:backward(inputs, df_do)  
        else
            assert(opt.batchSize % opt.batchEvalSize == 0)
            local nEvals = opt.batchSize / opt.batchEvalSize        
        
            --note: computing batch per-partes will not produce the exact same gradient as with single shot. 1) small numerical isues, 2) dropout generated differently, 3) ccn2 weird stuff
            for i = 1,opt.batchSize,opt.batchEvalSize do 
                local inp = (opt.batchEvalSize == 1) and inputs[i] or inputs:sub(i, math.min(opt.batchSize, i+opt.batchEvalSize-1))
                local lab = (opt.batchEvalSize == 1) and labels[i] or labels:sub(i, math.min(opt.batchSize, i+opt.batchEvalSize-1))

                local output = model:forward(inp)
                loss = loss + criterion:forward(output, lab)
                local df_do = criterion:backward(output, lab)                        
                model:backward(inp, df_do)

                if not outputs then outputs = torch.Tensor(opt.batchSize, output:nElement()/opt.batchEvalSize):typeAs(output) end
                outputs:sub(i, math.min(opt.batchSize, i+opt.batchEvalSize-1)):copy(output)
            end
            loss = loss / nEvals
            gradParameters:div(nEvals)       
        end   
        
        for i,module in ipairs(model:listModules()) do
            if (module.weight ~= nil and module.weight:nElement()>0 and module.pendingSharing==nil) then
                print(i, module, torch.mean(module.output), torch.std(module.output), torch.mean(module.gradWeight), torch.std(module.gradWeight), torch.mean(module.gradBias), torch.mean(module.weight), torch.mean(module.bias)) end end
        print(torch.mean(inputs), torch.std(inputs))--]]

        --if (opt.batchSize>1 and iter % loggingFreqBatch == 0) then learningDebugger:visit(model, datasetInfo.nTrain/opt.batchSize/loggingFreqBatch) end
        --if (opt.batchSize==1 and t % loggingFreqSingle == 0) then learningDebugger:visit(model, datasetInfo.nTrain/loggingFreqSingle) end
        
        -- return f and df/dX
        prepareGradPerModule(model, opt)
        return loss,gradParameters
    end



    --if (opt.batchSize>1 and iter % loggingFreqBatch == 0) or (opt.batchSize==1 and iter % loggingFreqSingle == 0) then
    --    learningLogger:logWeights(parameters)
    --end           

    -- optimize on current mini-batch
    optloss, optstep = doOptStep(model, parameters, feval, opt, config)
    
    --if (opt.batchSize>1 and iter % loggingFreqBatch == 0) or (opt.batchSize==1 and iter % loggingFreqSingle == 0) then
    --    learningLogger:logLoss(optloss)
    --    learningLogger:logGradUpd(optstep, config.learningRate)
    --end        
    
    cutorch.synchronize()
    if iter%10 == 0 then collectgarbage() end

    iter = iter + 1
    nlprint(('Epoch: [%d/%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
        model.epoch, opt.numEpochs, iter, nIters, timer:time().real, loss,
        config.learningRate, dataLoadingTime))
    loss_epoch = loss_epoch + loss
    
    --print output sizes the first time
    if model.epoch==1 and iter==1 then
        for i,module in ipairs(model:listModules()) do
            if module.output then print(i .. ': ' .. torch.type(module) .. ' ' .. formatSizeStr(module.output)) end
        end  
    end       

    -- top-1 and top-5 error
    do
        local gt = labelsCPU
        local _,prediction_sorted = outputs:float():sort(2, true) -- descending
        for i=1,opt.batchSize do
            local pi = prediction_sorted[i]
            if pi[1] == gt[i] then top1 = top1 + 1; top5 = top5 + 1;
            else for j=2,5 do if pi[j] == gt[i] then top5 = top5 + 1; break; end; end; end
        end
        top1_epoch = top1_epoch + top1; top5_epoch = top5_epoch + top5
        top1 = top1 * 100 / opt.batchSize; top5 = top5 * 100 / opt.batchSize
    end
    if (iter % 15) == 0 then
        nlprint(string.format('Accuracy ' ..
            'top1-%%: %.2f \t' ..
            'top5-%%: %.2f \t' ..
            'Loss: %.4f \t' ..
            'LR: %.0e',
            top1, top5, loss,
            config.learningRate))
    end
    dataTimer:reset()
end
