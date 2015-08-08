--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'opthelper'

local iter
local loss_epoch
local confusion = optim.ConfusionMatrix(datasetInfo.nClasses)
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

local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

-------------------------------------------------------------------------------------------------------------
function train()
    model:training()
    donkeys:specific(true)
    
    -- next epoch
    model.epoch = model.epoch + 1     
    iter = 0
    
    -- set the exact rng state
    if config.rngCuda then cutorch.setRNGState(config.rngCuda) end
    if config.rngCpu then 
        torch.setRNGState(config.rngCpu[0]) 
        for id=1,math.min(opt.nDonkeys, #config.rngCpu) do local s = config.rngCpu[id]; donkeys:addjob(id, function() return torch.setRNGState(s) end) end
    end

    -- do one epoch
    print('<trainer ' .. opt.runName .. '> on training set:')
    print("<trainer> online epoch # " .. model.epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    cutorch.synchronize()

    local tm = torch.Timer()
    confusion:zero()
    loss_epoch = 0
    
    assert(opt.batchSize % opt.numTSPatches == 0)

    -- threads may deliver in different order, dispatcher prevents this nondeterminism. 
    -- now randsamp; if randperm wanted: generate randperm in main and split it to threads (todo)
    local dispatcherRing = DispatcherRing(math.max(1,opt.nDonkeys))
    local semaIds = dispatcherRing:getSemaphoreIds()
    
    for i=1,nIters do
        donkeys:addjob(1+(i-1)%opt.nDonkeys,
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, labels = trainLoader:sample(opt.batchSize / opt.numTSPatches)
                
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

    --learningDebugger:reset()
    
    print(confusion)
    --confusion:updateValids()
    --print('++ global correct: ' .. (confusion.totalValid*100) .. '%')
    loss_epoch = loss_epoch / nIters

    print(string.format('Epoch: [%d/%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
        .. 'average loss (per batch): %.2f \t '
        .. 'accuracy(%%):\t top-1 %.2f',
        model.epoch, opt.numEpochs, tm:time().real, loss_epoch, confusion.totalValid))
    print('\n')
    
   trainLogger:add{
      ['% accuracy (train set)'] = confusion.totalValid * 100,
      ['avg loss (train set)'] = loss_epoch,
      ['epoch'] = model.epoch
   }    

    -- save the exact rng state
    config.rngCuda = cutorch.getRNGState()
    config.rngCpu = {[0]=torch.getRNGState()}
    for id=1,opt.nDonkeys do donkeys:addjob(id, function() return torch.getRNGState() end, function(c) config.rngCpu[id] = c end) end 

    -- save/log current net (a clean model without any gradients, inputs,... ; it takes less storage)
    cleanModelSave(model, parameters, config, opt, 'network.net')

    collectgarbage()

    local ret = {}
    ret['meanAccuracy'] = confusion.totalValid * 100 --% top1 accuracy (train set)
    ret['meanLoss'] = loss_epoch     --avg loss (train set)
    return ret    
end

----
--[[local function saveImgBatch(batch,lbls)
    for s=1,batch:size(1) do
        for m=1,batch:size(2) do
            image.save('/home/simonovm/tmp/' .. (lbls[s]==1 and 'pos' or 'neg').. s .. '_' .. m .. '.png',image.toDisplayTensor{input=batch[s][m]})
        end
    end
end--]]

-------------------------------------------------------------------------------------------------------------
function trainBatch(inputsCPU, labelsCPU)
    cutorch.synchronize()
    local dataLoadingTime = dataTimer:time().real
    timer:reset()

    -- transfer over to GPU
    inputs:resize(inputsCPU:size()):copy(inputsCPU)
    if opt.criterion == "bsvm" or opt.criterion == "emb" then labelsCPU[torch.eq(labelsCPU,2)] = -1 end
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
        
        --[[for i,module in ipairs(model:listModules()) do
            if (module.weight ~= nil and module.weight:nElement()>0 and module.pendingSharing==nil) then
                print(i, module, torch.mean(module.output), torch.std(module.output), torch.mean(module.gradInput), torch.std(module.gradInput))--, torch.mean(module.gradWeight), torch.std(module.gradWeight), torch.std(module.gradBias), torch.mean(module.weight), torch.mean(module.bias)) 
            else
                print(i, torch.type(module), torch.mean(module.output), torch.std(module.output), torch.mean(module.gradInput), torch.std(module.gradInput))
            end
        end
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
    
    dataTimer:reset()
end
