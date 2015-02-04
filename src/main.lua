----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------
package.path = "myrock/?.lua;" .. package.path
require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'opthelper'
require 'image'
require 'trepl'
require 'Cifar10'
require 'Sun397'
require 'LearningDebugger'
require 'myutils'
require 'sgdcaffe'
local _ = nil



----------------------------------------------------------------------
-- parse command-line options
--
local cmd = torch.CmdLine(nil,nil,true)  --my fix   --TODO: wont be accepted, replace with https://github.com/davidm/lua-pythonic-optparse/blob/master/lmod/pythonic/optparse.lua
cmd:text()
cmd:text('CIFAR Training')
cmd:text()
cmd:text('Options:')
cmd:option('-runName', '', '')  --!!!!!!
cmd:option('-save', '', 'subdirectory to save/log experiments in')
cmd:option('-t', false, 'test only (test set)')
cmd:option('-v', false, 'test only (valid set)')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-networkLoadOpt', true, 'Load also the inner state of gradient optimizer when network is given (good for further training)')
cmd:option('-networkJustAsInit', false, 'Will take just the parameters (and epoch num if networkLoadOpt) from the network')
cmd:option('-modelName', 'baseline', '')  --!!!!!!
--
cmd:option('-device', 1, 'CUDA device')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 1, 'nb of threads to use (blas on cpu)')
--
cmd:option('-criterion', 'nll', 'criterion: nll | svm | mse')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | SGDCaffe | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-lrdPolicy', 'inv', 'SGD: policy for learning rate decay. fixed | inv | step | expep')
cmd:option('-lrdGamma', 5e-7, 'SGD: factor of decaying; meaning depends on the policy applied')
cmd:option('-lrdStep', 0, 'SGD: stepsize of decaying; meaning depends on the policy applied')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-batchSampling', 'seq', 'sampling of batches seq | rand | randperm')
cmd:option('-batchMeanGrad', true, 'whether to normalize gradient by batch size')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum: state.dfdx:mul(mom):add(1-damp, dfdx) (SGD only)')
cmd:option('-momdampening', -1, 'dampening when momentum used, (default = "momentum", i.e. lin combination) (SGD only)')
cmd:option('-asgdE0', 1, 'start averaging at epoch e0 (ASGD only)')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-numEpochs', 30, 'number of epochs to train (<0=test only)')
--
cmd:option('-dataset', 'Cifar10', 'dataset Cifar10 | Sun397')
cmd:option('-subsampleData', 20, 'use full dataset (50,000 samples)')
cmd:option('-nValidationSamples', 5000, 'size of validation set (beg of training data)')
cmd:option('-augmentCropScaleFlip', 0, 'creates 36 different crops of each data sample (0=no, 1=train, 2=all)')
cmd:option('-normalizationMode', 'YuvScnChStat', 'what normalization to use RgbZca | YuvScnChStat (Cifar) | none | zscore (Sun)')
cmd:option('-sunMinSize', 64, 'Sun397: the shorter side of images')
cmd:option('-sunFullTrainSet', false, 'Sun397: whether to ignore official partitions and use all training data except of the test partition')
cmd:option('-sunPartition', 1, 'Sun397: test and/or train set partition')
--
cmd:option('-pyraFactor', 0.7, 'downscaling factor in pyramid')
cmd:option('-pyraFanIn', 3, 'scale-convolution kernel diameter')
cmd:option('-pyraFuseMode', 'scalespp', 'how scales are fused? linearize | basescale | maxscale | scalespp | maxhidden | maxresult')
cmd:option('-parSharing', true, 'use weight sharing?')
cmd:option('-bilinInterp', false, 'interp: bilin or adaptiv maxpool')
cmd:option('-baselineCArch', 'c_16_5,p_2,c_256_5,p_2,c_128_5', 'configuration of baseline model')
cmd:option('-scalespaceCArch', 'c_16_5_2,p_2,c_256_5_2,a_5,e_128', 'configuration of convolutional part of scalespace model')
cmd:option('-pyraCArch', 'c_16_5,p_2,c_256_5,p_2,e_128', 'configuration of convolutional part of pyra model')
cmd:option('-scalespaceStep', 'exp', 'scale stepping strategy sqrt | linear | exp')
cmd:option('-scalespaceNumSc', 4, 'number of scales')
cmd:option('-scalespaceCreateFixed', true, 'whether to forbid learning of filters for creating the scale space')
cmd:option('-winit', 'default', 'weight initialization   default | He')
cmd:option('-mpCeil', false, 'whether output size of pooling should be computed by ceil (as in Caffe and cuda-convnet2) or floor (torch)')
cmd:option('-caffeBiases', false, 'initializes biases with learning rate 2x that of weights and turns off weight decay for them')
--
cmd:option('-showPlots', false, 'show plots during training')
cmd:option('-scPerturbFixSize', true, 'test set perturbation: if true, crop/pad is performed to stick with the training image sizes')
cmd:option('-scPerturbMin', 0, 'test set perturbation: min-bound for rnd dataset scaling factor')
cmd:option('-scPerturbMax', 0, 'test set perturbation: max-bound for rnd dataset scaling factor')

cmd:text()
local opt = cmd:parse(arg)

if opt.t or opt.v then opt.runName=''; opt.numEpochs=-1 end
opt.runName = os.date("%Y%m%d-%H%M%S") .. '-' .. opt.runName


-- cuda
local doCuda = opt.device > 0
if doCuda then
    print('Will use device '..opt.device)
    require 'cutorch'
    require 'cunn'    
    cutorch.setDevice(1)--opt.device)
end

torch.setdefaulttensortype('torch.FloatTensor') -- shouldn't use 'torch.CudaTensor', breaks libraries (e.g. image)

-- fix seed
torch.manualSeed(opt.seed)
if doCuda then cutorch.manualSeed(opt.seed) end

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------

-- compute pyramid scales
if (opt.pyraFactor > 0 and opt.scalespaceStep == "exp") then
    opt.pyraScales = {1}
    for i=2,opt.scalespaceNumSc do
        table.insert(opt.pyraScales, opt.pyraScales[#opt.pyraScales]*opt.pyraFactor)
    end     
    --while (32*opt.pyraScales[#opt.pyraScales]*opt.pyraFactor > 10) do  --(the smallest scale min 10px)
    --    table.insert(opt.pyraScales, opt.pyraScales[#opt.pyraScales]*opt.pyraFactor) 
    --end
end

-- precompute basic dataset properties
if opt.dataset == 'Cifar10' then
    opt.expInputSize = opt.augmentCropScaleFlip==0 and {3,32,32} or {3,24,24}
    opt.nclasses = 10
elseif opt.dataset == 'Sun397' then
    opt.expInputSize = {3,opt.sunMinSize, opt.sunMinSize} 
    opt.nclasses = 397
end

----------------------------------------------------------------------
-- define model to train

dofile 'model.lua'
--xpcall(function() createModel(opt) end, breakpt)
model, criterion = createModel(opt)
prepareModel(model, opt)

if opt.network ~= '' then
    if not opt.networkJustAsInit then
        print('<trainer> reloading previously trained network')
        model = torch.load(opt.network) --createModel() was called before so we have all our classes registered as factory   
    else
        print('<trainer> initializing with previously trained network')
        local tmp = torch.load(opt.network)
        model:getParameters():copy(tmp:getParameters())  --just copy the learned params to the new model
        if opt.networkLoadOpt then model.epoch = tmp.epoch end
    end
    
    --model:insert(nn.ScaleRemover({4}), 19)
    --for _,v in ipairs({19,16,13,9,6,3}) do model:insert(nn.ScaleRemover({1}), v) end
    --for _,v in ipairs({22,18,16,11,7,3}) do model:insert(nn.ScaleRemover({3}), v) end
    --model:insert(nn.ScaleRemoverPyra({1}), 9)
end

cleanModelInit(model, opt) --just for saving

if doCuda then model:cuda(); criterion:cuda() else model:float(); criterion:float() end

moduleSharing(model, opt)

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()
model.epoch = model.epoch or 0

local _,fname = sys.fpath()
opt.save = paths.concat(opt.save, fname:gsub('.lua','') .. '-' .. opt.modelName, opt.runName)
os.execute('mkdir -p "' .. opt.save .. '"')
cmd:log(paths.concat(opt.save, 'stdout.log'), opt)

print('<cmdline> CUDA_VISIBLE_DEVICES='..(opt.device-1)..' qlua main.lua ' .. table.concat(arg, " "))
print('<cifar> using model with ' .. parameters:nElement() .. ' parameters')
print('<opt.pyraScales>' .. table.concat(opt.pyraScales, ", "))
print(model)
for i,module in ipairs(model:listModules()) do
    if module.output then print(i .. ': ' .. torch.type(module) .. ' ' .. formatSizeStr(module.output)) end
end 

----------------------------------------------------------------------
-- ** dataset function

function loadDatasets(opt)

    print('<trainer> loading dataset ' .. opt.dataset)
    local dataset
    if opt.dataset == 'Cifar10' then
        dataset = torch.Cifar10({nSampleRatio=opt.subsampleData, 
                                 augmentCropScaleFlip=opt.augmentCropScaleFlip, 
                                 normalizationMode=opt.normalizationMode,
                                 nValidations=opt.nValidationSamples})
    elseif opt.dataset == 'Sun397' then
        dataset = torch.Sun397({nSampleRatio=opt.subsampleData, 
                                partition=opt.sunPartition, 
                                minSize=opt.sunMinSize,
                                squareCrop=true,
                                fullTrainSet=opt.sunFullTrainSet,
                                nValidations=opt.nValidationSamples,
                                normalizationMode=opt.normalizationMode})
                              
                              
        --[[if opt.criterion == "nll" then --weighting by inverse class frequency
            local nsamp = torch.histc(dataset.trainData.labels, #dataset:classes(), 1, #dataset:classes()+0.01)
            criterion.weights = torch.Tensor(#dataset:classes()):fill(nsamp:min())
            criterion.weights:cdiv(nsamp)     
            criterion.weights:typeAs(criterion.outputTensor)
            print(criterion.weights)
        end--]]    
    end
    
    if (opt.scPerturbMin>0 and opt.scPerturbMax>0) then
        if (opt.scPerturbFixSize) then
            dataset.testData:randRescaleCropPad(opt, opt.scPerturbMin, opt.scPerturbMax)
        else
            dataset.testData:randRescale(opt, opt.scPerturbMin, opt.scPerturbMax)
        end
    end
    
    if (opt.scPerturbBlurMin~=nil and opt.scPerturbBlurMin>0 and opt.scPerturbBlurMax>0) then
        dataset.testData:randGBlur(opt, opt.scPerturbBlurMin, opt.scPerturbBlurMax)
    end    
    
        
    if string.starts(opt.modelName,'pyra') then dataset:toPyra(opt.pyraScales) end
    --if string.starts(opt.modelName,'scalespace') then dataset:toScalespaceTensor(opt.pyraScales) end
    
    --dataset.trainData:rescale(1/opt.pyraFactor)
    --dataset.validData:rescale(1/opt.pyraFactor)
    --dataset.testData:rescale(1/opt.pyraFactor)
    
    if doCuda then 
        dataset:cuda(); 
        --save GPU memory (one-by-one upload)
        if string.starts(opt.modelName,'pyra') then 
            dataset:setPostprocFun(function (d,t) return tensortableType(d, 'torch.CudaTensor'), t end)
            --dataset:setPostprocFun(function (d,t) if torch.bernoulli()==1 then local k = gaussianfilter2D(9, torch.uniform(0.1, 1.0)); for i=1,#d do d[i]=image.convolve(d[i], k, 'same') end; end; return tensortableType(d, 'torch.CudaTensor'), t end)
            --dataset:setPostprocFun(function (d,t) local k = gaussianfilter2D(9, torch.uniform(0.1, 1.5)); for i=1,#d do d[i]=image.convolve(d[i], k, 'same') end; return tensortableType(d, 'torch.CudaTensor'), t end)
        else
            dataset:setPostprocFun(function (d,t) return funcOnTensors(d, function (x) return x:cuda() end), t end)
            
            --local jitt = nn.JitteringModuleTranslate(6); jitt:training()
            --local jitt = nn.JitteringModuleScale(1/1.1,1.1,false,true); jitt:training()
            --dataset.trainData.postprocFun = function (d,t) return funcOnTensors(d, function (x) return jitt(x):cuda() end), t end ; print('Using jitter on train: '..torch.type(jitt))
            --dataset.testData.postprocFun = function (d,t) return funcOnTensors(d, function (x) return jitt(x):cuda() end), t end ; print('Using jitter on test: '..torch.type(jitt))
        end   
    end
     
    print('<trainer> dataset loaded and normalized (train size ' .. dataset.trainData:size() .. ')')
    return dataset
end

datasets = loadDatasets(opt)

----------------------------------------------------------------------

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(datasets:classes())

local learningDebugger = LearningDebugger()
local learningLogger = LearningLogger(opt.save, opt.showPlots)
local loggingFreqBatch = 40
local loggingFreqSingle = 100


----------------------------------------------------------------------
-- ** training function
local config = {}
if (opt.network ~= '' and opt.networkLoadOpt and opt.numEpochs > 0) then
    config = torch.load(opt.network..'.optconfig')
end


function train(dataset)
    model:training()

    -- local vars
    local ret = {}
    local time = sys.clock()  
    local iter = 0 
    
    -- next epoch
    model.epoch = model.epoch + 1       

    -- do one epoch
    print('<trainer ' .. opt.runName .. '> on training set:')
    print("<trainer> online epoch # " .. model.epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,dataset:size(),opt.batchSize do
        -- disp progress
        xlua.progress(t, dataset:size())
        local time1 = sys.clock()

        -- create mini batch
        local inputs, targets = dataset:trainBatch(opt.batchSize, opt.batchSampling)
        
        if opt.criterion == "mse" then for i = 1,#targets do local t = torch.Tensor(10):zero(); t[targets[i]] = 1; targets[i] = t:typeAs(inputs[1]) end end
        
        --cutorch.synchronize(); print('Loading batch: ' .. (sys.clock() - time1)*1000 .. 'ms')

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
                    -- get new parameters
                    if x ~= parameters then
                        parameters:copy(x)
                    end
        
                    -- reset gradients
                    gradParameters:zero()
     
                    -- f is the average of all criterions
                    local f = 0
                    
                --   learningDebugger:visit(model, 1); 
                --    learningDebugger:reset()                    
                              
                    -- evaluate function for complete mini batch    [btw, experimentally no speed gain in using batches, even with the method of nagadomi?]
                    for i = 1,#inputs do
                        -- estimate f
                        local output = model:forward(inputs[i])                         
                        local err = criterion:forward(output, targets[i])
                        f = f + err
   
                        -- estimate df/dW
                        local df_do = criterion:backward(output, targets[i])                        
                        model:backward(inputs[i], df_do)

                        -- update confusion
                        confusion:add(output, targets[i])
                    end

                    if (opt.batchSize>1 and iter % loggingFreqBatch == 0) then learningDebugger:visit(model, dataset:size()/opt.batchSize/loggingFreqBatch) end
                    if (opt.batchSize==1 and t % loggingFreqSingle == 0) then learningDebugger:visit(model, dataset:size()/loggingFreqSingle) end
 
                    -- normalize gradients and f(X)
                    if opt.batchMeanGrad then
                        gradParameters:div(#inputs)
                        f = f/#inputs
                    end
                    
                    -- return f and df/dX
                    prepareGradPerModule(model, opt)
                    return f,gradParameters
                end

        time1 = sys.clock()
    
        if (opt.batchSize>1 and iter % loggingFreqBatch == 0) or (opt.batchSize==1 and t % loggingFreqSingle == 0) then
            learningLogger:logWeights(parameters)
        end           
        
        -- optimize on current mini-batch
        local optloss, optstep = doOptStep(model, parameters, feval, opt, config)
        
        if (opt.batchSize>1 and iter % loggingFreqBatch == 0) or (opt.batchSize==1 and t % loggingFreqSingle == 0) then
            learningLogger:logLoss(optloss)
            learningLogger:logGradUpd(optstep, config.learningRate)
        end 

        for i,module in ipairs(model:listModules()) do
            if (module.postBackpropHook ~= nil) then
                module:postBackpropHook()           
            end
        end
    
        --cutorch.synchronize(); print('GD: ' .. (sys.clock() - time1)*1000 .. 'ms')
        
        if (opt.batchSize>1 and iter % 3 == 0 or opt.batchSize==1 and t % 30 == 0) then collectgarbage() end    --otherwise can use up whole gpu mem
        
        iter = iter + 1
    end

    -- time taken
    time = sys.clock() - time
    time = time / dataset:size()
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    --print(confusion)
    confusion:updateValids()
    print('++ global correct: ' .. (confusion.totalValid*100) .. '%')
    ret['meanAccuracy'] = confusion.totalValid * 100
    confusion:zero()
   
    learningDebugger:reset()
   
    -- save/log current net (a clean model without any gradients, inputs,... ; it takes less storage)
    cleanModelSave(model, parameters, config, opt, 'cifar.net')

    return ret
end


----------------------------------------------------------------------
-- ** test function
function test(dataset)
    model:evaluate()

    -- local vars
    local ret = {}
    local time = sys.clock()

    -- averaged param use (ASGD)?
    local cachedparams
    if config.ax then
        cachedparams = parameters:clone()
        parameters:copy(config.ax)
    end

    -- test over given dataset
    print('<trainer ' .. opt.runName .. '> on testing Set:')
    for t = 1,dataset:size(),dataset.nSampleVariants do
        -- disp progress
        xlua.progress(t, dataset:size())

        -- get and test new sample
        local input, target = dataset:at(t)
        local pred = model:forward(input)
        
        -- avg vote if using augmentation (~nagadomi)
        -- (btw, this is correct: loss of avg, not avg loss. I can't treat the augmented images as independent test samples; see also last work of Kokkinos).
        for i = 2,dataset.nSampleVariants do
            local input, targetI = dataset:at(t+i-1)
            assert(targetI==target)
            pred = pred + model:forward(input)
        end
        
        confusion:add(pred / dataset.nSampleVariants, target)
    end

    -- timing
    time = sys.clock() - time
    time = time / dataset:size()
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    --print(confusion)
    confusion:updateValids()
    print('++ global correct: ' .. (confusion.totalValid*100) .. '%')
    --for t = 1,confusion.nclasses do
    --   local pclass = string.format('%2.3f', confusion.valids[t] * 100)
    --   print(pclass .. ',' .. confusion.classes[t])
    --end
    ret['meanAccuracy'] = confusion.totalValid * 100
    confusion:zero()

    -- averaged param use (ASGD)?
    if config.ax  then
        -- restore parameters
        parameters:copy(cachedparams)
    end
    return ret
end

----------------------------------------------------------------------
-- and train&test!
--

local earlyStop = 1
while model.epoch <= opt.numEpochs do

    -- nagadomi ref
    --[[if model.epoch == 10 then
        opt.learningRateDecay = 0
        opt.learningRate = 0.001
    end--]]

    -- train/test
    local trainRet = train(datasets.trainData)
    local validRet = test(datasets.validData)
    local testRet = test(datasets.testData)
    
    learningLogger:logAccuracy(trainRet['meanAccuracy'], validRet['meanAccuracy'], testRet['meanAccuracy'])
    learningLogger:plot()
    
    -- run max 2 more epochs if training set saturated (measured by accuracy, not loss!)
    if (trainRet['meanAccuracy']==100) then
        if (earlyStop<=0) then
            print('Stopping training, accuracy 100% reached')
            break
        else
            earlyStop = earlyStop - 1
        end
    else
        earlyStop = 1
    end    
end


torch.manualSeed(opt.seed); if doCuda then cutorch.manualSeed(opt.seed) end --for objective testing (e.g. when any JitteringModule used)
if opt.v then
    print('<valid> (epoch '..model.epoch..')')
    test(datasets.validData)
else    
    print('<test> (epoch '..model.epoch..')')
    test(datasets.testData)
end

--[[print('<test> pertDown')
opt.scPerturbMin = 0.9; opt.scPerturbMax = 1
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)
print('<test> pertUp')
opt.scPerturbMin = 1; opt.scPerturbMax = 1.1
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)--]]


--[[print('<test> pertDown')
opt.scPerturbMin = 0.5; opt.scPerturbMax = 1
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)
print('<test> pertUp')
opt.scPerturbMin = 1; opt.scPerturbMax = 2
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)--]]


--[[print('<test> s0.5')
opt.scPerturbBlurMin=0.5; opt.scPerturbBlurMax=0.5
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)

print('<test> s1')
opt.scPerturbBlurMin=1; opt.scPerturbBlurMax=1
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)

print('<test> s1.5')
opt.scPerturbBlurMin=1.5; opt.scPerturbBlurMax=1.5
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)

print('<test> s2')
opt.scPerturbBlurMin=2; opt.scPerturbBlurMax=2
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)--]]


--[[--for pyraFactor = 0.7
opt.scPerturbMin = 1/0.7; opt.scPerturbMax = 1/0.7
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)
opt.scPerturbMin = 1/0.49; opt.scPerturbMax = 1/0.49
datasets = nil; collectgarbage(); datasets = loadDatasets(opt)
test(datasets.testData)--]]
