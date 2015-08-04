--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
package.path = "../myrock/?.lua;" .. package.path
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'strict'
require 'myrock'
require 'LearningDebugger'

local opts = paths.dofile('opts.lua')

opt, cmd = opts.parse(arg)

-- cuda
print('Will use device '..opt.device)
cutorch.setDevice(1) --opt.device)
torch.setdefaulttensortype('torch.FloatTensor') -- shouldn't use 'torch.CudaTensor', breaks libraries (e.g. image)

-- fix seed for model init
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

paths.dofile('data.lua')
if opt.numIters>0 then opt.numEpochs = math.ceil(opt.numIters * opt.batchSize / datasetInfo.nTrain) end

paths.dofile('model.lua')

local _,fname = sys.fpath()
opt.save = paths.concat(opt.save, fname:gsub('.lua','') .. '-' .. opt.modelName, opt.runName)
os.execute('mkdir -p "' .. opt.save .. '"')
opts.log(paths.concat(opt.save, 'stdout.log'), opt)

print('<cmdline> CUDA_VISIBLE_DEVICES='..(opt.device-1)..' qlua main.lua ' .. table.concat(arg, " "))
print('Using model with ' .. parameters:nElement() .. ' parameters')
print(model)

learningLogger = LearningLogger(opt.save, false)

require 'util'
paths.dofile('train.lua')
paths.dofile('test.lua')

-- fix seed for learning (don't depend on model-specific actions)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

local earlyStop
while model.epoch < opt.numEpochs do

    -- train/test
    local trainRet = train()
    local validRet = (datasetInfo.nValid>0 and opt.doValidation) and test(false) or {}
    local testRet = test(true)
    
    learningLogger:logAccuracy(trainRet['meanAccuracy'], validRet['meanAccuracy'] or 0, testRet['meanAccuracy'])
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


torch.manualSeed(opt.seed) --for objective testing (e.g. when any JitteringModule used)
cutorch.manualSeed(opt.seed)
if opt.v then
    print('<valid> (epoch '..model.epoch..')')
    test(false)
elseif opt.t then  
    print('<test> (epoch '..model.epoch..')')
    test(true)
end



