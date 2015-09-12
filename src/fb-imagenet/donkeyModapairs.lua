--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
package.path = "../myrock/?.lua;../?.lua;" .. package.path
require 'image'
require 'myutils'
paths.dofile('dataset.lua')
require 'util'
require 'torchzlib'
local pe = require 'patchExtraction'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = os.getenv('HOME')..'/datasets/cache/donkeyModapairs.trainCache_s'..opt.trainSplit..'_T1T2.t7'
local testCache = os.getenv('HOME')..'/datasets/cache/donkeyModapairs.testCache_T1T2.t7'
local meanstdCache = os.getenv('HOME')..'/datasets/cache/donkeyModapairs.meanstdCache_s'..opt.trainSplit..'_T1T2.t7'
local datapath = os.getenv('HOME')..'/datasets/IXI'
local patchdir = os.getenv('HOME')..'/datasets/IXI/volumes'
local modalitiesext = {'T1.t7img.gz', 'T2.t7img.gz'}

local sampleSize, maxBlacks
if pe.isVol then
    sampleSize = {2, opt.patchSize, opt.patchSize, opt.patchSize}
    maxBlacks = sampleSize[2]*sampleSize[3]*sampleSize[4]*opt.patchSampleMaxBlacks
else
    sampleSize = {2, opt.patchSize, opt.patchSize}
    maxBlacks = sampleSize[2]*sampleSize[3]*opt.patchSampleMaxBlacks
end    

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std

-- Check for existence of opt.data
if not os.execute('cd ' .. datapath) then
    error(("could not chdir to '%s'"):format(opt.data))
end

--------------------------------
local function loadImage(path)
    assert(path~=nil)
    local input = string.ends(path,'t7img') and torch.load(path) or (string.ends(path,'t7img.gz') and torch.load(path):decompress() or image.load(path))
    if input:dim() == 2 then -- 1-channel image loaded as 2D tensor
    elseif input:dim() == 3 and input:size(1) == 1 then -- 1-channel image
        input = input:view(input:size(2), input:size(3))
    elseif input:dim() == 3 and input:size(1) == 3 then -- 3-channel image
        input = input[1]
    end
    return input
end

--------------------------------
-- only one modality is indexed by dataset. The other modalities can be found in patchdir
local function loadImagePair(path)
    assert(path~=nil)   
    local input1 = loadImage(path)
    local input2 = loadImage(paths.concat(patchdir, string.sub(paths.basename(path),1,-string.len(modalitiesext[1])-1)..modalitiesext[2]))
    return input1, input2
end

--------------------------------
local function extractPatch(input, indices, transfpar)
    assert(input and indices)
    local out, transfpar = pe.extractPatch(input, indices, transfpar)
    -- ignore invalid patches (partial registration, missing values in one patch; don't assume any default val)
    local allvalid = not torch.any(torch.lt(out,0))
    return out, allvalid, transfpar 
end

--------------------------------
local function processImagePair(dataset, path, nSamples, traintime)
    assert(traintime~=nil)
    
    collectgarbage()
    local input1, input2 = loadImagePair(path)
    local oW, oH, oD
    if pe.isVol then oW, oH, oD = sampleSize[4], sampleSize[3], sampleSize[2] else oW, oH, oD = sampleSize[3], sampleSize[2], 1 end

    local output = pe.isVol and torch.Tensor(nSamples, 2, oD, oH, oW) or torch.Tensor(nSamples, 2, oW, oH)
    local doPos = paths.basename(paths.dirname(path)) == 'pos'

    for s=1,nSamples do
        local out1, out2
        local ok = false
                    
        for a=1,1000 do
            local in1idx = pe.samplePatch(oW, oH, oD, input1)
            local out2transfpar = nil
            
            if not doPos then 
                -- rejective sampling for neg position (can't overlap too much; also don't get too close between slices [->inflate])
               for b=1,1000 do    
                   local in2idx = pe.samplePatch(oW, oH, oD, input2)
                   
                   if opt.patchSampleNegDist=='center' then
                        local reldist = boxCenterDistance(in1idx, in2idx) / opt.patchSize
                        local distLimit = opt.patchSampleNegThres
                        ok = (distLimit>0 and reldist >= distLimit) or (distLimit<0 and reldist <= -distLimit and reldist > 0)
                   elseif opt.patchSampleNegDist=='inter' then
                        local inter
                        if pe.isVol then
                            inter = boxIntersectionUnion(in1idx, in2idx) /oW/oH/oD
                        else
                            local pad3d = sampleSize[2]/10/2
                            inter = boxIntersectionUnion(boxPad(in1idx, 0, pad3d), boxPad(in2idx, 0, pad3d)) /oW/oH                                        
                        end
                        local maxIntersection = opt.patchSampleNegThres
                        ok = (inter < maxIntersection)
                    else
                        assert(false)
                    end
 
                    if ok then                     
                        out2, ok = extractPatch(input2, in2idx)
                        break
                    end
                end    
            else          
                out2, ok, out2transfpar = extractPatch(input2, in1idx)
            end
            
            if ok then
                out1, ok = extractPatch(input1, in1idx, opt.patchSamplePosSameTransf and out2transfpar or nil)
            end
                
            -- ignore boring black patch pairs (they could be both similar and dissimilar)
            -- TODO: maybe uniform patches are bad, so check for std dev
            if ok then
                local o1b, o2b = torch.lt(out1,1e-6):sum()>maxBlacks, torch.lt(out2,1e-6):sum()>maxBlacks
                if (opt.patchSampleBlackPairs and o1b and o2b) or (not opt.patchSampleBlackPairs and (o1b or o2b)) then
                    ok = false
                end
            end
            
            if ok then break end
        end
        
        assert(ok, 'too many bad attemps, something went wrong with sampling from '..path)
        local out = output[s]
        out[1]:copy(out1)
        out[2]:copy(out2)
        
        -- mean/std
        for i=1,2 do -- channels/modalities
            if mean then out[i]:add(-mean[i]) end
            if std then out[i]:div(std[i]) end
        end      
        
        -- optionally flip
        if traintime then
            for d=2,opt.patchDim+1 do
                if torch.uniform() > 0.5 then out = image.flip(out,d) end   
            end 
        end        
        
        if false and doPos then
            pe.plotPatches(out)
            print(doPos)
        end        
    end

    return output
end

--------------------------------------------------------------------------------
-- function to load the image pair
local trainHook = function(self, path)
    return processImagePair(self, path, opt.numTSPatches, true)
end

--------------------------------------------------------------------------------
-- function to load the image (seed set in a test-sample specific way, repeatable)
local hash = require 'hash'
local hashstate = hash.XXH64()

local testHook = function(self, path)

    --TODO: not that this uses same constraints on sampling, ie. no 'nearly positives'!

    local rngState = torch.getRNGState()
    torch.manualSeed(opt.seed-1 + hashstate:hash(path))
    local out = processImagePair(self, path, opt.numTestSPatches, false)  
    torch.setRNGState(rngState)
    return out
end

--------------------------------------------------------------------------------
--[[ Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
]]--

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleSize = sampleSize
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(datapath, 'train')},
      loadSize = {2, 256, 256},
      sampleSize = sampleSize,
      forceClasses = {[1] = 'pos', [2] = 'neg'}, --(dataLoader can't handle -1)
      split = opt.trainSplit,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
end
trainLoader.sampleHookTrain = trainHook
trainLoader.sampleHookTest = testHook --(validation)
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

--[[ Section 2: Create a test data loader (testLoader),
   which can iterate over the test set--]]

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleSize = sampleSize
else
   print('Creating test metadata')
   testLoader = dataLoader{
      paths = {paths.concat(datapath, 'test')},
      loadSize = {2, 256, 256},
      sampleSize = sampleSize,
      split = 0,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
collectgarbage()
-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   trainLoader.meanstd = meanstd
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 500
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples*opt.numTSPatches .. ' randomly sampled training images')
   local meanEstimate = {0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,2 do for k=1,img:size(1) do
         meanEstimate[j] = meanEstimate[j] + img[k][j]:mean()
      end end
   end
   for j=1,2 do
      meanEstimate[j] = meanEstimate[j] / (nSamples*opt.numTSPatches)
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples*opt.numTSPatches .. ' randomly sampled training images')
   local stdEstimate = {0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,2 do for k=1,img:size(1) do
         stdEstimate[j] = stdEstimate[j] + img[k][j]:std()
      end end
   end
   for j=1,2 do
      stdEstimate[j] = stdEstimate[j] / (nSamples*opt.numTSPatches)
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   trainLoader.meanstd = cache
   print('Time to estimate:', tm:time().real)
   
    do -- just check if mean/std look good now
       local testmean = 0
       local teststd = 0
       for i=1,100 do
          local img = trainLoader:sample(1)
          testmean = testmean + img:mean()
          teststd  = teststd + img:std()
       end
       print('Stats of 100 randomly sampled images after normalizing. Mean: ' .. testmean/100 .. ' Std: ' .. teststd/100)
    end      
end
print('Mean: ', mean[1], mean[2], 'Std:', std[1], std[2])
