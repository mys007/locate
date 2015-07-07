--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
require 'myutils'
paths.dofile('dataset.lua')
require 'util'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = '/home/simonovm/datasets/cache/donkeyModapairs.trainCache.t7'
local testCache = '/home/simonovm/datasets/cache/donkeyModapairs.testCache.t7'
local meanstdCache = '/home/simonovm/datasets/cache/donkeyModapairs.meanstdCache.t7'
local datapath = '/home/simonovm/datasets/IXI/patches'
local patchdir = '/home/simonovm/datasets/IXI/patches/patches'
local modalitiesext = {'T1.t7img', 'T2.t7img'}

local sampleSize = {2, opt.patchSize, opt.patchSize}
local maxIntersection = 0.1
local maxBlacks = sampleSize[2]*sampleSize[3]/2

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std

-- Check for existence of opt.data
if not os.execute('cd ' .. datapath) then
    error(("could not chdir to '%s'"):format(opt.data))
end

--------------------------------
local function loadImage(path)
    assert(path~=nil)
    local input = (paths.extname(path)=='t7img') and torch.load(path) or image.load(path)
    if input:dim() == 2 then -- 1-channel image loaded as 2D tensor
        input = input:view(1,input:size(1), input:size(2))
    elseif input:dim() == 3 and input:size(1) == 1 then -- 1-channel image
    elseif input:dim() == 3 and input:size(1) == 3 then -- 3-channel image
        input = input[1]
    else
        print(#input)
        error('not 2-channel or 3-channel image')
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
local function processImagePair(dataset, path, nSamples, traintime)
    assert(traintime~=nil)
    collectgarbage()
    local input1, input2 = loadImagePair(path)
    local iW, iH = input1:size(3), input1:size(2)  
    local oW = sampleSize[3]
    local oH = sampleSize[2]   

    local output = torch.Tensor(nSamples, 2, oW, oH)
    local doPos = paths.basename(paths.dirname(path)) == 'pos'

    for s=1,nSamples do
        
        if false then 
            --todo: rotation: extract bigger patch surroundings (sqrt(2)-bigger), rotate it and crop centerpart. Actually, ideally should sample affine transformation-and-crop
        else
            local out1, out2
            local ok = false
                        
            for a=1,1000 do
                local x1 = math.ceil(torch.uniform(1e-2, iW-oW))
                local y1 = math.ceil(torch.uniform(1e-2, iH-oH))
                out1 = input1[{{},{y1,y1 + oH-1}, {x1,x1 + oW-1}}]

                if not doPos then 
                    -- rejective sampling for neg position (can't overlap too much)
                   for b=1,1000 do    
                        local x2 = math.ceil(torch.uniform(1e-2, iW-oW))
                        local y2 = math.ceil(torch.uniform(1e-2, iH-oH))
                        local inter, union = rectIntersectionUnion(x1,y1,oW,oH,x2,y2,oW,oH)
                        if inter/oW/oH < maxIntersection then
                            out2 = input2[{{},{y2,y2 + oH-1}, {x2,x2 + oW-1}}]
                            ok = true
                            break                 
                        end
                    end    
                else 
                    out2 = input2[{{},{y1,y1 + oH-1}, {x1,x1 + oW-1}}]
                    ok = true
                end
                
                -- ignore invalid patches (partial registration, missing values in one patch; don't assume any default val)
                if ok and (torch.any(torch.lt(out1,0)) or torch.any(torch.lt(out2,0))) then
                    ok = false
                end    
                    
                -- ignore boring black patch pairs (they could be both similar and dissimilar)
                -- TODO: maybe uniform patches are bad, so check for std dev
                if ok and torch.lt(out1,1e-6):sum()>maxBlacks and torch.lt(out2,1e-6):sum()>maxBlacks then
                    ok = false
                end  
                
                if ok then break end
            end
            
            assert(ok, 'too many bad attemps, something went wrong with sampling from '..path)
            local out = output[s]
            out[1]:copy(out1)
            out[2]:copy(out2)
            
            -- mean/std
            for i=1,2 do -- channels/modalities
                if mean then out[{{i},{},{}}]:add(-mean[i]) end
                if std then out[{{i},{},{}}]:div(std[i]) end
            end      
            
            -- optionally flip
            if traintime then
                if torch.uniform() > 0.5 then out = image.hflip(out) end
                --TODO: vflips and 90deg rots
            end        
            
            if false and not doPos then
                image.display{image=out[1], zoom=2, legend='Input1', padding=1, nrow=1}
                image.display{image=out[2], zoom=2, legend='Input2', padding=1, nrow=1}     
                print(doPos)            
            end        
        end

    end

   return output
end

--------------------------------------------------------------------------------
--[[ Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
]]--

-- function to load the image pair
local trainHook = function(self, path)
    return processImagePair(self, path, opt.numTSPatches, true)
end

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(datapath, 'train')},
      loadSize = {3, 256, 256},
      sampleSize = sampleSize,
      forceClasses = {[1] = 'pos', [2] = 'neg'},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

-- End of train loader section
--------------------------------------------------------------------------------

local hash = require 'hash'
local hashstate = hash.XXH64()

--[[ Section 2: Create a test data loader (testLoader),
   which can iterate over the test set--]]

-- function to load the image (seed set in a test-sample specific way, repeatable)
local testHook = function(self, path)
    local rngState = torch.getRNGState()
    torch.manualSeed(hashstate:hash(path))
    local out = processImagePair(self, path, 1, false)  
    torch.setRNGState(rngState)
    return out
end

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
else
   print('Creating test metadata')
   testLoader = dataLoader{
      paths = {paths.concat(datapath, 'val')},
      loadSize = {3, 256, 256},
      sampleSize = {3, 224, 224},
      split = 0,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local meanEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,3 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,3 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,3 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,3 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end
print('Mean: ', mean[1], mean[2], mean[3], 'Std:', std[1], std[2], std[3])

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
