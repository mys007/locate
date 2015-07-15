--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
paths.dofile('dataset.lua')
require 'util'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = os.getenv('HOME')..'/datasets/cache/ILSVRC2012_256_trainCache_s'..opt.trainSplit..'.t7'
local testCache = os.getenv('HOME')..'/datasets/cache/ILSVRC2012_256_testCache.t7'
local meanstdCache = os.getenv('HOME')..'/datasets/cache/ILSVRC2012_256_meanstdCache_s'..opt.trainSplit..'.t7'
local datapath = os.getenv('HOME')..'/datasets/ILSVRC2012_256'

-- Check for existence of datapath
if not os.execute('cd ' .. datapath) then
    error(("could not chdir to '%s'"):format(datapath))
end

local loadSize   = {3, 256, 256}
local sampleSize = {3, 224, 224}

local function loadImage(path)
   local input = image.load(path)
   if input:dim() == 2 then -- 1-channel image loaded as 2D tensor
      input = input:view(1,input:size(1), input:size(2)):repeatTensor(3,1,1)
   elseif input:dim() == 3 and input:size(1) == 1 then -- 1-channel image
      input = input:repeatTensor(3,1,1)
   elseif input:dim() == 3 and input:size(1) == 3 then -- 3-channel image
   elseif input:dim() == 3 and input:size(1) == 4 then -- image with alpha
      input = input[{{1,3},{},{}}]
   else
      print(#input)
      error('not 2-channel or 3-channel image')
   end
   -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, 256, 256 * iH / iW)
   else
      input = image.scale(input, 256 * iW / iH, 256)
   end
   return input
end


-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[ Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
]]--

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local iW, iH = input:size(3), input:size(2)  

   -- do random crop
   local oW = sampleSize[3];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)

   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   -- mean/std
   for i=1,3 do -- channels
      if mean then out[{{i},{},{}}]:add(-mean[i]) end
      if opt.imagenetCaffeInput==false and std then out[{{i},{},{}}]:div(std[i]) end
      if opt.imagenetCaffeInput then out[{{i},{},{}}]:mul(255) end
   end
   return out
end

if paths.filep(trainCache) then
   --print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(datapath, 'train')},
      loadSize = {3, 256, 256},
      sampleSize = {3, 224, 224},
      split = opt.trainSplit,
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

--[[ Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
   10 crops (center + 4 corners) and their hflips]]--

-- function to load the image, do 10 crops (center + 4 corners) and their hflips
local testHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   local iW, iH = input:size(3), input:size(2)
   
   local oH = sampleSize[2]
   local oW = sampleSize[3];
   local out = torch.Tensor(10, 3, oW, oH)

   -- mean/std
   for i=1,3 do -- channels
      if mean then input[{{i},{},{}}]:add(-mean[i]) end
      if opt.imagenetCaffeInput==false and std then input[{{i},{},{}}]:div(std[i]) end
      if opt.imagenetCaffeInput then input[{{i},{},{}}]:mul(256) end
   end

   local w1 = math.ceil((iW-oW)/2)
   local h1 = math.ceil((iH-oH)/2)
   out[1] = image.crop(input, w1, h1, w1+oW, h1+oW) -- center patch
   out[2] = image.hflip(out[1])
   h1 = 1; w1 = 1;
   out[3] = image.crop(input, w1, h1, w1+oW, h1+oW)  -- top-left
   out[4] = image.hflip(out[3])
   h1 = 1; w1 = iW-oW;
   out[5] = image.crop(input, w1, h1, w1+oW, h1+oW)  -- top-right
   out[6] = image.hflip(out[5])
   h1 = iH-oH; w1 = 1;
   out[7] = image.crop(input, w1, h1, w1+oW, h1+oW)  -- bottom-left
   out[8] = image.hflip(out[7])
   h1 = iH-oH; w1 = iW-oW;
   out[9] = image.crop(input, w1, h1, w1+oW, h1+oW)  -- bottom-right
   out[10] = image.hflip(out[9])

   return out
end

if paths.filep(testCache) then
   --print('Loading test metadata from cache')
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
   --print('Loaded mean and std from cache.')
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
