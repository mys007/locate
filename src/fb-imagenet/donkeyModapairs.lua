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

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = os.getenv('HOME')..'/datasets/cache/locate'..opt.datasetPostfix..'DonkeyModapairs.trainCache_s'..opt.trainSplit..'.t7'
local testCache = os.getenv('HOME')..'/datasets/cache/locate'..opt.datasetPostfix..'DonkeyModapairs.testCache.t7'
local meanstdCache = os.getenv('HOME')..'/datasets/cache/locate'..opt.datasetPostfix..'DonkeyModapairs.meanstdCache_s'..opt.trainSplit..'_'..opt.inputMode..'.t7'
local datapath = os.getenv('HOME')..'/datasets/Locate/patches'..opt.datasetPostfix
local patchdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset'

local shadesrange = {4,21}
local sampleSize = {3+3, opt.patchSize, opt.patchSize}
if opt.inputMode=='depth' then sampleSize[1] = 3 + 1
elseif opt.inputMode=='allshades' then sampleSize[1] = 3 + 3*(shadesrange[2]-shadesrange[1]+1)
elseif opt.inputMode=='allshadesG' then sampleSize[1] = 3 + shadesrange[2]-shadesrange[1]+1 end

local filecache = {}

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std

-- Check for existence of opt.data
if not os.execute('cd ' .. datapath) then
    error(("could not chdir to '%s'"):format(opt.data))
end

-------------------------------
-- load photo and synthetic images (hack: in database, there are just filenames)
local function loadImagePair(path)
    assert(path~=nil)
    
    local entry = mtcache:load(paths.basename(path,'t7img'))

    if entry then
        return unpack(entry)
    else  
        local input1 = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), 'maps', 'photo_crop.png')) --4 channels (alpha)
        local depthedges = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), 'maps', 'distance_crop.pfm_edges.png'))
        local input2
        
        if opt.inputMode=='allshades' or opt.inputMode=='allshadesG' then
            local nCh = opt.inputMode=='allshades' and 3 or 1
            for i=shadesrange[1],shadesrange[2] do
                local im = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), 'maps', string.format('panorama_crop_%02d.png', i)), nCh)
                input2 = input2 or torch.Tensor((shadesrange[2]-shadesrange[1]+1)*nCh,im:size(im:dim()-1),im:size(im:dim()))
                input2:narrow(1, nCh*(i-shadesrange[1])+1, nCh):copy(im)
            end
        elseif opt.inputMode=='camnorm' then
            input2 = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), 'maps', 'normalsCamera_crop.png'), 3)
        elseif opt.inputMode=='depth' then
            input2 = torch.load(paths.concat(patchdir, paths.basename(path,'t7img'), 'maps', 'distance_crop.t7img.gz')):decompress()
            --TODO: use log domain [note that sky<0]? should sky by e.g. +40000? 
        elseif opt.inputMode=='all' then
            --TODO (allshadesG, camnorm, depth)
        else
            assert(false)
        end
        
        -- cache samples by name; we have small DB
        mtcache:store(paths.basename(path,'t7img'), {input1, input2, depthedges})
        
        return input1, input2, depthedges
    end
end




--------------------------------
-- samples a quadratic patch (random side length, random position, random rotation; all uniform)
-- [scale and rot may additionally get jittered in extractPatch()]
local function samplePatch(input)
    local side = math.ceil(opt.patchSize * torch.uniform(opt.patchSampleMinScaleF, opt.patchSampleMaxScaleF))
    local x1 = math.ceil(torch.uniform(1e-2, input:size(3)-side))
    local y1 = math.ceil(torch.uniform(1e-2, input:size(2)-side))
    local rot = torch.uniform(-math.pi, math.pi) * opt.patchSampleRotMaxPercA              
    return {p={{}, {y1,y1 + side-1}, {x1,x1 + side-1}}, r=rot}
end

--------------------------------
-- Extracts a 2D patch as given by the indices
-- Optionally performs randomized rotation (uniform d) and scaling (normal d) jittering.
-- Surrounding data need to be available, doesn't do any zero-padding.
-- Note that bilinear interpolation introduces smoothing artifacts 
local function extractPatch(input, indices, jitter)
    local patchEx
    
    if indices.r~=0 or (jitter and (opt.patchJitterRotMaxPercA > 0 or opt.patchJitterMaxScaleF > 1)) then
    
        local patchJitterRotMaxPercA = jitter and opt.patchJitterRotMaxPercA or 0
        local patchJitterMaxScaleF = jitter and opt.patchJitterMaxScaleF or 1
        local iters = jitter and 100 or 1
         
        -- determine available space around patch
        local availablePad = 1e10
        for i=2,3 do
            if indices.p[i][1]~=indices.p[i][2] then
                availablePad = math.min(availablePad, math.min(indices.p[i][1] - 1, input:size(i) - indices.p[i][2]))
            end
        end
        
        -- sample rotation and scaling jitter until we fit into the available space
        local ok = false
        local patchSize = indices.p[2][2] - indices.p[2][1] + 1
        local alpha, sc, requiredPad = 0, 1, 0
        for a=1,iters do
            alpha = indices.r + torch.uniform(-math.pi, math.pi) * patchJitterRotMaxPercA
            if patchJitterMaxScaleF > 1 then
                sc = torch.normal(1, (patchJitterMaxScaleF-1)/2) --in [1/f;f] with 95% prob
                sc = math.max(math.min(sc, patchJitterMaxScaleF), 1/patchJitterMaxScaleF)
            end        

            -- norm distance of box corner point to rot center (not tight, but ok)
            local rotFactor = math.max(math.abs(math.cos(alpha-math.pi/4)), math.abs(math.sin(alpha-math.pi/4))) / math.cos(math.pi/4)          
            requiredPad = math.ceil( patchSize/2 * (sc*rotFactor - 1) )
            if requiredPad < availablePad then
                ok = true
                break
            end        
        end
        if not ok then return false, false end

        patchEx = input[boxPad(indices.p, requiredPad, 0)]
        
        -- rotate & crop center
        if (alpha ~= 0) then
            patchEx = image.rotate(patchEx, alpha, 'bilinear')
            local s = math.ceil((patchEx:size(2) - sc*patchSize)/2)
            local cidx = {s, s + math.floor(sc*patchSize)-1}
            patchEx = patchEx[{{}, cidx, cidx}]
        end
    else
        patchEx = input[indices.p]
    end
   
    -- scale
    patchEx = image.scale(patchEx, opt.patchSize, opt.patchSize, 'bilinear')
       
    return patchEx, true
end

--------------------------------
local function plotPatches(out)
    image.display{image=out:sub(1,3), zoom=2, legend='Input1', padding=1, nrow=1}
    image.display{image=out:sub(4,-1), zoom=2, legend='Input2', padding=1, nrow=math.ceil(math.sqrt(out:size(1)-3))}
end  

local function writePatches(out,i)
    local function dispAndZoom(src, zoom)
        local img = image.toDisplayTensor{input=src, min=0, max=1, padding=1, nrow=math.ceil(math.sqrt(src:size(1)-3))}
        return image.scale(img, img:size(img:dim())*zoom, img:size(img:dim()-1)*zoom, 'simple')
    end

    local plotpath = '/home/simonovm/tmp'
    image.save(plotpath..'/p' .. i.. '_1.png', dispAndZoom(out:sub(1,3),2))
    image.save(plotpath..'/p' .. i.. '_2.png', dispAndZoom(out:sub(4,-1),2))
end


--------------------------------
local function processImagePair(dataset, path, nSamples, traintime)
    assert(traintime~=nil)
    
    collectgarbage()
    local input1, input2, depthedges = loadImagePair(path)
    local output = torch.Tensor(nSamples, input1:size(1)-1+input2:size(1), opt.patchSize, opt.patchSize)
    local extrainfo = opt.sampleWeightMode ~= '' and torch.Tensor(nSamples, 1) or nil
    local doPos = paths.basename(paths.dirname(path)) == 'pos'
    --local map = {input1:narrow(1,1,3):clone(), input2:narrow(1,1,3):clone()}
    
    for s=1,nSamples do
        local out1, out2
        local in1idx, in2idx
        local ok = false
                    
        for a=1,1000 do
            in1idx = samplePatch(input1)
            
            if not doPos then 
                -- rejective sampling for neg position
                for b=1,1000 do    
                    in2idx = samplePatch(input2)

                    if opt.patchSampleNegDist=='center' then
                        local meanside = (in1idx.p[2][2] - in1idx.p[2][1] + 1)/2 + (in2idx.p[2][2] - in2idx.p[2][1] + 1)/2
                        local reldist = boxCenterDistance(in1idx.p, in2idx.p) / (math.sqrt(2)*meanside)                
                        local distLimit = opt.patchSampleNegThres
                        ok = (distLimit>0 and reldist >= distLimit) or (distLimit<0 and reldist <= -distLimit and reldist > 0)
                    elseif opt.patchSampleNegDist=='inter' then
                        local inter, union = boxIntersectionUnion(in1idx.p, in2idx.p)
                        local iou = inter / union   --note: <1 if one box is included in other, even if centered
                        local limit = opt.patchSampleNegThres
                        ok = (limit>0 and iou <= limit) or (limit<0 and iou >= -limit and iou < 1) or (limit==0 and iou < 1)                             
                    else
                        assert(false)
                    end       
 
                    if ok then            
                        out2, ok = extractPatch(input2, in2idx)
                        --map[2][in2idx.p]:fill(0)                     
                        break
                    end
                end    
            else          
                out2, ok = extractPatch(input2, in1idx, true)
            end
            
            -- prefer patches where there is structure in out2 (rejection sampling)
            --[[if false and ok then --just for positives?!
                local std
                if opt.inputMode=='allshades' then std = out2[(12-shadesrange[1]+1)*3]:std() --12 o'clock (least shades)
                elseif opt.inputMode=='allshadesG' then std = out2[12-shadesrange[1]+1]:std() --dtto
                else std = out2:std() end
                
                --TODO: sample uniform value in some range and check if std above.   []but even a black patch has to have chance!]
                
                image.display{image=out2, zoom=2, legend='Input2', padding=1, nrow=1}
                print(std)
            end   --]]         
     
            if ok then
                out1, ok = extractPatch(input1, in1idx)
                --map[1][in1idx.p]:fill(0)      
            end
            
            -- ignore patches with transparency (incomplete data)
            if ok then
                ok = not torch.any(torch.lt(out1[4],1))
            end

            if ok then break end
        end
        
        assert(ok, 'too many bad attemps, something went wrong with sampling from '..path)
        out1 = out1:narrow(1,1,3) --drop alpha-layer
        
        --set learning rate/importance for the sample
        if opt.sampleWeightMode=='synthstd' then
            local stddev
            if opt.inputMode=='allshades' then stddev = out2[(12-shadesrange[1]+1)*3]:std() --12 o'clock (least shades)
            elseif opt.inputMode=='allshadesG' then stddev = out2[12-shadesrange[1]+1]:std() --dtto
            else stddev = (out2[1]:std()+out2[2]:std()+out2[3]:std())/3 end
            extrainfo[s][1] = stddev-- * opt.sampleWeightFactor --then approx in [0,1]
        elseif opt.sampleWeightMode=='dedges' then
            local dedges = extractPatch(depthedges, in2idx or in1idx)
            extrainfo[s][1] = math.max(opt.sampleWeightFactor, dedges:max())
        end
        
        -- optionally jitter photometric properties of rgb
        if traintime then
            if opt.patchSampleClrJitter>0 and torch.uniform() > 0.5 then
                local img = image.rgb2hsv(out1)
                for c=1,3 do img[c]:add(torch.uniform(-opt.patchSampleClrJitter, opt.patchSampleClrJitter)):clamp(0,1) end
                out1 = image.hsv2rgb(img)
            end
        end         
        
        local out = output[s]
        out:narrow(1,1,3):copy(out1)
        out:narrow(1,4,out2:size(1)):copy(out2)
 
        -- mean/std
        for i=1,out:size(1) do -- channels/modalities
            if mean then out[i]:add(-mean[i]) end
            if std then out[i]:div(std[i]) end
        end      
        
        -- optionally flip
        if traintime then
            if torch.uniform() > 0.5 then out = image.hflip(out) end   
        end
        
        -- optionally warp (both mods the same, don't want to be invariant to this in pos and for negs doesn't matter)
        if traintime then
            if opt.patchSampleWarpF > 0 and torch.uniform() > 0.5 then
                local maxdisp = opt.patchSize*opt.patchSampleWarpF
                local field = torch.Tensor(2, math.ceil(torch.uniform(2, maxdisp/2)), math.ceil(torch.uniform(2, maxdisp/2))):uniform(-maxdisp, maxdisp)
                field = image.scale(field, opt.patchSize, opt.patchSize) --generate smooth vector field
                out = image.warp(out, field, 'lanczos')
            end 
        end
        
        if false and doPos then
            writePatches(out,s)
            --plotPatches(out)
            print(doPos)
        end
    end

    --image.display{image=map[1]}; image.display{image=map[2]}
    return {d=output, e=extrainfo}
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
      split = opt.trainSplit,   --NOTE: validation data are not really independent, pos can be in train and neg in valid!
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
   local nSamples = 50
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples*opt.numTSPatches .. ' randomly sampled training images')
   local meanEstimate = {}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,img:size(2) do for k=1,img:size(1) do
         meanEstimate[j] = (meanEstimate[j] or 0) + img[k][j]:mean()
      end end
   end
   for j=1,#meanEstimate do
      meanEstimate[j] = meanEstimate[j] / (nSamples*opt.numTSPatches)
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples*opt.numTSPatches .. ' randomly sampled training images')
   local stdEstimate = {}
   for i=1,nSamples do
      local img = trainLoader:sample(1)
      for j=1,img:size(2) do for k=1,img:size(1) do
         stdEstimate[j] = (stdEstimate[j] or 0) + img[k][j]:std()
      end end
   end
   for j=1,#stdEstimate do
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
print('Mean: ', mean, 'Std:', std)
