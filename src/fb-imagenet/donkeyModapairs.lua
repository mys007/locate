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
require 'lmdb'
--package.path = "/home/simonovm/torch/extra/torch-opencv/?.lua;" .. package.path --temp workaround, no rockspec
--local cv = dofile "/home/simonovm/torch/extra/torch-opencv/cv/init.lua"
--cv.imgproc = dofile "/home/simonovm/torch/extra/torch-opencv/cv/imgproc/init.lua"


-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = os.getenv('HOME')..'/datasets/cache/locate'..opt.datasetPostfix..'DonkeyModapairs.trainCache_s'..opt.trainSplit..'.t7'
local testCache = os.getenv('HOME')..'/datasets/cache/locate'..opt.datasetPostfix..'DonkeyModapairs.testCache.t7'
local meanstdCache = os.getenv('HOME')..'/datasets/cache/locate'..opt.datasetPostfix..'DonkeyModapairs.meanstdCache_s'..opt.trainSplit..'_'..opt.inputMode..'.t7'
local datapath = os.getenv('HOME')..'/datasets/Locate/patches'..opt.datasetPostfix
local patchdir = '/media/simonovm/Slow/datasets/Locate/' .. (string.starts(opt.datasetPostfix,'gt') and opt.datasetPostfix or 'match_descriptors_dataset')

local shadesrange = {4,21}
local sampleSize = {3+3, opt.patchSize, opt.patchSize}
if opt.inputMode=='depth' then sampleSize[1] = 3 + 1
elseif opt.inputMode=='allshades' then sampleSize[1] = 3 + 3*(shadesrange[2]-shadesrange[1]+1)
elseif opt.inputMode=='allshadesG' then sampleSize[1] = 3 + shadesrange[2]-shadesrange[1]+1
elseif opt.inputMode=='shadesnormG' then sampleSize[1] = 3 + 3 + shadesrange[2]-shadesrange[1]+1 end

local filecache = {}

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std

-- Check for existence of opt.data
if not os.execute('cd ' .. datapath) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local db= lmdb.env{
    Path = datapath .. '/'..opt.trainSplit..'_'..opt.inputMode,
    Name = 'trainDB',
    MaxReaders = 30 
}
db:open()
db:reader_check()

--[[lmdb.serialize = function(object) --not working. motivation was to compress memory file, without much copying around
   local f = torch.MemoryFile():binary()
   f:writeObject(object)
   local storage = f:storage()
   f:close()
   
   local compressed = torch.CompressedTensor(torch.CharTensor(storage), 1)
   
   f = torch.MemoryFile():binary()
   f:writeObject(compressed)
   local storage = f:storage()
   f:close()

   return storage:string(), #storage
end

lmdb.deserialize =  function(val, sz, binary)
    local str = ffi.string(val, sz)
   --print(val, sz)
    local storage = torch.CharStorage():string(str)--torch.CharStorage(sz, val)
    --storage:retain()
   local f = torch.MemoryFile(storage, 'r'):binary()   
   local object = f:readObject()
   f:close()
   
   local decomp = object:decompress()
   
   f = torch.MemoryFile(decomp:storage(), 'r'):binary()
   local object = f:readObject()
   f:close()   
   return object
end--]]

-------------------------------
-- load photo and synthetic images (hack: in memory-mapped database shared among processes/threads, there are just filenames in the dir)
local function loadImagePair(path)
    assert(path~=nil)
    
    local timer = torch.Timer()
    timer:reset()
    
    local txn = db:txn(true)
    lmdb.verbose = false
    local entry = txn:get(paths.basename(path,'t7img'))
    lmdb.verbose = true
    txn:abort()
    
    if entry then   
        --return entry[1]:float()/255, entry[2]:float()/255, entry[3]:float()/255
        return entry[1]:decompress():float()/255, entry[2]:decompress():float()/255, entry[3]:decompress():float()/255
        --return entry[1]:decompress(), entry[2]:decompress(), entry[3]:decompress()
    else
        local subdir = string.starts(opt.datasetPostfix,'gt') and 'maps/cyl' or 'maps'
        local input1 = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subdir, 'photo_crop.png')) --4 channels (alpha)
        local depthedges = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subdir, 'distance_crop.pfm_edges.png'))
        local input2
        
        if opt.inputMode=='allshades' or opt.inputMode=='allshadesG' then
            local nCh = opt.inputMode=='allshades' and 3 or 1
            for i=shadesrange[1],shadesrange[2] do
                local im = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subdir, string.format('panorama_crop_%02d.png', i)), nCh)
                input2 = input2 or torch.Tensor((shadesrange[2]-shadesrange[1]+1)*nCh,im:size(im:dim()-1),im:size(im:dim()))
                input2:narrow(1, nCh*(i-shadesrange[1])+1, nCh):copy(im)
            end
        elseif opt.inputMode=='camnorm' then
            input2 = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subdir, 'normalsCamera_crop.png'), 3)
        elseif opt.inputMode=='depth' then
            input2 = torch.load(paths.concat(patchdir, paths.basename(path,'t7img'), subdir, 'distance_crop.t7img.gz')):decompress()
            --TODO: use log domain [note that sky<0]? should sky by e.g. +40000?
        elseif opt.inputMode=='shadesnormG' then
            local nCh = 1
            for i=shadesrange[1],shadesrange[2] do
                local im = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subdir, string.format('panorama_crop_%02d.png', i)), nCh)
                input2 = input2 or torch.Tensor(3+(shadesrange[2]-shadesrange[1]+1)*nCh,im:size(im:dim()-1),im:size(im:dim()))
                input2:narrow(1, nCh*(i-shadesrange[1])+1, nCh):copy(im)
            end        
            input2:narrow(1, input2:size(1)-2, 3):copy( image.load(paths.concat(patchdir, paths.basename(path,'t7img'), subdir, 'normalsCamera_crop.png'), 3) )
        else
            assert(false)
        end
        
        -- cache compressed samples by name
        local txn = db:txn()
        --txn:put(paths.basename(path,'t7img'), {(input1*255):byte(), (input2*255):byte(), (depthedges*255):byte()}, lmdb.C.MDB_NODUPDATA)
        
        txn:put(paths.basename(path,'t7img'), {torch.CompressedTensor((input1*255):byte(), 1), torch.CompressedTensor((input2*255):byte(), 1), torch.CompressedTensor((depthedges*255):byte(), 1)}, lmdb.C.MDB_NODUPDATA)
 
        --8.3GB,~4sec
        --txn:put(paths.basename(path,'t7img'), {torch.CompressedTensor(input1, 1), torch.CompressedTensor(input2, 1), torch.CompressedTensor(depthedges, 1)}, lmdb.C.MDB_NODUPDATA)
        txn:commit()        
 
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
        
            --local r = cv.getRotationMatrix2D{center={patchEx:size(3)/2, patchEx:size(2)/2}, angle=alpha*180/math.pi, scale=1}
            --patchEx = cv.warpAffine{src=patchEx:transpose(1, 3):contiguous(), M=r}:transpose(1, 3)  --slow due to contiguous
            patchEx = image.rotate(patchEx, alpha, 'bilinear') --bilinear major cause of slowdown

            local s = math.ceil((patchEx:size(2) - sc*patchSize)/2)
            local cidx = {s, s + math.floor(sc*patchSize)-1}
            patchEx = patchEx[{{}, cidx, cidx}]
        end
    else
        patchEx = input[indices.p]
    end
   
    --local interp = patchEx:size(2)>opt.patchSize and cv.INTER_LINEAR or cv.CV_INTER_AREA
    --patchEx = cv.resize{src=patchEx:transpose(1, 3):contiguous(), dsize={opt.patchSize, opt.patchSize}, interpolation=interp}:transpose(1, 3):contiguous()  --slow due to contiguous
    patchEx = image.scale(patchEx, opt.patchSize, opt.patchSize, 'bilinear') --bilinear major cause of slowdown
       
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
    image.save(plotpath..'/p' .. i.. '_1x.png', dispAndZoom(out:sub(1,3),2))
    image.save(plotpath..'/p' .. i.. '_2.png', dispAndZoom(out:sub(4,-1),2))
end


local tm1 = torch.Timer():stop()  ;local tm2 = torch.Timer():stop() ;local tm3 = torch.Timer():stop()


--------------------------------
local function processImagePair(dataset, path, nSamples, traintime)
    assert(traintime~=nil)
tm1:resume()  
    collectgarbage()
    local input1, input2, depthedges = loadImagePair(path)
    local output = torch.Tensor(nSamples, input1:size(1)-1+input2:size(1), opt.patchSize, opt.patchSize)
    local extrainfo = opt.sampleWeightMode ~= '' and torch.Tensor(nSamples, 1) or nil
    local doPos = paths.basename(paths.dirname(path)) == 'pos'
    --local map = {input1:narrow(1,1,3):clone(), input2:narrow(1,1,3):clone()}
tm1:stop()    
    for s=1,nSamples do
        local out1, out2
        local in1idx, in2idx
        local ok = false
tm2:resume()                     
        for a=1,10000 do
            in1idx = samplePatch(input1)
            
            if not doPos then 
                -- rejective sampling for neg position
                for b=1,10000 do    
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
                ok = torch.eq(out1[4],0):sum()/out1[4]:numel() < 0.05
            end

            if ok then break end
        end
tm2:stop()
tm3:resume()        
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
tm3:stop()        
        if false and doPos then
            writePatches(out,s)
            --plotPatches(out)
            print(doPos)
        end
 
    end
    --print(tm1:time().real, tm2:time().real, tm3:time().real)
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
