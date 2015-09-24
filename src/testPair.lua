require 'cunn'
require 'cudnn'
package.path = 'myrock/?.lua;' .. package.path
require('myrock')
require 'libmattorch'
require 'xlua'
require 'pl'


opt = lapp[[
   -n        (default "/home/simonovm/workspace/E/locate/main-2ch2d/20150914-012907-allshadesG-p64-th07-lr1e2-ex")      netpath
   -s        (default "zumsteinspitze")          setname
   -e        (default "")        	experiment name
   -o        (default 1)      		whether to use orig image (1) or _crop image (0)
   --step	 (default 0.1)		relative step size (image1)
   --win	 (default 0.5)      relative window size (to input2)
]]


local netpath = opt.n
local setname = opt.s
local expername = opt.e

local patchdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset'
local shadesrange = {4,21}
local opath = ''



--------------------
--as in donkeyModapairs
local function loadImagePair(path)
    local input1 = image.load(paths.concat(patchdir, paths.basename(path,'t7img'), 'maps', 'photo_crop.png')) --4 channels (alpha)
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
    
    return input1, input2
end

--------------------
function loadNetwork(path)
	cutorch.setDevice(1)
	torch.setdefaulttensortype('torch.FloatTensor')
	torch.manualSeed(1)
	cutorch.manualSeed(1)
	
	local model = torch.load(path)
	model = model:cuda()
	
	--handle legacy
	if not model.inputMode then model.inputMode = 'allshadesG' end

	print('Loaded '..path)
	print(model)
	return model
end

--------------------
-- Dense evaluation of similarity for aligned image pairs
function densePairEval(model, input1, input2)
    local sz = model.inputDim[3]
    local img = torch.CudaTensor(model.inputDim[1], input1:size(2), input1:size(3))
    img:narrow(1,1,3):copy(input1:narrow(1,1,3)) --(drop alpha-layer)
    img:narrow(1,4,input2:size(1)):copy(input2)
    
    local scores = torch.CudaTensor(1, input1:size(2), input1:size(3)):fill(0)
    local inputGpu = torch.CudaTensor(img:size(2)-sz+1, img:size(1), sz, sz)
    
    for x = 1,img:size(3)-sz+1 do
        for y = 1,img:size(2)-sz+1 do
            inputGpu[y]:copy(img:narrow(2,y,sz):narrow(3,x,sz))
        end
        
        model:forward(inputGpu)
        local output = (model.outputDim == 1) and model.output or model.output:select(2,1)
        scores:narrow(2,sz/2, inputGpu:size(1)):select(3, x+sz/2):copy(output)
        
        --TODO: should probably filter out patches with transparency?
 
        if x%10==0 or x==img:size(3)-sz+1 then 
            libmattorch.saveTensor(opath .. '/dense_scores-'..setname..'-'..expername..'.mat', scores:double())
        end
        xlua.progress(x,img:size(3)-sz+1)
    end 
end




-- TODO: compute dense matches. sample grid of points in im1 and evaluate neighborhood in im2, store best (or whole img:P)   [i..e x1 y1 x2 y2 score]
-- (should give more precise results than samling grids in both and then matching)
-- (works on both aligned and unalingned imgs)

function pointMatches(model, input1, input2, stepPerc, winSizePerc)
    -- make images of same scale (normalize by y), downscale the bigger
    -- this is necessary unless the network is super-invariant to scale changes in positives
    local unevenFactor = 1 --1.5
    local sf1, sf2 = 1,1
    if input1:size(2) > input2:size(2) then
        sf1 = input2:size(2) / input1:size(2) *unevenFactor
        input1 = image.scale(input1, input1:size(3)*sf1, input2:size(2) *unevenFactor, 'bilinear')
    else
        sf2 = input1:size(2) / input2:size(2) *unevenFactor
        input2 = image.scale(input2, input2:size(3)*sf2, input1:size(2) *unevenFactor, 'bilinear')    
    end
    
    local alpha = input1[4]
    input1 = input1:narrow(1,1,3):cuda() --(drop alpha-layer)
    input2 = input2:cuda()
    
    local step = {0, math.ceil(stepPerc * input1:size(2)), math.ceil(stepPerc * input1:size(3))}
    local halfwin = {0, math.ceil(winSizePerc/2 * input2:size(2)), math.ceil(winSizePerc/2 * input2:size(3))}
    local sz = model.inputDim[3]
    local inputGpu = torch.CudaTensor(1, model.inputDim[1], sz, sz)    
    local scores = {}
 
    for x1 = 1, input1:size(3)-sz+1, step[3] do
        for y1 = 1, input1:size(2)-sz+1, step[2] do
        
    --for x1 = 722,722 do
      --  for y1 = 22,22 do   
      
      		if not torch.any(torch.lt(alpha:narrow(1,y1,sz):narrow(2,x1,sz),1)) then --only valid  
		    
		        local bestMatch = {0,0,-1e10}
		        local scoremap = torch.CudaTensor(1, input2:size(2), input2:size(3)):fill(-1e20)

				-- fully conv evalution not possible for 2ch nets
		        for x2 = math.max(1, x1-halfwin[3]), math.min(input2:size(3)-sz+1, x1+halfwin[3]) do
		            
		            local i = 1
		            local coords = {}
		            local rangeY = {math.max(1, y1-halfwin[2]), math.min(input2:size(2)-sz+1, y1+halfwin[2])}
		            inputGpu:resize(rangeY[2]-rangeY[1]+1, model.inputDim[1], sz, sz)        
		        
		            for y2 = rangeY[1], rangeY[2] do
		                inputGpu[i]:narrow(1,1,3):copy( input1:narrow(2,y1,sz):narrow(3,x1,sz) )
		                inputGpu[i]:narrow(1,4,input2:size(1)):copy( input2:narrow(2,y2,sz):narrow(3,x2,sz) )
		                coords[i] = y2
		                i = i + 1
		            end
		            
		            model:forward(inputGpu)
		            local output = (model.outputDim == 1) and model.output or model.output:select(2,1)
		            
		            local s,i = model.output:max(1)
		            s, i = s:squeeze(), i:squeeze()
		            if s > bestMatch[3] then bestMatch = {(x2+sz/2)/sf2, (coords[i]+sz/2)/sf2, s} end
		            
		            scoremap:narrow(2, rangeY[1] + sz/2, inputGpu:size(1)):select(3, x2 + sz/2):copy(output)         
		        end

		        table.insert(scores, {(x1+sz/2)/sf1, (y1+sz/2)/sf1, unpack(bestMatch)})
		  
		        libmattorch.saveTensor(opath .. '/matches-'..setname..'-scoremap-'..x1..','..y1..expername..'.mat', scoremap:double())
		        image.save(opath .. '/matches-'..setname..'-scoremap-'..x1..','..y1..expername..'.png', input1:narrow(2,y1,sz):narrow(3,x1,sz):float())      
	        end       
        end
        
        libmattorch.saveTensor(opath .. '/matches-'..setname..expername..'.mat', torch.DoubleTensor(scores))
        xlua.progress(x1,input1:size(3)-sz+1)            
    end
        
    libmattorch.saveTensor(opath .. '/matches-'..setname..expername..'.mat', torch.DoubleTensor(scores))
end




local model = loadNetwork(paths.concat(netpath, 'network.net'))

opt.inputMode = model.inputMode
opath = paths.concat(netpath, 'plots_ep'..model.epoch)
paths.mkdir(opath)


local input1, input2 = loadImagePair(setname)
if opt.o>0 then input1 = image.load(paths.concat(patchdir, paths.basename(setname,'t7img'), 'maps', 'orig_image.jpg')) end

print('Loaded input of size '..input1:size(2)..'x'..input1:size(3))

--densePairEval(model, input1, input2)

pointMatches(model, input1, input2, opt.step, opt.win)
