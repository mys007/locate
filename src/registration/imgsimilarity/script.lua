require 'nn'
require 'cunn'
package.path = os.getenv('HOME')..'/Multimodal/medipatch/src/myrock/?.lua;' .. package.path
require('myrock')
require('SpatialMaxPoolingCaffe')

local model, inputGpu

function loadNetwork(path)
	cutorch.setDevice(1)
	torch.setdefaulttensortype('torch.FloatTensor')
	torch.manualSeed(1)
	cutorch.manualSeed(1)
	
	model = torch.load(path)
	model = model:cuda()
	
	--handle legacy
	if not model.meanstd then
		model.meanstd = {mean = {445.3861101981, 226.72220925297}, std = {286.5396871777, 162.75464235422}}
	end
	if not model.inputDim then
		for _,module in ipairs(model:listModules()) do
            if torch.typename(module) == 'nn.SpatialConvolutionMM' then model.inputDim = {2, 64, 64}; break end
        end
        if not model.inputDim then model.inputDim = {2, 16, 16, 16} end
	end
	
	print('Loaded '..path)
	print(model)
end

function getInputPatchSize()
	if #model.inputDim == 3 then
		return 1, model.inputDim[2], model.inputDim[3]
	else
		return model.inputDim[2], model.inputDim[3], model.inputDim[4]
	end
end

function forward(input, output)
	inputGpu = inputGpu or input:cuda()
	inputGpu:copy(input)
		
    -- normalize
    for i=1,2 do
        inputGpu:select(2,i):add(-model.meanstd.mean[i])
        inputGpu:select(2,i):div(model.meanstd.std[i])
    end		
	
	xpcall(function() model:forward(inputGpu) end, function(err) print(debug.traceback(err)) end)
	
	if model.outputDim == 1 then 
		output:copy(model.output)
	else
		output:copy(model.output:select(2,1))
	end	
	
	--[[for i=1,input:size(1) do
		for j=1,2 do
			--image.save(os.getenv('HOME')..'/Multimodal/medipatch/src/gridtest/build/luas_'..i..'_'..j..'_'..math.exp(output[i])..'.png',input[i][j]/1000)
			image.save(os.getenv('HOME')..'/Multimodal/medipatch/src/gridtest/build/luas_'..i..'_'..j..'_'..output[i]..'.png',input[i][j]/1000)
		end
	end
	boom()--]]
end

