require 'nn'
require 'cunn'

package.path = "/home/beckerbe/Multimodal/medipatch/src/myrock/?.lua;" .. package.path
require('SpatialMaxPoolingCaffe')

local model, inputGpu

function loadNetwork(path)
	cutorch.setDevice(1)
	torch.setdefaulttensortype('torch.FloatTensor')
	torch.manualSeed(1)
	cutorch.manualSeed(1)
	
	model = torch.load(path)
	model = model:cuda()
	
	print('Loaded '..path)
	print(model)
end

function forward(input, output)
	inputGpu = inputGpu or input:cuda()
	inputGpu:copy(input)
	xpcall(function() output:copy(model:forward(inputGpu)) end, function(err) print(debug.traceback(err)) end)
end

