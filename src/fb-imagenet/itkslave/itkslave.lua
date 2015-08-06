require 'torch'
require 'paths'
local ffi = require 'ffi'

ffi.cdef[[
void getSourceBox(float axis1, float axis2, float axis3, float angle, float scale, THFloatTensor* dest);
void transformVolume(float axis1, float axis2, float axis3, float angle, float scale, float center1, float center2, float center3, THFloatTensor* src, THFloatTensor* dst);
]]

local itkslave = {}

itkslave.C = ffi.load(paths.dirname(paths.thisfile())..'/build/libitkslave.so')
local C = itkslave.C

function itkslave.getSourceBox(axis, angle, scale)
	local res = torch.FloatTensor()
	C.getSourceBox(axis[1], axis[2], axis[3], angle, scale, res:cdata())
	return res
end

function itkslave.transformVolume(axis, angle, scale, center, image)
	local res = torch.FloatTensor()	
	if not image:isContiguous() then image = image:clone() end
	C.transformVolume(axis[1], axis[2], axis[3], angle, scale, center[1], center[2], center[3], image:cdata(), res:cdata())
	return res
end

return itkslave
