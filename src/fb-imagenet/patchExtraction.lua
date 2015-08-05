local patchExtraction = {}

assert(opt.patchDim==3 or opt.patchDim==2)
patchExtraction.isVol = opt.patchDim==3

if patchExtraction.isVol then
    __threadid  = 0
    os.execute("sleep " .. tonumber(__threadid)*3) --needs just one initialization
    python = require "lunaticpython"
    python.execute("import sys; sys.dont_write_bytecode = True") --don't clutter with pyc
    local patchEx3D = python.import("patchEx3D")
    python.execute("import numpy as np")
    
    
    
    
local ffi=require 'ffi'

ffi.cdef[[
void THFloatStorage_clearFlag(THFloatStorage *storage, const char flag);
]]

function nparrayToTensor(npAr, shared)
    assert(npAr.dtype.name == 'float32')
    local ptr = tonumber(ffi.cast('intptr_t',npAr.__array_interface__['data'][0]))
    local shape = npAr.shape
    local nelem = npAr.size
    local dims = npAr.ndim
    
    local sizes = torch.LongStorage(dims)
    for i=1,dims do sizes[i] = shape[i-1] end
    
    if shared then 
        local storage = torch.FloatStorage(nelem, ptr)
        ffi.C['THFloatStorage_clearFlag'](ffi.cast('THFloatStorage*', torch.pointer(storage)), 6) -- TH_STORAGE_RESIZABLE & TH_STORAGE_FREEMEM
        
        local tensor =  torch.FloatTensor(storage, 1, sizes)
        --tensor.pyref = npAr --prevent npAr from being gc-ed (and thus the py-object possibly freed) while the storage is in use. Doesn't work!
        return {tensor, npAr}
    else
        local tensor = torch.FloatTensor(sizes)
        ffi.copy(torch.data(tensor), ffi.cast('void*',ptr), nelem*4)
        return tensor
    end    
    --[[assert(python.eval(pyName..".dtype.name") == 'float32')
    local ptr = tonumber(ffi.cast('intptr_t',python.eval(pyName..".__array_interface__['data'][0]")))
    local shape = python.eval(pyName..".shape")
    local nelem = python.eval(pyName..".size")
    local dims = python.eval(pyName..".ndim")
    
    local sizes = torch.LongStorage(dims)
    for i=1,dims do sizes[i] = shape[i-1] end
    
    if shared then 
        local storage = torch.FloatStorage(nelem, ptr)
        ffi.C['THFloatStorage_clearFlag'](ffi.cast('THFloatStorage*', torch.pointer(storage)), 6) -- TH_STORAGE_RESIZABLE & TH_STORAGE_FREEMEM
        return torch.FloatTensor(storage, 1, sizes)
    else
        local tensor = torch.FloatTensor(sizes)
        ffi.copy(torch.data(tensor), ffi.cast('void*',ptr), nelem*4)
        return tensor
    end--]]
end

   ---NOTE: this can then be moved to C-code  (can use C intrafce of numpy, like wrapper from ITK)
   ----NOTE: the other way round: http://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy , http://stackoverflow.com/questions/7543675/how-to-convert-pointer-to-c-array-to-python-array
    

    --------------------------------
    -- samples oD x oH x oW patch
    function patchExtraction.samplePatch(oW, oH, oD, input)
        local z1 = math.ceil(torch.uniform(1e-2, input:size(1)-oD))
        local y1 = math.ceil(torch.uniform(1e-2, input:size(2)-oH))
        local x1 = math.ceil(torch.uniform(1e-2, input:size(3)-oW))
        return {{z1,z1 + oD-1}, {y1,y1 + oH-1}, {x1,x1 + oW-1}}
    end

    --------------------------------
    -- Extracts a 3D patch from volume.
    -- Optionally performs randomized rotation and scaling (normal d). Surrounding data need to be available, doesn't do any zero-padding.
    -- Note that trilinear (?) interpolation introduces smoothing artifacts 
    function patchExtraction.extractPatch(input, indices)
        if opt.patchSampleRotMaxPercA > 0 or opt.patchSampleMaxScaleF > 1 then 
            local patchCenter = {}
            for i=1,#indices do patchCenter[i] = (indices[i][2] + indices[i][1])/2 end
            --input[patchCenter] = 5000
            
            local srcPatch, srcCenter
            
            -- sample rotation and scaling until we fit into the available space
            local ok = false
            local alpha, axis, sc = 0, {1,0,0}, 1
            for a=1,20 do
                if opt.patchSampleRotMaxPercA > 0 then
                    alpha = torch.uniform(-math.pi, math.pi) * opt.patchSampleRotMaxPercA
                    axis = {torch.normal(0,1), torch.normal(0,1), torch.normal(0,1)}
                end
                if opt.patchSampleMaxScaleF > 1 then
                    sc = torch.normal(1, (opt.patchSampleMaxScaleF-1)/2) --in [1/f;f] with 95% prob
                    sc = math.max(math.min(sc, opt.patchSampleMaxScaleF), 1/opt.patchSampleMaxScaleF)
                end
                
                --compute inverse transformation of a axis-aligned box centered at (0,0,0) with vertices at points like (1,1,1) 
                -- to get the source area, the bounding box of which we need to crop (defined as box, at least as big as destbox)
                local pyBox = patchEx3D.get_source_box(axis[1], axis[2], axis[3], alpha, sc)
                local luaBox = nparrayToTensor(pyBox, true)[1]
                local mi = torch.cmin(luaBox:min(1):squeeze(), -1) * opt.patchSize/2
                local ma = torch.cmax(luaBox:max(1):squeeze(), 1) * opt.patchSize/2
                local srcIndices = {}
                for i=1,#indices do srcIndices[i] = {math.floor(patchCenter[i] + mi[i]), math.ceil(patchCenter[i] + ma[i])} end
                
                --try to crop it             
                ok, srcPatch = pcall(function() return input[srcIndices] end)
                if ok then
                    srcCenter = {} --patchCenter in srcPatch coordinates 
                    for i=1,#indices do srcCenter[i] = patchCenter[i] - srcIndices[i][1] end
                    break
                end    
            end
            
            if not ok then return input[indices]:squeeze() end
        
            --transform the crop
            local buffPy = python.eval("np.empty(("..srcPatch:size(1)..','..srcPatch:size(2)..','..srcPatch:size(3)..'), dtype=np.float32 )')
            local buffLua = nparrayToTensor(buffPy, true) --todo: this should be more like "create np.array for patchEx"
            buffLua[1]:copy(srcPatch)
            patchEx3D.transform_volume(axis[1], axis[2], axis[3], alpha, sc, buffPy, srcCenter[1], srcCenter[2], srcCenter[3])
            
            --finally, extract just the center crop of the result
            local cidx = {}
            for i=1,#indices do cidx[i] = {math.ceil(srcCenter[i] - opt.patchSize/2 + 1), math.floor(srcCenter[i] + opt.patchSize/2 + 1)} end
            local patchEx = buffLua[1][cidx]:clone() 
                        
            --image.display{image=srcPatch, zoom=4, legend='Input1', padding=1, nrow=math.ceil(math.sqrt(srcPatch:size(1)))}
            --image.display{image=buffLua, zoom=4, legend='Input2', padding=1, nrow=math.ceil(math.sqrt(buffLua:size(1)))} 
            --image.display{image=patchEx, zoom=4, legend='Fin', padding=1, nrow=math.ceil(math.sqrt(patchEx:size(1)))}  
            --print(alpha, sc, axis[1], axis[2], axis[3])          
               
            return patchEx
        
        else
            return input[indices]:squeeze()
        end    
    end
    
    function patchExtraction.plotPatches(out)
        image.display{image=out[1], zoom=2, legend='Input1', padding=1, nrow=math.ceil(math.sqrt(out[1]:size(1)))}  
        image.display{image=out[2], zoom=2, legend='Input2', padding=1, nrow=math.ceil(math.sqrt(out[1]:size(1)))}  
    end  



else
    --------------------------------
    -- samples oH x oW patch from random slice of random dimension (in case of volumetric input)
    function patchExtraction.samplePatch(oW, oH, oD, input)
        assert(oD <= 1) 
        if input:dim()==3 then
            local dim = math.ceil(torch.uniform(1e-2, 3))
            local sliceidx = math.ceil(torch.uniform(1e-2, input:size(dim)))
            local sizes = torch.totable(input:size())        
            table.remove(sizes, dim)
            local x1 = math.ceil(torch.uniform(1e-2, sizes[2]-oW))
            local y1 = math.ceil(torch.uniform(1e-2, sizes[1]-oH))                    
            local indices = {{y1,y1 + oH-1}, {x1,x1 + oW-1}}
            table.insert(indices,dim,{sliceidx, sliceidx})
            return indices
        else
            local x1 = math.ceil(torch.uniform(1e-2, input:size(2)-oW))
            local y1 = math.ceil(torch.uniform(1e-2, input:size(1)-oH))                
            return {{y1,y1 + oH-1}, {x1,x1 + oW-1}}
        end 
    end

    --------------------------------
    -- Extracts a 2D patch from volume.
    -- Optionally performs randomized rotation (uniform d) and scaling (normal d). Surrounding data need to be available, doesn't do any zero-padding.
    -- Note that bilinear interpolation introduces smoothing artifacts 
    function patchExtraction.extractPatch(input, indices)
        if opt.patchSampleRotMaxPercA > 0 or opt.patchSampleMaxScaleF > 1 then 
            -- determine available space around patch
            local availablePad = 1e10
            for i=1,#indices do
                if indices[i][1]~=indices[i][2] then
                    availablePad = math.min(availablePad, math.min(indices[i][1] - 1, input:size(i) - indices[i][2]))
                end
            end
            
            -- sample rotation and scaling until we fit into the available space
            local ok = false
            local alpha, sc, requiredPad = 0, 1, 0
            for a=1,100 do
                alpha = torch.uniform(-math.pi, math.pi) * opt.patchSampleRotMaxPercA
                if opt.patchSampleMaxScaleF > 1 then
                    sc = torch.normal(1, (opt.patchSampleMaxScaleF-1)/2) --in [1/f;f] with 95% prob
                    sc = math.max(math.min(sc, opt.patchSampleMaxScaleF), 1/opt.patchSampleMaxScaleF)
                end     --sc has here the inverse meaning as in the 3D version:)
                
                -- norm distance of box corner point to rot center (not tight, but ok)
                local rotFactor = math.max(math.abs(math.cos(alpha-math.pi/4)), math.abs(math.sin(alpha-math.pi/4))) / math.cos(math.pi/4)          
                requiredPad = math.ceil( opt.patchSize/2 * (sc*rotFactor - 1) )
                if requiredPad < availablePad then
                    ok = true
                    break
                end        
            end
            if not ok then return input[indices]:squeeze() end
        
            local patchEx = input[boxPad(indices, requiredPad, 0)]:squeeze()
            
            -- rotate & crop center
            if (alpha ~= 0) then
                patchEx = image.rotate(patchEx, alpha, 'bilinear')
                local s = math.ceil((patchEx:size(1) - sc*opt.patchSize)/2)
                local cidx = {s, s + math.floor(sc*opt.patchSize)-1}
                patchEx = patchEx[{cidx, cidx}]
            end
    
            -- scale
            if (sc ~= 1) then
                patchEx = image.scale(patchEx, opt.patchSize, opt.patchSize, 'bilinear')
            end    
    
            return patchEx
        
        else
            return input[indices]:squeeze()
        end
    end
    
    function patchExtraction.plotPatches(out)
        image.display{image=out[1], zoom=2, legend='Input1', padding=1, nrow=1}
        image.display{image=out[2], zoom=2, legend='Input2', padding=1, nrow=1}  
    end    
end


return patchExtraction