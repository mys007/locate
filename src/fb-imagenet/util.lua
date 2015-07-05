local ffi=require 'ffi'
------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

function setFloatStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
   local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THFloatStorage_free'](cstorage)
   end
   local storage = ffi.cast('THFloatStorage*', storage_p)
   tensor:cdata().storage = storage
end

function setLongStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
   local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THLongStorage_free'](cstorage)
   end
   local storage = ffi.cast('THLongStorage*', storage_p)
   tensor:cdata().storage = storage
end

function sendTensor(inputs)
   local size = inputs:size()
   local ttype = inputs:type()
   local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
   inputs:cdata().storage = nil
   return {i_stg, size, ttype}
end

function receiveTensor(obj, buffer)
   local pointer = obj[1]
   local size = obj[2]
   local ttype = obj[3]
   if buffer then
      buffer:resize(size)
      assert(buffer:type() == ttype, 'Buffer is wrong type')
   else
      buffer = torch[ttype].new():resize(size)      
   end
   if ttype == 'torch.FloatTensor' then
      setFloatStorage(buffer, pointer)
   elseif ttype == 'torch.LongTensor' then
      setLongStorage(buffer, pointer)
   else
      error('Unknown type')
   end
   return buffer
end



-- Deterministically serializes donkeys' outputs
-- Motivation: Donkeys call main thread back at random times -> the order of samples presented to optimization is not guarranted -> irreproducible results
-- Solution: Donkeys write their outputs to slots of ring buffer, at a place respective to the thread index. The ringbuffer is cycled sequentially in order 
--           (waiting if the next slot is not ready yet) and a function is called on it. 
local DispatcherRing = torch.class('DispatcherRing')

function DispatcherRing:__init(nSlots)
    assert(nSlots>0)

    self.idxToDispatch = 1
    self.inputsCPU, self.labelsCPU, self.slotReady, self.tSemaphore = {}, {}, {}, {}
    for i=1,nSlots do 
        -- create tensor buffers in main thread. the thread loaders will push their storages to these buffers when done loading    
        self.inputsCPU[i] = torch.FloatTensor()
        self.labelsCPU[i] = torch.LongTensor()
        self.slotReady[i] = false 
        -- create semaphore for blocking a donkey while his slot is still undispatched (we abuse mutex for it)
        local Threads = require 'threads'
        self.tSemaphore[i] = Threads.Mutex()  
    end
end

function DispatcherRing:getSemaphoreIds()
    local ids = {}
    for i=1,#self.tSemaphore do ids[i] = self.tSemaphore[i]:id() end
    return ids
end

function DispatcherRing:receive(tidx, inputsThread, labelsThread)
    assert(tidx<=#self.slotReady)
    
    -- set the data and labels to the main thread tensor buffers (free any existing storage)
    --print('Receiving'..tidx)
    assert(self.slotReady[tidx] == false, 'Slot'..tidx..'is busy')
    receiveTensor(inputsThread, self.inputsCPU[tidx])
    receiveTensor(labelsThread, self.labelsCPU[tidx])    
    self.slotReady[tidx] = true
end

function DispatcherRing:dispatch(fun)
    assert(fun)
    
    while (self.slotReady[self.idxToDispatch]) do
        --print('Dispatching'..self.idxToDispatch)
        local ok = xpcall(function() fun(self.inputsCPU[self.idxToDispatch], self.labelsCPU[self.idxToDispatch]) end, function(err) print(debug.traceback(err)) end)
        if not ok then os.exit() end
        self.slotReady[self.idxToDispatch] = false
        self.tSemaphore[self.idxToDispatch]:unlock()
        
        self.idxToDispatch = self.idxToDispatch % #self.slotReady + 1
    end
end
