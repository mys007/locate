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
        self.labelsCPU[i] = torch.FloatTensor()
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







--------------------------------------------------------------------------------------
-- A table used as cache synchronized among threads under exclusive access. Tensors contained in the table are shared,
-- tables are copied over (no way how to get a C-pointer to them:()
local MultithreadCache = torch.class('MultithreadCache')

local Ss = require 'threads/sharedserialize' --like serialize but serializes storages as pointers
local Threads = require 'threads'
ffi.cdef[[
void THCharStorage_free(THCharStorage *self);
]]

-- to be called from main thread and the result to be given to constructor in each donkey
function MultithreadCache.createSharedData(nSlots)
    -- held so that it doesn't get gced (esp heldMem)
    assert(not MultithreadCache.heldMutex)
    MultithreadCache.heldMutex = Threads.Mutex()
    MultithreadCache.heldMem = ffi.new('intptr_t[?]', 1+nSlots)
    return {nSlots, MultithreadCache.heldMutex:id(), tonumber(ffi.cast('intptr_t', MultithreadCache.heldMem))}
end

function MultithreadCache:__init(pp)
    self.locCache = {} --per-donkey copy of cache (but tensors within are shared)
    self.nSlots = pp[1]
    self.mutex = Threads.Mutex(pp[2])     
    self.ipcPointer = ffi.cast('intptr_t*', pp[3]) --[0] = shared CharStorage, [1]..[nSlots] = needs_to_update flag
end

local function retainfree(input, retain)
    if torch.isTensor(input) then
        if retain then input:retain() else input:free() end
    elseif torch.type(input) == 'table' then
        for k,v in pairs(input) do retainfree(v, retain) end
    end
end

function MultithreadCache:_updateCacheWhenLocked()
    if self.ipcPointer[__threadid] > 0 then
        assert(__threadid>0 and self.ipcPointer[0] ~= 0)
        local tensor = torch.CharTensor()  
        tensor:cdata().storage = ffi.cast('THCharStorage*', self.ipcPointer[0]) 
        self.locCache = Ss.load(tensor:storage())
        retainfree(self.locCache, true) --mark that we use the tensors too now (Ss.load doesn't do it for us)
        tensor:cdata().storage = nil --don't free the shared storage until told so
        self.ipcPointer[__threadid] = 0
    end
end

function MultithreadCache:load(key)
    if self.nSlots > 0 then
        self.mutex:lock()
        self:_updateCacheWhenLocked()
        self.mutex:unlock()
    end
    return self.locCache[key]
end

function MultithreadCache:store(key, value)
    if self.nSlots > 0 then
        self.mutex:lock()

        self:_updateCacheWhenLocked() --checkout pending changes before writing
        self.locCache[key] = value
        local buff = Ss.save(self.locCache)
        retainfree(self.locCache, false) --undo retain() from Ss; they do it only once, we need to do it as many times there are Ss.load() calls

        --replace the shared storage
        if self.ipcPointer[0] ~= 0 then ffi.C['THCharStorage_free'](ffi.cast('THCharStorage*', self.ipcPointer[0])) end   
        self.ipcPointer[0] = ffi.cast('intptr_t', torch.pointer(buff))
        buff:retain() 
        self.mutex:unlock()
        
        for i=1,self.nSlots do self.ipcPointer[i] = 1 end --tell others to update
        self.ipcPointer[__threadid] = 0        
    else
        self.locCache[key] = value
    end
end
