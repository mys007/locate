--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local ffi = require 'ffi'
local Threads = require 'threads'

local function doDonkey(opt)
    if opt.dataset=='imagenet' then
    	paths.dofile('donkeyImagenet.lua')
    elseif opt.dataset=='pascal3d' then
    	paths.dofile('donkeyP3D.lua')
    else
    	assert(false, 'unknown dataset')
  	end 	
end

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
      --   function()
      --      require 'torch'
       --  end,
         function(idx)
         	require 'torch'
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local tseed = opt.seed + idx
            torch.manualSeed(tseed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, tseed))
            doDonkey(opt)
            torch.manualSeed(tseed) --reseed again, don't depend on whether dataset loaded/created
         end
      );
   else -- single threaded data loading. useful for debugging
      doDonkey(opt)
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

datasetInfo = {}

donkeys:addjob(function() return trainLoader.classes end, function(c) datasetInfo.classes = c end)
donkeys:addjob(function() return trainLoader:sizeTrain() end, function(c) datasetInfo.nTrain = c end)
donkeys:synchronize()
datasetInfo.nClasses = #datasetInfo.classes
assert(datasetInfo.nClasses, "Failed to get nClasses")
--torch.save(paths.concat(opt.save, 'classes.t7'), datasetInfo.classes)

donkeys:addjob(function() return testLoader:sizeTest() end, function(c) datasetInfo.nTest = c end)
donkeys:synchronize()
assert(datasetInfo.nTest > 0, "Failed to get nTest")

print('nClasses: ', datasetInfo.nClasses, 'nTrain: ', datasetInfo.nTrain, 'nTest: ', datasetInfo.nTest)
