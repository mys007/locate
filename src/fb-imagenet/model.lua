--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'opthelper'

----------------------------------------------------------------------
-- define model to train

paths.dofile('modeldefs.lua')
--xpcall(function() createModel(opt) end, breakpt)
model, criterion = createModel(opt)

prepareModel(model, opt)

-- Load prev model?
if opt.network ~= '' then
    if not opt.networkJustAsInit then
        print('<trainer> reloading previously trained network')
        model = torch.load(opt.network) --createModel() was called before so we have all our classes registered as factory   
    else
        print('<trainer> initializing with previously trained network')
        local tmp = torch.load(opt.network)
        model:getParameters():copy(tmp:getParameters())  --just copy the learned params to the new model
        if opt.networkLoadOpt then model.epoch = tmp.epoch end
    end
end

--model.modules[2]:reset(); model.modules[5]:reset()

cleanModelInit(model, opt) --just for saving

model:cuda()
criterion:cuda()

moduleSharing(model, opt)

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()
model.epoch = model.epoch or 0

