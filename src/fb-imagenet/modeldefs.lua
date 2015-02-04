require 'torch'
require 'nn'
require 'nnx'
require 'inn'
require 'image'
require 'myutils'
require 'JitteringModule'
require 'strict'
require 'SpatialMaxPoolingCaffe'

function createModel(opt)
    assert(opt ~= nil)
    
    local model = nn.Sequential()
    local criterion
    
    if opt.modelName == 'baseline' then
    	
    	-- load imagenet pretrained network
        require 'loadcaffe'
		model = loadcaffe.load('/home/simonovm/caffe/models/bvlc_alexnet/deploy.prototxt', '/home/simonovm/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel', opt.backend == 'cunn' and 'nn' or opt.backend)
    	print(model)
    	
    	-- disable learning for conv layers
    	for i=1,16 do
        	if (model.modules[i].weight ~= nil) then
        		model.modules[i].lrFactorW = 0
        		model.modules[i].lrFactorB = 0
        	end	          
        end
    
    	--replace fc8
    	model.modules[#model.modules] = nil
    	model.modules[#model.modules] = nn.Linear(4096, outputSize)
    			--TODO: softmax will be tricky (need to ignore nonclass data). but in test time? probably fine

    	print(model)
    
    
       --[[
        local nPlanes = opt.expInputSize[1]
        local expectedInput = torch.Tensor(opt.expInputSize[1], opt.expInputSize[2], opt.expInputSize[3]):zero()
        local finallin = true
        
        -- stage 1 : configurable convolutional part
        for token in string.gmatch(opt.baselineCArch, "[^,]+") do
            local mType = nil
            local args = {}
            for a in string.gmatch(string.trim(token), "[^_]+") do
                if mType==nil then mType=a elseif tonumber(a)~=nil then table.insert(args, tonumber(a)) else table.insert(args, a) end
            end
            
            if (mType=='c') then        --c,1output_planes,2filter_size,3padding_size,4share_with__prev_convlayer \\-1=prev,-2=prevprev,..\\,5stride,6lrfactorweight,7lrfactorbias
                local conv = nn.SpatialConvolutionMM(nPlanes, args[1], args[2], args[2], args[5] or 1, args[5] or 1, args[3])
                model:add(conv)

                if (args[4] and args[4]<0) then
                    for i=model:size()-1, 1, -1 do
                        if (torch.typename(model:get(i)) == 'nn.SpatialConvolutionMM') then
                            args[4] = args[4] + 1
                            if (args[4] == 0) then
                                conv.pendingSharing = function(self) self:share(model:get(i),'bias','weight','gradBias','gradWeight') end   
                                break                         
                            end
                        end
                    end
                    assert(args[4]==0, 'Wrong sharing argumet')
                end
                
                if (args[6] and args[6]~=1) then conv.lrFactorW = args[6] end
                if (args[7] and args[7]~=1) then conv.lrFactorB = args[7] end

                
                --hook for Srivastava14a ref impl
                ----conv.postBackpropHook = function(self) self.weight = torch.renorm(self.weight, math.huge, 1, 4) end --for each output neuron
                --conv.postBackpropHook = function(self) remaxnorm(self.weight, 1, 4) end --for each output neuron
               
                model:add(nn.ReLU())
                --model:add(nn.QuantileReLU(0.1))                  
                nPlanes = args[1]
                
            elseif (mType=='p') then    --p,pooling_factor,stride(optional)
                if (args[1] == math.floor(args[1])) then
                    if opt.mpCeil then
                        model:add(nn.CudaAdapter(myrock.SpatialMaxPoolingCaffe(args[1], args[1], args[2] or args[1], args[2] or args[1])))
                    else    
                        model:add(nn.SpatialMaxPooling(args[1], args[1], args[2] or args[1], args[2] or args[1]))
                    end
                else
                    local sofar = model:forward(expectedInput)
                    model:add(nn.SpatialAdaptiveMaxPooling(math.ceil(sofar:size(3)*args[1]-0.5), math.ceil(sofar:size(2)*args[1]-0.5)))
                end
            
            elseif (mType=='pg') then    --p,pooling_factor,stride(optional)   
                model:add(nn.SpatialAveragePooling(args[1], args[1], args[2] or args[1], args[2] or args[1]))
            elseif (mType=='ps') then    --p,pooling_factor,stride(optional) 
                --if opt.mpCeil then zeroPadMPCeil(model, model:forward(expectedInput):size(3), model:forward(expectedInput):size(2), args[1], args[1], args[2], args[2]) end
                --model:add(nn.CudaAdapter(nn.SpatialStochasticPooling(args[1], args[1], args[2] or args[1], args[2] or args[1])))
                assert(opt.mpCeil)
                model:add(nn.CudaAdapter(inn.SpatialStochasticPooling(args[1], args[1], args[2] or args[1], args[2] or args[1])))

            elseif (mType=='a') then    --a,out_size
                model:add(nn.SpatialAdaptiveMaxPooling(args[1], args[1]))
                
            elseif (mType=='d') then    --d,dropout_rate    //0=no dropout
                model:add(nn.Dropout(args[1]))           
                
            elseif (mType=='n') then    --n
                model:add(createPixelwiseL2Normalization(nPlanes))
                
            elseif (mType=='fin') then
                finallin = false
                
            -- experimental --
            elseif (mType=='e') then    --e,pooling_factor
                --this pools over featmaps, can create alternatives=further nonlinearities, i.e. e.g. sum(w_i*max(x_2*i,x_2*i+1))   ... TODO: see Max-out paper!
                local m, nOutPlanes = createScaleMaxPooling(nPlanes, 1, args[1])
                model:add(m)
                nPlanes = nOutPlanes
                
            elseif (mType=='f') then    --f,output_planes,filter_size,padding_size,if_div_then_1_else_0
                --this adds divisive and multiplicative neurons
                local mConcat = nn.ConcatTable() --over all fanning in scales
                model:add(mConcat)
                mConcat:add(nn.SpatialConvolutionMM(nPlanes, args[1], args[2], args[2], 1, 1, args[3]))
                if (args[4]==1) then
                    local seq = nn.Sequential()
                    mConcat:add(seq)
                    seq:add(nn.SpatialConvolutionMM(nPlanes, args[1], args[2], args[2], 1, 1, args[3]))
                    seq:add(nn.Threshold(1e-6,1e-6))    --todo: tolerant to neg values 
                    model:add(nn.CDivTable())
                else    
                    mConcat:add(nn.SpatialConvolutionMM(nPlanes, args[1], args[2], args[2], 1, 1, args[3]))
                    model:add(nn.CMulTable())
                end    
                model:add(nn.ReLU())            
                nPlanes = args[1]
                
            elseif (mType=='g') then    --g,nhidden
                --this adds multiplicative hidden layer exp(log(x)+log(y))=x*y
                model:add(nn.View(-1):setNumInputDims(3))
                model:add(nn.Threshold(1e-6,1e-6))
                model:add(nn.Log())
                local sofar = model:forward(expectedInput)
                local lin = nn.Linear(sofar:nElement(),args[1])
                lin.weight:fill(0); lin.bias:fill(0) --safe initialization 
                model:add(lin)
                model:add(nn.Exp())
                
            elseif (mType=='q') then     --q,dropout_factor
                model:add(nn.QuantileReLU(args[1], false))   
                
            elseif (mType=='jn') then    --jn,stddev_factor
                model:add(nn.JitteringModuleGNoise(args[1]))
            elseif (mType=='js') then    --js,min_scale_factor,max_scale_factor,fix_ratio?,rndcrop?
                model:add(nn.JitteringModuleScale(args[1],args[2],args[3]>0,args[4]>0))        
            elseif (mType=='jt') then    --jt,max_shift
                model:add(nn.JitteringModuleTranslate(args[1]))
                
            elseif (mType=='os') then     --os,pooling_factor,stride(optional)    
                if opt.mpCeil then zeroPadMPCeil(model, model:forward(expectedInput):size(3), model:forward(expectedInput):size(2), args[1], args[1], args[2], args[2]) end 
                model:add(createOverfeatSubsamplingStart(args[1], args[1], args[2], args[2]))
            elseif (mType=='osa') then    --os,pooling_factor|amp_out_size,stride,type(amp|sp|)
                if opt.mpCeil and args[3]~='amp' then zeroPadMPCeil(model, model:forward(expectedInput):size(3), model:forward(expectedInput):size(2), args[1], args[1], args[2], args[2]) end
                model:add(createOverfeatSubsamplingStartAlt(args[1], args[1], args[2], args[2], args[3], model:forward(expectedInput)))            
            elseif (mType=='oe') then    --oe
                model:add(createOverfeatSubsamplingEnd())     
            elseif (mType=='od') then    --od,dropout_rate    //0=no dropout
                model:add(nn.DropoutBatchConsistent(args[1]))      
            elseif (mType=='osr') then     --os,pooling_factor,stride(optional)    
                if opt.mpCeil then zeroPadMPCeil(model, model:forward(expectedInput):size(3), model:forward(expectedInput):size(2), args[1], args[1], args[2], args[2]) end 
                model:add(createOverfeatSubsamplingRnd(args[1], args[1], args[2], args[2]))         
            elseif (mType=='osrt') then     --os,pooling_factor,stride(optional)    
                if opt.mpCeil then zeroPadMPCeil(model, model:forward(expectedInput):size(3), model:forward(expectedInput):size(2), args[1], args[1], args[2], args[2]) end 
                model:add(createOverfeatSubsamplingRndTestAvgFast(args[1], args[1], args[2], args[2]))   

            else
                assert(false, 'error in parsing model configuration')
            end
        end
        
        -- stage 2 : standard 1-layer neural network   
        model:add(nn.View(-1):setNumInputDims(3))
        if finallin then
            local sofar = model:forward(expectedInput)
            local lin = nn.Linear(sofar:nElement(),opt.nclasses)
            model:add(lin)
        end
        
        model:forward(expectedInput)
        
        --hook for Srivastava14a ref impl
        ----lin.postBackpropHook = function(self) self.weight = torch.renorm(self.weight, math.huge, 1, 4) end --for each output neuron       
        --lin.postBackpropHook = function(self) remaxnorm(self.weight, 1, 4) end --for each output neuron
        
        --]]
        
    else
		paths.dofile('models/' .. opt.modelName .. '_' .. opt.backend .. '.lua')
		return createModel(1), nn.ClassNLLCriterion()
    end 
    
    
    -- loss function:
    if opt.criterion == "nll" then
        model:add(nn.LogSoftMax())
        criterion = nn.ClassNLLCriterion()  --negative log-likelihood
    elseif opt.criterion == "mse" then    
        model:add(nn.SoftMax())
        criterion = nn.MSECriterion()       --mse
    elseif opt.criterion == "svm" then
        criterion = nn.MultiMarginCriterion() --hinge loss
    else
        assert(false)
    end

    return model, criterion
end


