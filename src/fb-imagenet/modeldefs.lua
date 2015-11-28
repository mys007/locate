require 'torch'
require 'nn'
require 'nnx'
require 'inn'
require 'image'
require 'myrock'
require 'strict'
--require 'SpatialMaxPoolingCaffe' --legacy
require 'SampleWeighter'
if opt.backend=='cudnn' then require 'cudnn' end

local SpatialMaxPooling = opt.backend=='cudnn' and cudnn.SpatialMaxPooling or nn.SpatialMaxPooling
local SpatialConvolution = opt.backend=='cudnn' and cudnn.SpatialConvolution or nn.SpatialConvolutionMM
local ReLU = opt.backend=='cudnn' and cudnn.ReLU or nn.ReLU

if opt.cudnnMode=='fastest' then
    cudnn.fastest = true
elseif opt.cudnnMode=='benchmark' then
    cudnn.benchmark = true
else
    cudnn.fastest = false; cudnn.benchmark = false
end


function createModel(opt)
    assert(opt ~= nil)
    
    local model = nn.Sequential()
    model.inputDim = datasetInfo.sampleSize
    model.outputDim = (opt.criterion == "bsvm" or opt.criterion == "emb") and 1 or 2
    model.meanstd = datasetInfo.meanstd
    model.inputMode = opt.inputMode
    local criterion   
    local expectedInput = torch.Tensor(1, unpack(datasetInfo.sampleSize)):zero()
        
    if opt.modelName == 'siam2d' or opt.modelName == 'siam3d' then 
    
        -- SZ's models cannot be used directly. for /home/simonovm/workspace/medipatch/szagoruyko/siam_notredame_nn.t7
        -- use -baselineCArch c_96_7_0_0_3,p_2,c_192_5,p_2,c_256_3,join,c_512_1,fin -network /home/simonovm/workspace/medipatch/szagoruyko/siam_notredame_nn.t7 -networkLoadOpt false -networkJustAsInit true
   
        local par
        if opt.criterion == "emb" then 
            model:add(nn.SplitTable(1, #datasetInfo.sampleSize))
            par = nn.ParallelTable()
        else
            par = opt.batchSize>1 and nn.Parallel(2,2) or nn.Parallel(1,1)
        end
        model:add(par)
  
        local towers = {nn.Sequential(), nn.Sequential()} 
        for _,twr in pairs(towers) do
            par:add(twr)
            local size = torch.LongStorage(datasetInfo.sampleSize)
            size[1]=1
            twr:add(nn.Reshape(size, true))
        end
               
        local nPlanes = 1
        
        -- stage 1 : configurable convolutional part
        for token in string.gmatch(opt.baselineCArch, "[^,]+") do
            local mType = nil
            local args = {}
            for a in string.gmatch(string.trim(token), "[^_]+") do
                if mType==nil then mType=a elseif tonumber(a)~=nil then table.insert(args, tonumber(a)) else table.insert(args, a) end
            end
    
            if (mType=='c' or mType=='cb') then        --c,1output_planes,2filter_size,3padding_size,4ignored,5stride,6lrfactorweight,7lrfactorbias
                for _,twr in pairs(towers) do
                    local conv
                    if opt.patchDim==3 and opt.backend=='cunn' then
                        if args[3] and args[3]>0 then
                            for d=2,4 do twr:add(nn.Padding(d, args[3], 4)); twr:add(nn.Padding(d, -args[3], 4)) end
                        end
                        conv = nn.VolumetricConvolution(nPlanes, args[1], args[2], args[2], args[2], args[5] or 1, args[5] or 1, args[5] or 1)
                    elseif opt.patchDim==3 and opt.backend=='cudnn' then
                        conv = cudnn.VolumetricConvolution(nPlanes, args[1], args[2], args[2], args[2], args[5] or 1, args[5] or 1, args[5] or 1, args[3], args[3], args[3])
                    else
                        conv = SpatialConvolution(nPlanes, args[1], args[2], args[2], args[5] or 1, args[5] or 1, args[3])
                    end    
                   
                    twr:add(conv)
                   
                    if (args[6] and args[6]~=1) then conv.lrFactorW = args[6] end
                    if (args[7] and args[7]~=1) then conv.lrFactorB = args[7] end
                    
                    if mType=='cb' then twr:add(nn.SpatialBatchNormalization(args[1])) end
                    
                    twr:add(ReLU(mType~='cb')) --batchnorm has some issues with inplace relu
                end
          
                nPlanes = args[1]          
                
            elseif (mType=='p') then    --p,pooling_factor,stride(optional),pad(optional/cudnn)
                for _,twr in pairs(towers) do
                    if opt.patchDim==3 and opt.backend=='cunn' then
                        twr:add(myrock.CudaAdapter(nn.VolumetricMaxPooling(args[1], args[1], args[1], args[2] or args[1], args[2] or args[1], args[2] or args[1])))
                    elseif opt.patchDim==3 and opt.backend=='cudnn' then
                        twr:add(cudnn.VolumetricMaxPooling(args[1], args[1], args[1], args[2] or args[1], args[2] or args[1], args[2] or args[1], args[3] or 0, args[3] or 0, args[3] or 0))
                    else
                        if (args[1] == math.floor(args[1])) then
                            twr:add(SpatialMaxPooling(args[1], args[1], args[2] or args[1], args[2] or args[1]):ceil())
                        else
                            local sofarsz = model:clone():forward(expectedInput):size()
                            twr:add(nn.SpatialAdaptiveMaxPooling(math.ceil(sofarsz[3]*args[1]-0.5), math.ceil(sofarsz[2]*args[1]-0.5)))
                        end
                    end
                end
                
            elseif (mType=='d') then    --d,dropout_rate    //0=no dropout
                for _,twr in pairs(towers) do
                    twr:add(nn.Dropout(args[1]))           
                end           
                
            elseif (mType=='join') then
                towers = {model}     
                nPlanes = 2*nPlanes
                
            elseif (mType=='pwd') then  --pwd,1norm
                assert(opt.criterion == "emb")
                for _,twr in pairs(towers) do
                    twr:add(opt.batchSize>1 and nn.Select(#datasetInfo.sampleSize+1,1) or nn.Select(#datasetInfo.sampleSize,1))
                    twr:add(opt.batchSize>1 and nn.Select(#datasetInfo.sampleSize,1) or nn.Select(#datasetInfo.sampleSize-1,1))           
                end                
                model:add(nn.PairwiseDistance(args[1] or 2))
                towers = {model}     
                
            elseif (mType=='fin') then  --fin,1lrfactorweight,2lrfactorbias
                model:add(nn.View(-1):setNumInputDims(#datasetInfo.sampleSize))
                local n = opt.backend=='cudnn' and model:clone():cuda():forward(expectedInput:cuda()):nElement() or model:clone():forward(expectedInput):nElement()             
                local lin = nn.Linear(n, model.outputDim)
                if (args[1] and args[1]~=1) then conv.lrFactorW = args[1] end
                if (args[2] and args[2]~=1) then conv.lrFactorB = args[2] end
                model:add(lin)        

            else
                assert(false, 'error in parsing model configuration')
            end
        end

 
    elseif opt.modelName == '2ch2d' or opt.modelName == '2ch3d' then       
        
        -- SZ's models cannot be used directly. for /home/simonovm/workspace/medipatch/szagoruyko/2ch_notredame_nn.t7
        -- use -baselineCArch c_96_7_0_0_3,p_2,c_192_5,p_2,c_256_3,fin -network /home/simonovm/workspace/medipatch/szagoruyko/2ch_notredame_nn.t7 -networkLoadOpt false -networkJustAsInit true
        -- for /home/simonovm/workspace/medipatch/szagoruyko/2chdeep_notredame_nn.t7
        -- use -baselineCArch c_96_4_0_0_3,c_96_3,c_96_3,c_96_3,p_2,c_192_3,c_192_3,c_192_3,fin -network /home/simonovm/workspace/medipatch/szagoruyko/2chdeep_notredame_nn.t7 -networkLoadOpt false -networkJustAsInit true
       
        local nPlanes = model.inputDim[1]
        
        -- stage 1 : configurable convolutional part
        for token in string.gmatch(opt.baselineCArch, "[^,]+") do
            local mType = nil
            local args = {}
            for a in string.gmatch(string.trim(token), "[^_]+") do
                if mType==nil then mType=a elseif tonumber(a)~=nil then table.insert(args, tonumber(a)) else table.insert(args, a) end
            end
            
            if (mType=='c' or mType=='cb') then        --c,1output_planes,2filter_size,3padding_size,4ignored,5stride,6lrfactorweight,7lrfactorbias
                local conv
                if opt.patchDim==3 and opt.backend=='cunn' then
                    if args[3] and args[3]>0 then
                        for d=2,4 do model:add(nn.Padding(d, args[3], 4)); model:add(nn.Padding(d, -args[3], 4)) end
                    end
                    conv = nn.VolumetricConvolution(nPlanes, args[1], args[2], args[2], args[2], args[5] or 1, args[5] or 1, args[5] or 1)
                elseif opt.patchDim==3 and opt.backend=='cudnn' then
                    conv = cudnn.VolumetricConvolution(nPlanes, args[1], args[2], args[2], args[2], args[5] or 1, args[5] or 1, args[5] or 1, args[3], args[3], args[3])
                else
                    conv = SpatialConvolution(nPlanes, args[1], args[2], args[2], args[5] or 1, args[5] or 1, args[3])
                end 
                                
                model:add(conv)
               
                if (args[6] and args[6]~=1) then conv.lrFactorW = args[6] end
                if (args[7] and args[7]~=1) then conv.lrFactorB = args[7] end
                
                if mType=='cb' then model:add(nn.SpatialBatchNormalization(args[1])) end

                model:add(ReLU(mType~='cb')) --batchnorm has some issues with inplace relu
                nPlanes = args[1]             
                
            elseif (mType=='p') then    --p,pooling_factor,stride(optional),pad(optional/cudnn)
                if opt.patchDim==3 and opt.backend=='cunn' then
                    model:add(myrock.CudaAdapter(nn.VolumetricMaxPooling(args[1], args[1], args[1], args[2] or args[1], args[2] or args[1], args[2] or args[1])))
                elseif opt.patchDim==3 and opt.backend=='cudnn' then
                    model:add(cudnn.VolumetricMaxPooling(args[1], args[1], args[1], args[2] or args[1], args[2] or args[1], args[2] or args[1], args[3] or 0, args[3] or 0, args[3] or 0))
                else
                    if (args[1] == math.floor(args[1])) then
                        model:add(SpatialMaxPooling(args[1], args[1], args[2] or args[1], args[2] or args[1]):ceil())
                    else
                        local sofarsz = model:clone():forward(expectedInput):size()
                        model:add(nn.SpatialAdaptiveMaxPooling(math.ceil(sofarsz[3]*args[1]-0.5), math.ceil(sofarsz[2]*args[1]-0.5)))
                    end
                end            
                
            elseif (mType=='d') then    --d,dropout_rate    //0=no dropout
                model:add(nn.Dropout(args[1]))     
                
            elseif (mType=='s') then   --s,sz1,sz2,..,szN  (sz1<sz2,.. and currently sz2=2*sz1). szi = size of the crops before downsampling to sz1.
                if torch.isTypeOf(model:get(model:size()), 'cudnn.ReLU') then model:remove(model:size()); model:add(nn.ReLU()) end
                local branch = model:clone()
                model.modules = {}
                local conc = nn.Concat(2)
                model:add(conc)
                
                --SZ's two-stream net (fovea/iris) as multi-stream net. Crop central region and/or downscale.
                for a=1,#args do
                    local seq = nn.Sequential()
                    local halfcrop = (model.inputDim[2] - args[a]) / 2
                    if halfcrop>0 then seq:add(nn.SpatialZeroPadding(-halfcrop,-halfcrop,-halfcrop,-halfcrop)) end
                    local downf = args[a] / args[1]
                    if a>1 then seq:add(cudnn.SpatialAveragePooling(downf,downf,downf,downf)) end --alt:SpatialScaling
                    if a==1 then 
                        seq:add(branch)
                    else
                        local br = branch:clone()
                        br.pendingSharing = function(self) self:share(branch,'bias','weight','gradBias','gradWeight') end   
                        seq:add(br)
                    end
                    conc:add(seq)
                end     
                nPlanes = nPlanes * #args                
                               
            elseif (mType=='fin') then  --fin,1lrfactorweight,2lrfactorbias
                model:add(nn.View(-1):setNumInputDims(#datasetInfo.sampleSize))
                local n = opt.backend=='cudnn' and model:clone():cuda():forward(expectedInput:cuda()):nElement() or model:clone():forward(expectedInput):nElement()             
                local lin = nn.Linear(n, model.outputDim)            
                if (args[1] and args[1]~=1) then conv.lrFactorW = args[1] end
                if (args[2] and args[2]~=1) then conv.lrFactorB = args[2] end
                model:add(lin)        

            else
                assert(false, 'error in parsing model configuration')
            end
        end     

    else
		paths.dofile('models/' .. opt.modelName .. '_' .. opt.backend .. '.lua')
		return createModel(1), nn.ClassNLLCriterion()
    end 
    
    
    -- loss function:
    if opt.sampleWeightMode ~= '' then
        model:add(nn.SampleWeighter())
    end
    
    if opt.criterion == "nll" then
        model:add(nn.LogSoftMax())
        criterion = nn.ClassNLLCriterion()  --negative log-likelihood
    elseif opt.criterion == "mse" then    
        model:add(nn.SoftMax())
        criterion = nn.MSECriterion()       --mse
    elseif opt.criterion == "svm" then
        criterion = nn.MultiMarginCriterion() --hinge loss
    elseif opt.criterion == "bsvm" then
        criterion =  nn.MarginCriterion()      -- binary hinge loss (expects classes -1,1)
    elseif opt.criterion == "emb" then
        criterion =  nn.HingeEmbeddingCriterion(opt.modelParams['embmargin'] or 1)   -- pos l2 & neg margin (expects classes -1,1)  
    else
        assert(false)
    end

    return model, criterion
end


