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
        
    if opt.modelName == 'siam2d' then 
    
        -- SZ's models cannot be used directly. for /home/simonovm/workspace/medipatch/szagoruyko/siam_notredame_nn.t7
        -- use -baselineCArch c_96_7_0_0_3,p_2,c_192_5,p_2,c_256_3,join,c_512_1,fin -network /home/simonovm/workspace/medipatch/szagoruyko/siam_notredame_nn.t7 -networkLoadOpt false -networkJustAsInit true
    
        local par = opt.batchSize>1 and nn.Parallel(2,2) or nn.Parallel(1,1)
        if opt.criterion == "emb" then model:add(nn.SplitTable(1, 3)); par = nn.ParallelTable() end
        model:add(par)
        
        local towers = {nn.Sequential(), nn.Sequential()} 
        for _,twr in pairs(towers) do
            par:add(twr)
            twr:add(nn.Reshape(1,opt.patchSize,opt.patchSize, true))
        end
         
        local nPlanes = 1
        
        -- stage 1 : configurable convolutional part
        for token in string.gmatch(opt.baselineCArch, "[^,]+") do
            local mType = nil
            local args = {}
            for a in string.gmatch(string.trim(token), "[^_]+") do
                if mType==nil then mType=a elseif tonumber(a)~=nil then table.insert(args, tonumber(a)) else table.insert(args, a) end
            end
            
            if (mType=='c') then        --c,1output_planes,2filter_size,3padding_size,4ignored,5stride,6lrfactorweight,7lrfactorbias
                for _,twr in pairs(towers) do
                    local conv = nn.SpatialConvolutionMM(nPlanes, args[1], args[2], args[2], args[5] or 1, args[5] or 1, args[3])
                    twr:add(conv)
                   
                    if (args[6] and args[6]~=1) then conv.lrFactorW = args[6] end
                    if (args[7] and args[7]~=1) then conv.lrFactorB = args[7] end
                    
                    twr:add(nn.ReLU())
                end
          
                nPlanes = args[1]          
                
            elseif (mType=='p') then    --p,pooling_factor,stride(optional)
                for _,twr in pairs(towers) do
                    if (args[1] == math.floor(args[1])) then
                        twr:add(myrock.SpatialMaxPoolingCaffe(args[1], args[1], args[2] or args[1], args[2] or args[1]):ceil(true))
                    else
                        local sofar = model:forward(expectedInput)
                        twr:add(nn.SpatialAdaptiveMaxPooling(math.ceil(sofar:size(3)*args[1]-0.5), math.ceil(sofar:size(2)*args[1]-0.5)))
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
                    twr:add(opt.batchSize>1 and nn.Select(4,1) or nn.Select(3,1))
                    twr:add(opt.batchSize>1 and nn.Select(3,1) or nn.Select(2,1))           
                end                
                model:add(nn.PairwiseDistance(args[1] or 2))
                towers = {model}     
                
            elseif (mType=='fin') then  --fin,1lrfactorweight,2lrfactorbias
                model:add(nn.View(-1):setNumInputDims(3))
                local lin = nn.Linear(nPlanes,1)
                if (args[1] and args[1]~=1) then conv.lrFactorW = args[1] end
                if (args[2] and args[2]~=1) then conv.lrFactorB = args[2] end
                model:add(lin)        

            else
                assert(false, 'error in parsing model configuration')
            end
        end

 
    elseif opt.modelName == '2ch2d' then       
        
        -- SZ's models cannot be used directly. for /home/simonovm/workspace/medipatch/szagoruyko/2ch_notredame_nn.t7
        -- use -baselineCArch c_96_7_0_0_3,p_2,c_192_5,p_2,c_256_3,fin -network /home/simonovm/workspace/medipatch/szagoruyko/2ch_notredame_nn.t7 -networkLoadOpt false -networkJustAsInit true
        local nPlanes = 2
        
        -- stage 1 : configurable convolutional part
        for token in string.gmatch(opt.baselineCArch, "[^,]+") do
            local mType = nil
            local args = {}
            for a in string.gmatch(string.trim(token), "[^_]+") do
                if mType==nil then mType=a elseif tonumber(a)~=nil then table.insert(args, tonumber(a)) else table.insert(args, a) end
            end
            
            if (mType=='c') then        --c,1output_planes,2filter_size,3padding_size,4ignored,5stride,6lrfactorweight,7lrfactorbias
                local conv = nn.SpatialConvolutionMM(nPlanes, args[1], args[2], args[2], args[5] or 1, args[5] or 1, args[3])
                model:add(conv)
               
                if (args[6] and args[6]~=1) then conv.lrFactorW = args[6] end
                if (args[7] and args[7]~=1) then conv.lrFactorB = args[7] end
                
                model:add(nn.ReLU())
                nPlanes = args[1]          
                
            elseif (mType=='p') then    --p,pooling_factor,stride(optional)
                if (args[1] == math.floor(args[1])) then
                    model:add(myrock.SpatialMaxPoolingCaffe(args[1], args[1], args[2] or args[1], args[2] or args[1]):ceil(true))
                else
                    local sofar = model:forward(expectedInput)
                    model:add(nn.SpatialAdaptiveMaxPooling(math.ceil(sofar:size(3)*args[1]-0.5), math.ceil(sofar:size(2)*args[1]-0.5)))
                end
                
            elseif (mType=='d') then    --d,dropout_rate    //0=no dropout
                model:add(nn.Dropout(args[1]))           
                               
            elseif (mType=='fin') then  --fin,1lrfactorweight,2lrfactorbias
                model:add(nn.View(-1):setNumInputDims(3))
                local lin = nn.Linear(nPlanes,1)
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
    if opt.criterion == "nll" then
        model:add(nn.LogSoftMax())
        criterion = nn.ClassNLLCriterion()  --negative log-likelihood
    elseif opt.criterion == "mse" then    
        model:add(nn.SoftMax())
        criterion = nn.MSECriterion()       --mse
    elseif opt.criterion == "svm" then
        criterion = nn.MultiMarginCriterion() --hinge loss
    elseif opt.criterion == "bsvm" then
        criterion =  nn.MarginCriterion()        
    elseif opt.criterion == "emb" then
        criterion =  nn.HingeEmbeddingCriterion(opt.modelParams['embmargin'] or 1)             
    else
        assert(false)
    end

    return model, criterion
end


