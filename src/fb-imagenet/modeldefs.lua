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
    
    	model = torch.load('/home/simonovm/workspace/medipatch/szagoruyko/siam_notredame_nn.t7')	
        for i,module in ipairs(model.modules[1]:listModules()) do
            if (module.weight ~= nil) then
                module.lrFactorW = opt.modelParams['lrtower'] or 1
                module.lrFactorB = opt.modelParams['lrtower'] or 1
            end  
        end     	
 
    elseif opt.modelName == '2ch2d' then

        model = torch.load('/home/simonovm/workspace/medipatch/szagoruyko/2ch_notredame_nn.t7')    
        
        model.modules[1].lrFactorW = opt.modelParams['lrd1'] or 1
        model.modules[1].lrFactorB = opt.modelParams['lrd1'] or 1
        model.modules[4].lrFactorW = opt.modelParams['lrd2'] or 1
        model.modules[4].lrFactorB = opt.modelParams['lrd2'] or 1
        model.modules[7].lrFactorW = opt.modelParams['lrd3'] or 1
        model.modules[7].lrFactorB = opt.modelParams['lrd3'] or 1
        model.modules[10].lrFactorW = opt.modelParams['lrd4'] or 1
        model.modules[10].lrFactorB = opt.modelParams['lrd4'] or 1                        

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
    else
        assert(false)
    end

    return model, criterion
end


