--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

local cmd = nil

function M.parse(arg)
	
	----------------------------------------------------------------------
	-- parse command-line options
	--
	cmd = myrock.CmdLine()
	cmd:text()
	cmd:text('Torch-7 Imagenet Training script')
	cmd:text()
	cmd:text('Options:')
	cmd:option('-runName', '', '')  --!!!!!!
	cmd:option('-save', os.getenv('HOME')..'/workspace/E/medipatch', 'subdirectory to save/log experiments in')
    cmd:option('-t', false, 'test only (test set)')
    cmd:option('-v', false, 'test only (valid set)')
	cmd:option('-network', '', 'reload pretrained network')
	cmd:option('-networkLoadOpt', true, 'Load also the inner state of gradient optimizer when network is given (good for further training)')
	cmd:option('-networkJustAsInit', false, 'Will take just the parameters (and epoch num if networkLoadOpt) from the network')
	cmd:option('-modelName', 'baseline', '')  --!!!!!!
	--
	cmd:option('-device', 1, 'CUDA device')
	cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
    cmd:option('-backend', 'cunn', 'Options: cudnn | ccn2 | cunn')
    cmd:option('-nDonkeys', 2, 'number of donkeys to initialize (data loading threads)')	
	cmd:option('-threads', 1, 'nb of threads to use (blas on cpu)')
	--
	cmd:option('-criterion', 'bsvm', 'criterion: nll | svm | mse')
	cmd:option('-optimization', 'SGD', 'optimization method: SGD | SGDCaffe | ASGD | CG | LBFGS')
	cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
	cmd:option('-lrdPolicy', 'inv', 'SGD: policy for learning rate decay. fixed | inv | step | expep')
	cmd:option('-lrdGamma', 5e-7, 'SGD: factor of decaying; meaning depends on the policy applied')
	cmd:option('-lrdStep', 0, 'SGD: stepsize of decaying; meaning depends on the policy applied')
	cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
	cmd:option('-weightDecay', 5e-4, 'weight decay (SGD only)')
	cmd:option('-momentum', 0.9, 'momentum: state.dfdx:mul(mom):add(1-damp, dfdx) (SGD only)')
	cmd:option('-momdampening', -1, 'dampening when momentum used, (default = "momentum", i.e. lin combination) (SGD only)')
	cmd:option('-asgdE0', 1, 'start averaging at epoch e0 (ASGD only)')
	cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
	cmd:option('-numEpochs', 55, 'number of epochs to train (<0=test only)')
	cmd:option('-numIters', 0, 'number of iters to train (caffe-parameter) (0=not used, >0 overrides numEpochs)')
	cmd:option('-batchEvalSize', 0, 'number of samples evaluated at once (0=use batchSize, 1=per-sample fw/bw, else subsets)')
	--
	cmd:option('-dataset', 'imagenet', 'dataset imagenet | pascal3d')
	cmd:option('-trainSplit', 90, '% of train dataset to use for training (rest for validation)')
	cmd:option('-doValidation', true, 'whether validate')
    cmd:option('-nTestSamples', -1, 'size of test set (-1 = all)')
    cmd:option('-imagenetCaffeInput', false, 'input data scaling. true (caffe): img[0,255] - mean. false (torch): (img[0,1] - mean) / std')
    --cmd:option('-batchSampling', 'seq', 'sampling of batches seq | rand | randperm')    
    cmd:option('-numTSPatches', 32, 'num patches to sample from a single traing sample')
    cmd:option('-numTestSPatches', 32, 'num patches to sample from a single test sample')
    cmd:option('-patchSize', 64, 'w or h of a patch')
    cmd:option('-patchDim', 2, '3D or 2D patches')
    cmd:option('-patchSampleRotMaxPercA', 0, 'whether should rotate patches by random angle (0 = disabled ..up to 1 == full angle [-pi,pi])')
    cmd:option('-patchSampleMaxScaleF', 1, 'max factor for randomly up/down-scaling patches. 1 == disabled.')
    cmd:option('-patchSamplePosSameTransf', false, 'whether positive patch should have the same transformation in both modalities.')
    cmd:option('-patchSampleNegDist', 'inter', 'distance type to distinguish neg samples inter | center')
    cmd:option('-patchSampleNegThres', 0.3, 'threshold for patchSampleNegDist, distance type specific, always relative to patch dimensions (see donkeyModapairs)')
    cmd:option('-patchSampleMaxBlacks', 0.5, '% of black pixels/voxels to skip a patch (pair)')
    cmd:option('-patchSampleBlackPairs', true, 'whether both samples from a pair or any of them being black leads to being skipped')
	--
    cmd:option('-parSharing', true, 'use weight sharing?')	
	cmd:option('-winit', 'default', 'weight initialization   default | He')
	cmd:option('-caffeBiases', false, 'initializes biases with learning rate 2x that of weights and turns off weight decay for them')
	--
    cmd:option('-baselineCArch', 'c_96_7_0_0_3,p_2,c_192_5,p_2,c_256_3,join,c_512_1,fin', 'configuration of baseline model')
    cmd:option('-modelParams', '', 'model-specific list of parameters key1=value,key2=value')
    
	cmd:text()
	local opt = cmd:parse(arg)
	
	if opt.t or opt.v then opt.runName=''; opt.numEpochs=-1; opt.numIters=-1 end
	opt.runName = os.date("%Y%m%d-%H%M%S") .. '-' .. opt.runName
	
	opt.modelParams = assert(loadstring('return {'..opt.modelParams..'}')())
	
    return opt, cmd
end

nlprint = print --save non-logging print before cmd:log

return M
