% Demo for Structured Edge Detector (please see readme.txt first).

%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
%opts.modelFnm='modelBsdsLocate_uneqDiff-l3-';        % model name
%opts.modelFnm='modelBsdsLocate_uneqDiffCannyFix-l3-';        % model name
opts.modelFnm='modelBsdsLocate_uneqDiffCannyFixTreeNoUint32-l3-fullmap';        % model name
%opts.modelFnm='modelBsdsLocate_uneqDiffCannyFixTreeNoUint32-l3Only-fullmap';        % model name
%opts.modelFnm='modelBsdsLocate_uneqDiffCannyFixTreeNoUint32-l4-fullmap';        % model name
%opts.modelFnm='modelBsdsLocate_uneqDiffCannyFixTreeNoUint32-l2-fullmap';        % model name
%opts.modelFnm='modelBsdsLocate_uneqDiffCannyFix-l3-noInfer';        % model name
%opts.modelFnm='modelBsdsLocate_uneqDiffCannyFixTreeNoUint32-l3-';        % model name
%opts.modelFnm='modelBsdsLocate_uneqDiffCannyFix-l3Only-';        % model name
%opts.modelFnm='modelBsdsLocate_labels-l3-';        % model name
%opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training    [doesn't have much influence]
opts.useParfor=0;                 % parallelize if sufficient memory
%opts.bsdsDir = '/media/simonovm/Slow/BSR/BSDS500/data/';
%opts.bsdsDir = '/media/simonovm/Slow/datasets/Locate/gt_dataset_905_BSDS_labels/data/';
opts.bsdsDir = '/media/simonovm/Slow/datasets/Locate/gt_dataset_905_BSDS/data/';

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=1;          % for top accuracy set multiscale=1
model.opts.sharpen=0;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results
%I = imread('peppers.png');
%I = imread('/media/simonovm/Slow/datasets/Locate/gt_dataset_905_BSDS_labels/data/images/train/sp_dom4.png');
I = imread('/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset/finsteraarhorn05/maps/photo_crop.png');
tic, E=edgesDetect(I,model); toc
figure(1); im(I); figure(5); im(1-E); title(opts.modelFnm)
