function [ output_args ] = locateTestPano(dir)


    addpath /home/simonovm/workspace/locate/src/preproc/pfm2torch
    
    pano = imread([dir '/synthetic_panorama_depth.pfm.gz_edges.png']);
    %panodt = - doDT(pano, 1) .^2;
    panodt = doDT(pano, 0.2);
    figure(4); imagesc(panodt)
      
    opts=edgesTrain();                % default options (good settings)
    opts.modelDir='models/';          % model will be in models/forest
    %opts.modelFnm='modelBsdsLocate_labels-l3-';        % model name
    opts.modelFnm='modelBsdsLocate_uneqDiff-l3-';        % model name
    %opts.modelFnm='modelBsds';        % model name
    opts.nPos=5e5*2; opts.nNeg=5e5*2;     % decrease to speedup training
    opts.useParfor=0;                 % parallelize if sufficient memory
    %opts.bsdsDir = '/media/simonovm/Slow/BSR/BSDS500/data/';
    opts.bsdsDir = '/media/simonovm/Slow/datasets/Locate/gt_dataset_905_BSDS_labels/data/';
    %opts.bsdsDir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset_BSDS/data/';
    model=edgesTrain(opts); 
    
    model.opts.multiscale=1;          % for top accuracy set multiscale=1
    model.opts.sharpen=2;             % for top speed set sharpen=0
    model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
    model.opts.nThreads=4;            % max number threads for evaluation
    model.opts.nms=1;                 % set to true to enable nms

    I = imread([dir '/photo_crop.png']);
%    I = imread([dir '/../pinhole/rescaled_orig_image.png']);
    tic, E=edgesDetect(I,model); toc
    figure(2); im(I); figure(2); im(E);

    E(E<0.2) = 0;
    E = E.*repmat(size(E,1)':-1:1,size(E,2),1)'; %"upper edges are more important to match"
    figure(5); im(E);   
    
    
    
    tic
   
    C = gather(conv2(gpuArray(panodt), gpuArray(rot90(E,2)),'valid'));
    %C = conv2( panodt, rot90(E,2), 'valid'); --runtime varies a lot based on non-0 elems
    
   
    
    norm = 2;
    if norm==1
        Edt = doDT(E*255, 0);%0.3);
        pano=single(pano)/255;
        figure(31); im(pano);
        figure(32); im(Edt);
        Creverse = gather(conv2(gpuArray(pano), gpuArray(rot90(Edt,2)),'valid'));
        figure(30); im(Creverse); colormap jet;
        C = C + Creverse*0.2; %motivation: do reverse check (penalize for edges in pano not explained by E)
    elseif norm==2
        pano=single(pano)/255;
        block = E; block(:)=1;
        Creverse = gather(conv2(gpuArray(pano), gpuArray(rot90(block,2)),'valid'));
        figure(30); im(Creverse); colormap jet;
        C = C ./ (Creverse + 1); %motivation: normalize by number of edge pixels (blocks with many edges are easier to match)
    end

    toc
    [max_cc, imax] = max(C(:));
    [ypeak, xpeak] = ind2sub(size(C),imax(1));
    figure(3); im(C); colormap jet;
    
    
    panoRgb = imread([dir '/synthetic_panorama.png']);
    panoRgb(ypeak:ypeak+size(E,1)-1, xpeak:xpeak+size(E,2)-1, :) = panoRgb(ypeak:ypeak+size(E,1)-1, xpeak:xpeak+size(E,2)-1, :)/3 + 2/3*I;
    figure(1); im(panoRgb);
        
%    name = 'bietschhorn'; export_fig([name '_result'], '-png','-native', 1); export_fig([name '_cost'], '-png','-native', 3); export_fig([name '_edges'], '-png','-native', 2); export_fig([name '_dt'], '-png','-native', 4)

end

% "layered" distance transform
function panodt = doDT(pano, fact)
    panodt = [];
    
    for th=250:-50:50
        dt = -bwdist(pano>th, 'euclidean');
        if numel(panodt)==0
            panodt = dt;
        else
            panodt = max(panodt, dt-(250-th)*fact);
        end
    end
end