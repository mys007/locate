function plotScoresMapAll(dirName, ps)
    patchdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset';
    shapeInserter = vision.ShapeInserter('Shape','Circles','BorderColor','Custom','CustomBorderColor',uint8([0 0 1]));
    if nargin==1; ps = 64; end

    files = rdir(fullfile(dirName,'/**/*-scoremap-*.mat'));
    files = {files.name}';
    setstats = struct('path','','name','');
    
    for i=1:numel(files)
        scorefile = files{i};
        [path,name] = fileparts(scorefile);                                          
        setname = strsplit(name,'-'); setname = setname{2};
        pxpos = strsplit(name, {'-',',','.'}); pxpos = [str2num(pxpos{end-1}) str2num(pxpos{end})];

        im2 = double(imread([patchdir '/' setname '/maps/panorama_crop_12.png']))/255;
        patch = double(imread(strrep(scorefile, '.mat', '.png')))/255;
        if size(patch,1) > ps; patch = patch(1:ps,1:ps,:); end
        %scores = load(scorefile); scores = scores.x';
        scores = load(scorefile); scores = min(scores.x,[],3); scores = scores'; %for imagenet

        im2 = imresize(im2, [size(scores,1), size(scores,2)]); 
        nscores = scores;
        nscores(nscores<0) = 0;
        nscores = nscores / (max(max(nscores))+1e-20);
        
        %penalize pixels having high positive score (=too many matches)  [0.6 seems to be a good thres]
        uniqueness = 1 - sum(nscores(nscores>0).^2) / sum(scores(:)>-1e20); 
        
        im2(:,:,1) =  ( nscores ).^2; %power for visu
        
        im2(1:size(patch,1),1:size(patch,2),:) = patch;
        
        im2 = insertText(im2, [ps+5,5], ['max: ' num2str(max(max(scores))) ' min: ' num2str(max(0,min(min(scores)))) ' u:' num2str(uniqueness)],'FontSize',10, 'TextColor','black');
        [r,c] = find(nscores==max(max(nscores)),1);
        if numel(r)==0; r=1e10; c=1e10; end
        im2 = step(shapeInserter, im2, int32([c r 5]));
        gt_r = pxpos(2)+size(patch,1)/2; gt_c = pxpos(1)+size(patch,1)/2;
        im2 = step(shapeInserter, im2, int32([gt_c gt_r 2]));
        
        imwrite(im2, strrep(scorefile, '.mat', '.png')) %overwrite patch img
        
        
        % Compute and store point statistics (gt ~~ output)
        if ~strcmp(path, setstats.path) | ~strcmp(setname, setstats.name) | i==numel(files) %=end of a set
            if length(setstats.name)>0
                fileID = fopen([setstats.path '/matchresult_' setstats.name '.txt'],'w');
                fprintf(fileID,'Mean uniqueness | %%<10px | %%<25px | %%<50px | #pts\n');
                fprintf(fileID,'%6.2f\t%.2f\t%.2f\t%.2f\t%d\n', mean(setstats.uniq), mean(setstats.d10)*100, mean(setstats.d25)*100, mean(setstats.d50)*100, numel(setstats.uniq));
                fclose(fileID);
            end
            setstats.uniq = []; setstats.d10 = []; setstats.d25 = []; setstats.d50 = [];  
        end

        setstats.name = setname;
        setstats.path = path;
        setstats.uniq = [setstats.uniq uniqueness];
        dist = sqrt((gt_c-c)^2 + (gt_r-r)^2);
        setstats.d10 = [setstats.d10 dist<=10];
        setstats.d25 = [setstats.d25 dist<=25];
        setstats.d50 = [setstats.d50 dist<=50];        
    end 