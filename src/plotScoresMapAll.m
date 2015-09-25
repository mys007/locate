function plotScoresMapAll(dirName)
    patchdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset';
    shapeInserter = vision.ShapeInserter('Shape','Circles','BorderColor','Custom','CustomBorderColor',uint8([0 0 1]));

    files = rdir(fullfile(dirName,'/**/*-scoremap-*.mat'));
    files = {files.name}';

    data = cell(numel(files),1);
    for i=1:numel(files)
        scorefile = files{i};
        [~,name] = fileparts(scorefile);                                          
        setname = strsplit(name,'-'); setname = setname{2};
        pxpos = strsplit(name, {'-',',','.'}); pxpos = [str2num(pxpos{end-1}) str2num(pxpos{end})];

        im2 = double(imread([patchdir '/' setname '/maps/panorama_crop_12.png']))/255;
        patch = double(imread(strrep(scorefile, '.mat', '.png')))/255;
        if size(patch,1) > 64; patch = patch(1:64,1:64,:); end
        scores = load(scorefile); scores = scores.x';

        im2 = imresize(im2, [size(scores,1), size(scores,2)]); 
        nscores = scores / max(max(scores));
        nscores(nscores<0) = 0;
        
        %penalize pixels having high positive score (=too many matches)  [0.6 seems to be a good thres]
        uniqueness = 1 - sum(nscores(nscores>0).^2) / sum(scores(:)>-1e20); 
        
        im2(:,:,1) =  ( nscores ).^2; %power for visu
        
        im2(1:size(patch,1),1:size(patch,2),:) = patch;
        
        im2 = insertText(im2, [70,5], ['max: ' num2str(max(max(scores))) ' min: ' num2str(max(0,min(min(scores)))) ' u:' num2str(uniqueness)],'FontSize',10, 'TextColor','black');
        [r,c] = find(nscores==max(max(nscores)),1);
        im2 = step(shapeInserter, im2, int32([c r 5]));
        im2 = step(shapeInserter, im2, int32([pxpos(1)+size(patch,1)/2, pxpos(2)+size(patch,1)/2, 2]));
        
        imwrite(im2, strrep(scorefile, '.mat', '.png')) %overwrite patch img
    end 