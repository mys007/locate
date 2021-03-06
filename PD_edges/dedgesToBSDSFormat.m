function dedgesToBSDSFormat()
    
    addpath /home/simonovm/workspace/locate/src/preproc/pfm2torch
    
    srcdir = '/media/simonovm/Slow/datasets/Locate/gt_dataset_905';
    destdir = '/media/simonovm/Slow/datasets/Locate/gt_dataset_905_BSDS_labels';
    %srcdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset';
    %destdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset_BSDS';    
    
    files = rdir(fullfile(srcdir,'/**/*pfm_edges.png'));
    files = {files.name}';

    for i=1:numel(files)
        [path1,name] = fileparts(files{i});
        [path,name2] = fileparts(path1);
        [path,name2] = fileparts(path);        %--disable for small dataset
        [path,longname] = fileparts(path);
        
        destname = [destdir '/data/groundTruth/train/' longname '.mat'];        
        copyfile([path1 '/photo_crop.png'], [destdir '/data/images/train/' longname '.png'])
        %if exist(destname, 'file') continue; end
        
        emap = imread(files{i});
        
        dmap=single(flipud(parsePfm([path1 '/distance_crop.pfm'])));
        dmap(dmap<=0) = 1e-20;
        dmap = log(dmap);

        groundTruth = {};
        j = 1;
        for th=0:50:250
            %groundTruth{j}.Segmentation = dmap;
            groundTruth{j}.Segmentation = inferLabels(emap, th);            
            groundTruth{j}.Boundaries = (emap > th);
            j = j +1;
        end

        copyfile([path1 '/photo_crop.png'], [destdir '/data/images/train/' longname '.png'])
        save(destname, 'groundTruth')
    end 
end    
    


function I = inferLabels(I, th)
    Ith = im2bw(I,th/255);

    for j=1:size(I,2)
        la=1;
        lifetime = 0;
        for i=1:size(I,1)    
            if Ith(i,j)==1 && Ith(i-1,j)~=1
                la = la+1;
                I(i,j)=0;
                lifetime = 32;
            else
                I(i,j)=la;
                lifetime = lifetime - 1;
                if lifetime==0 && la>1
                    la = 1;
                end
            end
        end
    end
    %figure(2);imshow(I);
    I = uint8(I);
end