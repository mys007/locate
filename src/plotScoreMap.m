scorefile = '/home/simonovm/workspace/E/locate/main-2ch2d/20150914-012907-allshadesG-p64-th07-lr1e2-ex/plots_ep1000/matches_zumsteinspitze_scoremap_722,22.mat';
setname = 'zumsteinspitze';

patchdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset';
im2 = double(imread([patchdir '/' setname '/maps/panorama_crop_12.png']))/255;
patch = imread(strrep(scorefile, '.mat', '.png'));
scores = load(scorefile); scores = scores.x';

im2 = imresize(im2, [size(scores,1), size(scores,2)]); 
nscores = scores / max(max(scores));
nscores(nscores<0) = 0;
im2(:,:,1) =  ( nscores ).^2; %power for visu

%if ~usejava('Desktop')
%    imwrite(im2,filename)
figure(1)
imshow(patch)
figure(2)
imshow(im2)