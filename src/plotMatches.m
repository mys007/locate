matchfile = '/home/simonovm/workspace/E/locate/main-2ch2d/20150914-093248-allshadesG-p64-th-0.001-lr1e2/plots_ep345/matches_zumsteinspitze.mat';
setname = 'zumsteinspitze';
th = 0;%0.8;

patchdir = '/media/simonovm/Slow/datasets/Locate/match_descriptors_dataset';
im1 = imread([patchdir '/' setname '/maps/orig_image.jpg']);
im2 = imread([patchdir '/' setname '/maps/panorama_crop_12.png']);
matches = load(matchfile); matches = matches.x';

scalef = size(im1,2)/size(im2,2);
im2 = imresize(im2, scalef);                          

fprintf('Loaded matches: %d\n',size(matches,1));
if th>0
    pass = matches(:,5)>th;
    matches = matches(pass,:);
    fprintf('Filtered matches: %d\n',size(matches,1));
end

%h = showMatchedFeatures(imrotate(im1,-90), imrotate(im2,-90), [matches(:,2) matches(:,1)], [matches(:,4) matches(:,3)]*scalef, 'montage');
%az = 90;
%el = 90;
%view(az, el);                                                                 

showMatchedFeatures(im1, im2, matches(:,1:2), matches(:,3:4)*scalef, 'falsecolor');


% ideally, (not zumstein), the offset should be 0 if input aligned images
    % -> parameterized testpair (opt) and then anotehr lua script which checks
    % it. test it on validation data (find out whcih is it for a
    % dataettype) and that's what we optimize for.