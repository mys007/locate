function status = cbiNiftiMerge( inImages, outImage)
%  status = cbiNiftiMerge( inImages, outImage)
%
% Combine images in inImages (an array) and write them out to outImage

nt = 0;

for inIndex = 1:numel(inImages)
    nfdIn{inIndex} = niftifile(inImages{inIndex});
    nfdIn{inIndex} = fopen(nfdIn{inIndex}, 'read');
    nt = nt + nfdIn{inIndex}.nt;
end

outhd = niftifile(outImage, nfdIn{1});
outhd.nt = nt;
outhd = fopen(outhd, 'write');

for inIndex = 1:numel(inImages)
    [nfdIn{inIndex}, data, size ] = fread(nfdIn{inIndex});
    outhd = fwrite(outhd, data, size);
    nfdIn{inIndex} = fclose(nfdIn{inIndex});
end
outhd = fclose(outhd);


