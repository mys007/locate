function status = cbiNiftiVols( inImage, outImage, startVol, numVols )
%  status = cbiNiftiVols( inImage, outImage, startVol, numVols )
%
% Extract numVols from inImage and store in outImage starting at startVol

status = 0;

inhd = niftifile(inImage);
inhd = fopen(inhd, 'read');

endVol = startVol + numVols - 1;

% Make sure we don't try to read too much
if ( endVol > inhd.nt )
    endVol = inhd.nt;
end
[inhd, data, size ] = fread(inhd, [], [], [], [startVol endVol]);
inhd = fclose(inhd);
outhd = niftifile(outImage, inhd);
outhd.nt = (endVol - startVol + 1);
outhd = fopen(outhd, 'write');
outhd = fwrite(outhd, data, size);
outhd = fclose(outhd);

status = 1;