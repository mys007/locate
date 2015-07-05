function nii2mat(dirpath) 
    addpath('NIfTImatlab/matlab');
    dirpath = [dirpath '/'];
    files = dir([dirpath '*.nii.gz']);
    
    for file = files'
        % Create the nifti file object
        filenames = gunzip([dirpath file.name]);
        nfdin = niftifile(filenames{1});

        % Open the nifti file for reading
        nfdin = fopen(nfdin,'read');
        % nfdin now contains the header information

        % initialize the data array
        % Matlab is column major so order the data y,x
        data = zeros(nfdin.ny, nfdin.nx, nfdin.nz, nfdin.nt);

        % Read the data one volume at a time
        % This could be done in lots of ways 
        for i = 1:nfdin.nt
          [nfdin, databuff] = fread(nfdin, nfdin.nx*nfdin.ny*nfdin.nz);
          % Note that matlab assumes column major format
          % NIFTI is stored so x increases fastest. We need permute.
          data(:,:,:,i) = permute(reshape(databuff, [nfdin.nx nfdin.ny nfdin.nz]), [2 1 3]);
        end

        % This closes the file, but lets us keep using the header
        nfdin = fclose(nfdin);

        % Draw a slice
        %figure();
        %imagesc(data(:,:,1,1))
        %axis image
        %axis ij
        % The upper left hand corner is pixel (1,1)
        % y goes top to bottom
        % x goes left to right
        
        data = single(data); %torch/cuda works with floats anyway -- would 'short' be sufficient?
        save([dirpath file.name(1:end-7) '.mat'], 'data')
    end

end

