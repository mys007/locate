function [numpix] = checkranges(nfd, args)
% [numpix] = checkranges(nfd, args)
%
% Verify that the ranges specified in 'args' are consistent 
% with the limits in the header 'nfd'. Return the total number o pixels.
  
  numpix = uint64(1);
  if length(args) > nfd.niftiheader.ndim
    for narg = nfd.niftiheader.ndim + 1 : length(args)
      if args{narg} > 1
        error('Image has only %d dimensions', nfd.niftiheader.ndim);
      end
    end
  end
  dims = nfd.niftiheader.dim;
  for ind = 1:min(nfd.niftiheader.ndim, length(args))
    entry = args{ind};
    if length(entry) == 2
      if (entry(1) > entry(2)) | (entry(1) < 1) | (entry(2) > dims(ind+1))
        error('Illegal range ', mat2str(entry));
      end
      numpix = numpix * uint64(entry(2) - entry(1) + 1);
    elseif length(entry) == 0
      numpix = numpix * uint64(dims(ind+1));  
    else   
      if (length(entry) > 2) | (entry(1) < 1) | (entry(1) > dims(ind+1))
        error('Illegal range ', mat2str(entry));
      end
    end
  end
