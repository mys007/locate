#!/bin/bash

shopt -s globstar
for i in /media/simonovm/Slow/datasets/Locate/match_descriptors_dataset/**/*.pfm; do # Whitespace-safe and recursive
	matlab -nodisplay -nosplash -nodesktop -r "data=single(parsePfm('$i')); save('/home/simonovm/tmp/tmp.mat', 'data'); exit"
	qlua -e "require 'mattorch'; require 'torchzlib'; M = mattorch.load('/home/simonovm/tmp/tmp.mat'); f='$i'; torch.save(string.sub(f,1,string.len(f)-3)..'t7img.gz', torch.CompressedTensor(M.data, 2))"
    echo "$i"
done

rm '/home/simonovm/tmp/tmp.mat'
