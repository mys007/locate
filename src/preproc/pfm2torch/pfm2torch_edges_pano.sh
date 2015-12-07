#!/bin/bash

shopt -s globstar
for i in /media/simonovm/Slow/datasets/Locate/gt_dataset_905/**/*panorama_depth.pfm.gz; do # Whitespace-safe and recursive
	matlab -nodisplay -nosplash -nodesktop -r "f=gunzip('$i', '/home/simonovm/tmp'); data=single(flipud(parsePfm(f{1}))); data=significantEdges(data, 4); imwrite(data, ['$i' '_edges.png']); save('/home/simonovm/tmp/tmp.mat', 'data'); exit"
	qlua -e "require 'mattorch'; require 'torchzlib'; M = mattorch.load('/home/simonovm/tmp/tmp.mat'); f='$i'; torch.save(string.sub(f,1,string.len(f)-3)..'_edges.t7img.gz', torch.CompressedTensor(M.data, 2))"
    echo "$i"
done

rm '/home/simonovm/tmp/tmp.mat'
