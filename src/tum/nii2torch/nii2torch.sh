#!/bin/bash
matlab -nodisplay -nosplash -nodesktop -r "nii2mat('$1'); exit"
rm "$1"/*.nii
qlua mat2torch.lua $1
rm "$1"/*.mat
