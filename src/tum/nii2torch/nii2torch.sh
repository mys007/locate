#!/bin/bash
../slices/build/AlignVolumes $1/ $2/
matlab -nodisplay -nosplash -nodesktop -r "nii2mat('$2'); exit"
rm "$2"/*.nii
rm "$2"/*.nii.gz
qlua mat2torch.lua $2
rm "$2"/*.mat
qlua rem_orphans.lua $2
