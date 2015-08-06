# Dependencies
* Torch et al (Especially cudnn v3. TODO: explicitely state rocks?)
* ITK4.7 For preprocessing.

Two steps are necessary in addition:
- install myrock: `cd src/myrock; luarocks install myrock-1.0-0.rockspec`
- compile itkslave: `cd src/fb-imagent/itkslave; mkdir build; cd build; cmake ..; make`. This needs to be done after any update to itkslave:).


# Preprocessing the IXI dataset #
* Extract archive of each modality (currently, T1 and T2 are sufficient) into the same directory `RAWDIR`
* Now we need to align the volumes of each modality (T1 is assumed to be the modality providing the origin and scaling) and convert them to the torch format. In addition, T1 volumes without a corresponding T2 volume will be ignored. For this, `cd SOMEDIR/medipatch/src/preproc/nii2torch; bash nii2torch.sh RAWDIR ~/datasets/IXI/volumes`. If you don't like the destination directory, you need to change the paths in `donkeyModapairs.lua`.
* The data loading algorithm in the Torch code (`dataset.lua`) requires a special data organization structure, so we will need some hacks:). Run
```
cd ~/datasets/IXI
mkdir -p test/pos
mkdir -p train/pos
cd test; ln -s pos neg; cd ..
cd train; ln -s pos neg; cd ..
cd volumes
find . -name "*T1.t7img.gz" -type f -exec ln -s ../../volumes/'{}' ../train/pos/'{}' ';'
```
Now move a couple of training volumes to the test set. I chose 20 files by random:
```
R_IXI014-HH-1236-T1.t7img.gz    R_IXI388-IOP-0973-T1.t7img.gz
R_IXI027-Guys-0710-T1.t7img.gz  R_IXI427-IOP-1012-T1.t7img.gz
R_IXI049-HH-1358-T1.t7img.gz    R_IXI464-IOP-1029-T1.t7img.gz
R_IXI094-HH-1355-T1.t7img.gz    R_IXI492-Guys-1022-T1.t7img.gz
R_IXI136-HH-1452-T1.t7img.gz    R_IXI541-IOP-1146-T1.t7img.gz
R_IXI181-Guys-0790-T1.t7img.gz  R_IXI556-HH-2452-T1.t7img.gz
R_IXI222-Guys-0819-T1.t7img.gz  R_IXI582-Guys-1127-T1.t7img.gz
R_IXI256-HH-1723-T1.t7img.gz    R_IXI627-Guys-1103-T1.t7img.gz
R_IXI298-Guys-0861-T1.t7img.gz  R_IXI635-HH-2691-T1.t7img.gz
R_IXI344-Guys-0905-T1.t7img.gz  R_IXI648-Guys-1107-T1.t7img.gz
```
You need to move them (actually, the symlinks) from `train/pos` to `test/pos`.

# Example training commands #
Before execution, `medipatch/src/fb-imagenet` needs to be the current directory. 

* Siamese net from scratch (no rotation or scale): `CUDA_VISIBLE_DEVICES=0 qlua main.lua -backend cunn -modelName siam2d -runName base-lr1e4 -nDonkeys 3 -caffeBiases true -dataset modapairs -baselineCArch c_96_7_0_0_3_1_1,p_2,c_192_5_0_0_1_1_1,p_2,c_256_3_0_0_1_1_1,join,c_512_1_0_0_1_1_1,fin_1_1 -learningRate 1e-4 -numEpochs 300 -device 1`
* Siamese net from scratch (rotation and scale): `CUDA_VISIBLE_DEVICES=0 qlua main.lua -backend cunn -modelName siam2d -runName base-lr1e2-rotsc -nDonkeys 3 -caffeBiases true -dataset modapairs -baselineCArch c_96_7_0_0_3_1_1,p_2,c_192_5_0_0_1_1_1,p_2,c_256_3_0_0_1_1_1,join,c_512_1_0_0_1_1_1,fin_1_1 -learningRate 1e-2 -numEpochs 300 -patchSampleRotMaxPercA 1 -patchSampleMaxScaleF 1.1 -device 1 `
* Siamese net finetuning (no rotation or scale): `CUDA_VISIBLE_DEVICES=0 qlua main.lua -backend cunn -modelName siam2d -runName ft-notredame-b0.1-t1-lr1e4 -nDonkeys 3 -caffeBiases true -dataset modapairs -network ~/workspace/medipatch/szagoruyko/siam_notredame_nn.t7 -networkLoadOpt false -networkJustAsInit true -baselineCArch c_96_7_0_0_3_0.1_0.1,p_2,c_192_5_0_0_1_0.1_0.1,p_2,c_256_3_0_0_1_0.1_0.1,join,c_512_1_0_0_1_1_1,fin_1_1 -learningRate 1e-4 -numEpochs 300 -device 1`
* 2ch-net finetuning (no rotation or scale): `CUDA_VISIBLE_DEVICES=0 qlua main.lua -backend cunn -modelName 2ch2d -runName ft-notredame-lr1e2 -nDonkeys 3 -caffeBiases true -dataset modapairs -network /home/simonovm/workspace/medipatch/szagoruyko/2ch_notredame_nn.t7 -networkLoadOpt false -networkJustAsInit true -baselineCArch c_96_7_0_0_3,p_2,c_192_5,p_2,c_256_3,fin -learningRate 1e-2 -numEpochs 300 -device 1`

Results will appear in `~/workspace/E/medipatch` by default, this can be changed by the `-save` command-line parameter. You can notice that the actual structure of the network is defined by the `-baselineCArch` parameter. It's a bit cryptic but expressive and efficient, parsing is done in `modeldefs.lua`.

# Trained networks #
2D
* [Siamese net from scratch (no rotation or scale)](http://imagine.enpc.fr/~simonovm/medipatch/nets/main-siam2d/20150712-205257-base-lr1e4/network.net)
* [Siamese net from scratch 32x32 patches (no rotation or scale)](http://imagine.enpc.fr/~simonovm/medipatch/nets/main-siam2d/20150712-205529-base-patch32-lr1e4/network.net)
* [Siamese net from scratch (rotation and scale)](http://imagine.enpc.fr/~simonovm/medipatch/nets/main-siam2d/20150805-124204-base-lr1e2-rotsc-fixfullrot-ex/network.net) --updated 6.8.
* [Siamese net finetuning (no rotation or scale)](http://imagine.enpc.fr/~simonovm/medipatch/nets/main-siam2d/20150712-205257-base-lr1e4/network.net)

3D
* [Siamese net 16x16x16 (no rotation or scale)](http://imagine.enpc.fr/~simonovm/medipatch/nets/main-siam3d/20150804-022816-base-ps16-5_3p_3-lr1e2-cudnn/network.net)