package.path = "myrock/?.lua;" .. package.path
require 'torch'
require 'myutils'
require 'paths'
require 'sys'
local matio = require 'matio'
require 'strict'


local p3ddir = '/home/simonovm/datasets/PASCAL3D_release1.1/'
matio.use_lua_strings = true

local function processList(listpath, classname, subset, outdir)
    --scan for all names in the filelist and create softlinks in the form ourdir/class/name.jpg|mat
    local f = assert(io.open(listpath, 'r'))
    local fstr = f:read('*all')
    f:close()     
    
    outdir = outdir..'/'..classname
    os.execute('mkdir -p "' .. outdir .. '"')

    for line in fstr:gmatch("[^\r\n]+") do
        local fields = string.split(line, '%s')
        if not fields[2] or fields[2]~='-1' then
            local imgpath = p3ddir..'Images/'..classname..'_'..subset..'/'..fields[1]
            if paths.filep(imgpath..'.jpg') then imgpath = imgpath..'.jpg' else imgpath = imgpath..'.JPEG' end
            local annopath = p3ddir..'Annotations/'..classname..'_'..subset..'/'..fields[1]..'.mat'
            assert(paths.filep(annopath), annopath..' not found')
            assert(paths.filep(imgpath), imgpath..' not found')
            
            -- if the image contains more bboxes of interest, create additional symbolic links (for the donkeys, bbox is the unit, not image) 
            require 'image'
            local  time = sys.clock()
            local annos = matio.load(annopath)
            
               time = sys.clock() - time
    
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    
    torch.save('/home/simonovm/tst', annos)
    time = sys.clock()
    annos = torch.load('/home/simonovm/tst')
    time = sys.clock() - time
    print("<XXXXXX" .. (time*1000) .. 'ms')
    
       time = sys.clock()
    local input = image.load(imgpath)
    time = sys.clock() - time
    print("<IIIII" .. (time*1000) .. 'ms')      --- -> todo: save just interesting fields as torch structure. loading the mat files again and again costs time
            
            if annos['record']['objects'][1] ~= nil then
                local nObjs = 0
                for i,obj in pairs(annos['record']['objects'][1]) do
                    if obj['viewpoint'] then nObjs = nObjs + 1 end
                end    
                for i=2,nObjs do
                    os.execute('ln -s "' .. imgpath .. '" "' .. outdir .. '/' .. fields[1] .. '-A' .. i .. '.' .. paths.extname(imgpath) .. '"')
                end
            end
            
            os.execute('ln -s "' .. imgpath .. '" "' .. annopath .. '" "' .. outdir .. '"')
            print(imgpath, annopath)
        end
    end  
end

local function processDir(dirname, fsuffix, subset, outdir)
    --process all files ending with fsuffix in dirname
    local classes = {'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'sofa', 'tvmonitor'} 
    for i, classname in pairs(classes) do
        processList(paths.concat(dirname, classname..fsuffix), classname, subset, outdir)
    end
end


processDir(p3ddir..'PASCAL/VOCdevkit/VOC2012/ImageSets/Main', '_train.txt', 'pascal', p3ddir..'Datasets/Pascal_train')
processDir(p3ddir..'PASCAL/VOCdevkit/VOC2012/ImageSets/Main', '_val.txt', 'pascal', p3ddir..'Datasets/Pascal_val')
processDir(p3ddir..'Image_sets', '_imagenet_val.txt', 'imagenet', p3ddir..'Datasets/Imagenet_val')
processDir(p3ddir..'Image_sets', '_imagenet_train.txt', 'imagenet', p3ddir..'Datasets/Imagenet_train')