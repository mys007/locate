require 'torch'
require 'paths'
require 'mattorch'
require 'torchzlib'
	
for f in paths.files(arg[1]) do
	f = arg[1] .. '/' .. f
	if paths.extname(f)=='mat' then 
		loaded = mattorch.load(f)
		--print(loaded.data:size())
		torch.save(string.sub(f,1,string.len(f)-3)..'t7img.gz', torch.CompressedTensor(loaded.data, 2))
   end
end
