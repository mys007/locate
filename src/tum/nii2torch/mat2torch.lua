require 'torch'
require 'paths'
require 'mattorch'
	
for f in paths.files(arg[1]) do
	f = arg[1] .. '/' .. f
	if paths.extname(f)=='mat' then 
		loaded = mattorch.load(f)
		--print(loade.data:size())
		torch.save(string.sub(f,1,string.len(f)-3)..'t7img', loaded.data)
   end
end
