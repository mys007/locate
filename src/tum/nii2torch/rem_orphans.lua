require 'torch'
require 'paths'
require 'os'

function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

assert(paths.dirp(arg[1]))

files = {}
for f in paths.files(arg[1]) do 
	files[f] = f
end

for i, file in pairs(files) do 
	if string.ends(file, "T1.t7img") then
		if not files[file:gsub("T1.t7img", "T2.t7img")] then
			os.remove(paths.concat(arg[1],file))
			print('removed ' .. file)
		end
	end
end

