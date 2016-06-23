require 'torch'
require 'image'
require 'optim'
require 'hdf5'
require 'nn'

-- Command line arguments
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
-- Model settings
cmd:option('-model', './outputs/model_210.bin', 'Binary file of the model saved by train.lua')
cmd:option('-n_layers', 4, 'Number of encoder layers.')
cmd:option('-n_filters', 64, 'Number of filters per layer.')
cmd:option('-embedding_size', 1024, 'Size of the embedding (or bottleneck layer)')
cmd:option('-patch_width', 40, 'Size of inner-most glimpse patch')
cmd:option('-n_patch', 4, 'Number of patches per glimpse')
cmd:option('-glimpse_blur', 2, 'Blur scale for each glimpse further from center. Uses: 1 / (glimpse# ^ glimpse_blur)')
-- CPU/GPU
cmd:option('-gpuid', -1, 'GPU to execute on')
cmd:option('-threads', 1, 'threads')
cmd:option('-seed', 123456, 'random seed')
cmd:text()

opt = cmd:parse(arg)

-- Configure GPU
if opt.gpuid >= 0 then
    print('Loading Torch GPU libraries...')
    local ok, _ = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('Using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        gen = torch.Generator()
        torch.setnumthreads(opt.threads)
        torch.manualSeed(opt.seed)
        opt.gpuid = -1 -- overwrite user setting
    end
end

--[[
extractGlimpse(x, phi, patch_width, n_patches)
	Takes an input image/tensor and returns a glimpse which consists
	of progressively subsampled squares/cubes (called patches)
	around the glimpse center.

Args:
	x (torch.Tensor): Input image or tensor of dimension d
	phi (Table): Length d list containing the glimpse center
	patch_width (float): Width in pixels, of each patch
	n_patches (float): Number of patches to include in the glimpse
	scaling (float): Specifies the degree of "blur" or interpolation. Larger
					 values indicates more downsampling at each patch.

(Returns:
	glimpse (torch.Tensor): Glimpse which has same dimensions as input x
]]
function extractGlimpse(x, phi, patch_width, n_patches, scale_factor)
    -- Get the full resolution hypercube centered at phi
    local glimpse_width = n_patches * patch_width
    local hypercube = extractHypercube(x, phi, glimpse_width)

    -- Downsample the input
    local scales = {}
    for i = 1, n_patches do scales[i] = 1 / math.pow(i, scale_factor) end
    -- Create progressively downsampled patches
    local pyramid = image.gaussianpyramid(hypercube, scales)
    -- Scale all back to side length = glimpse_width
    for i = 1, #pyramid do pyramid[i] = image.scale(pyramid[i], glimpse_width) end

    -- Combine all patches into single tensor
    glimpse = torch.zeros(pyramid[1]:size())
    for i = 1, #pyramid do
        local corner1 = (i - 1) * (patch_width / 2) + 1
        local corner2 = glimpse_width - corner1 + 1
        local slice = {}
        for j = 1, x:dim() do table.insert(slice, { corner1, corner2 }) end
        local j = #pyramid - (i - 1)
        glimpse[slice] = pyramid[j][slice]
    end
    --os.execute("sleep " .. tonumber(5))
    --os.exit()
    return glimpse
end

--[[
extractHypercube(x, phi, patch_width, n_patches)
	Extracts a full resolution hypercube centerd at phi. Used as a helper
	function for extractGlimpse(...).

Args:
	x (torch.Tensor): Input image or tensor of dimension d
	phi (Table): Length d list containing the glimpse center
	glimpse_width (float): Length of one side of the hypercube

Returns:
	hypercube (torch.Tensor): Hypercube centered at phi with length
							  glimpse_width in each dimension.
]]
function extractHypercube(x, phi, glimpse_width)
    local gw2 = math.floor(glimpse_width / 2)

    -- Find the start and end index of the hypercube
    local slice = {}
    for d = 1, x:dim() do
        table.insert(slice, { phi[d] - gw2 + 1, phi[d] + gw2 })
    end
    local hypercube = x[slice]
    return hypercube
end

--[[
matrixToimage(mat)
	Converts a matrix (2D torch.Tensor) which may contain floating point values
	in an arbitrary range into an image format in range [0, 255], or [0, 1]

Args:
	mat (torch.Tensor): Two-dimensional torch tensor
	max_val (float): 255 or 1. Determines the max scale.

Returns:
	img (torch.Tensor): Same as incoming mat but scaled to [0, 255] or [0, 1]
]]
function matrixToImage(mat, max_val)
    mat = mat - torch.min(mat)
    mat = mat / torch.max(mat)
    img = mat * max_val
    return img
end

--[[
randomGlimpseCenter(x_size, patch_width, n_patch)
	Computes a random glimpse center such that the entire glimpse fits within
	the input space (i.e. no padding required).

Args:
	x_size (torch.Tensor): Contains the dimensions of the input
	glimpse_width (float): The width of the entire glimpse

Returns:
	phi (Table): Point denoting the glimpse center. Same dimensions as x_size.
]]
function randomGlimpseCenter(x_size, glimpse_width)
    local gw2 = glimpse_width / 2
    local phi = {}
    for dim = 1, #x_size do
        -- Find min and max values
        local min_val = gw2
        local max_val = x_size[dim] - gw2
        table.insert(phi, torch.random(min_val, max_val))
    end
    return phi
end

--[[
tableToTensor(table)
	Takes a lua table and converts it to a torch.Tensor

Args:
	table (table): Lua table, can be any number of dimensions

Returns:
	tensor (torch.Tensor): Original lua table converted to a torch.Tensor
]]
function tableToTensor(table)
    local tensorSize = table[1]:size()
    local tensorSizeTable = { -1 }
    for i = 1, tensorSize:size(1) do
        tensorSizeTable[i + 1] = tensorSize[i]
    end
    merge = nn.Sequential():add(nn.JoinTable(1)):add(nn.View(unpack(tensorSizeTable)))
    tensor = merge:forward(table)
    return tensor
end

-- Load the dataset
print('Loading the dataset...')
local input_file = hdf5.open('../../datasets/DPI-T_train_depth_map.h5', 'r')
local data = input_file:read('/'):all()
input_file:close()
local X_train_table = {}
for p_id, person in pairs(data) do
    for v_id, video in pairs(person) do
        for i = 1, video:size()[1] do
            nans = video[i]:ne(video[i]):max(2)
            if torch.sum(nans) == 0 then
                table.insert(X_train_table, video[i])
            end
        end
    end
end

X_train = tableToTensor(X_train_table)
-- Since depth maps are of size (H, W), reshape to (1, H, W) to enable conv
s = X_train:size()
n_train = s[1]
H = s[2]
W = s[3]
gw = opt.n_patch * opt.patch_width

print('Loading model from: ' .. opt.model)
module = torch.load(opt.model, 'ascii')
print(module)

local glimpse_width = opt.n_patch * opt.patch_width

for i = 1, n_train do
    local x = X_train[i]
    local phi = randomGlimpseCenter(x:size(), glimpse_width)
    glipmse = extractGlimpse(x, phi, opt.patch_width, opt.n_patch, opt.glimpse_blur)
    glimpse = glimpse:clone():resize(1, glimpse_width, glimpse_width)
    if opt.gpuid >= 0 then
        input_ = glimpse:clone():cuda()
    else
        input_ = glimpse:clone()
    end
    reconstruction = module:forward(input_)
    -- Get the embedding from the encoder
    embedding = module:get(1):get(17).output
end
