------------------------------------------------------------------------
--[[ SpatialGlimpse ]] --
-- Ref A.: http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- a glimpse is the concatenation of down-scaled cropped images of
-- increasing scale around a given location in a given image.
-- input is a pair of Tensors: {image, location}
-- locations are x,y coordinates of the center of cropped patches.
-- Coordinates are between -1,-1 (top-left) and 1,1 (bottom right)
-- output is a batch of glimpses taken in image at location (x,y)
-- glimpse size is {height, width}, or width only if square-shaped
-- depth is number of patches to crop per glimpse (one patch per scale)
-- Each successive patch is scale x size of the previous patch
------------------------------------------------------------------------
local SpatialGlimpse, parent = torch.class("nn.SpatialGlimpse", "nn.Module")

--[[
-- 	SpatialGlimpse:__init(encoder, width, n_patch, blur)
-- 		Creates a spatial glimpse module. The spatial glimpse module takes as input
-- 		an image and a location, and will output an embedding representing the glimpse.
--
--	Args:
--		encoder: nn.module trained offline
--		width: Width of a single glimpse patch
--		n_patch: number of patches in the glimpse
--		blue: Blur scale (default=2 for quadratic)
--]]
function SpatialGlimpse:__init(encoder, width, n_patch, blur)
    -- Encoder model trained offline
    self.encoder = encoder
    -- Width of a single patch, namely, the center most
    self.width = width
    -- Number of consecutive patches
    self.n_patch = n_patch
    -- Blur scale. The nth patch is downsampled by blur^n
    self.blur = blur
    -- The final glimpse width. This is the size input into the encoder
    self.glimpse_width = width * n_patch
    parent.__init(self)
    self.gradInput = { torch.Tensor(), torch.Tensor() }
    self.modules = self.encoder
end

--[[
--	SpatialGlimpse:updateOutput(inputTable)
-- 		Performs the actual glimpse extraction and encoding process.
--
--	Args
--		inputTable: Contains the input (image) and the glimpse location
--			[1] input_imgs of size (batch_size, channels, width, height)
--			[2] location of size (batch_size, n_dim)
--
--	Returns
--		output: The result of a "forward pass" through this spatial glimpse module.
--			The output will be a matrix representing the encoded embeddings of the glimpses.
--]]
function SpatialGlimpse:updateOutput(inputTable)
    local input_imgs, location = unpack(inputTable)
    input_imgs, location = self:toBatch(input_imgs, 3), self:toBatch(location, 1)
    -- input: (batch_size, channels, width, height)
    -- location: (batch_size, 2)
    dim = location:dim()
    embedding_size = 1024 -- Depends on the pre-trained autoencoder

    local batch_size = input_imgs:size()[1]
    local input_glimpses = torch.Tensor(batch_size, 1, self.glimpse_width, self.glimpse_width)
    local scales = {}
    for i = 1, self.n_patch do scales[i] = 1 / math.pow(i, self.blur) end
    -- Check if input location, when converted to H, W is in image bounds
    location[{ {}, 1 }] = torch.clamp(torch.round(location[{ {}, 1 }] * 120) + 120, 80, 160)
    location[{ {}, 2 }] = torch.clamp(torch.round(location[{ {}, 2 }] * 160) + 160, 80, 240)
    local gw2 = math.floor(self.glimpse_width / 2)

    -- For each image in the batch, extract the glimpse
    for i = 1, batch_size do
        x = input_imgs[i]
        phi = location[i]

        -- Get the full resolution hypercube centered at phi
        -- Find the start and end index of the hypercube
        local slice = {}
        for d = 1, dim do
            table.insert(slice, { phi[d] - gw2 + 1, phi[d] + gw2 })
        end
        -- Index 1 for 1st dim due to #channels as first index
        hypercube = x[1][slice]

        -- Downsample the input. Create progressively downsampled patches
        local pyramid = image.gaussianpyramid(hypercube, scales)
        -- Scale all back to side length = glimpse_width
        for i = 1, #pyramid do
            pyramid[i] = image.scale(pyramid[i], self.glimpse_width)
        end

        -- Combine all patches into single tensor
        glimpse = torch.zeros(pyramid[1]:size())
        for i = 1, #pyramid do
            local corner1 = (i - 1) * (self.width / 2) + 1
            local corner2 = self.glimpse_width - corner1 + 1
            local slice = {}
            for j = 1, dim do
                table.insert(slice, { corner1, corner2 })
            end
            glimpse[slice] = pyramid[#pyramid - (i - 1)][slice]
        end
        input_glimpses[{ i, 1, {}, {} }] = glimpse
    end

    -- Forward pass
    self.encoder:forward(input_glimpses)
    self.output = self.encoder:get(1).output

    return self.output
end

--[[
 SpatialGlimpse:updateGradInput(inputTable, gradOutput)
    Controls the gradient flow through this module. For our spatial glimpse,
    we do not want any gradients updating the encoder.

]]
function SpatialGlimpse:updateGradInput(inputTable, gradOutput)
    -- Do not update any gradients
    local input, location = unpack(inputTable)
    local gradInput, gradLocation = unpack(self.gradInput)
    input, location = self:toBatch(input, 3), self:toBatch(location, 1)
    -- input is of size (batch_size, n_channel, img H, img W)
    gradInput:resizeAs(input):zero()
    gradLocation:resizeAs(location):zero()

    --    gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.height, self.width)
    --
    --    for sampleIdx = 1, gradOutput:size(1) do
    --        local gradOutputSample = gradOutput[sampleIdx]
    --        local gradInputSample = gradInput[sampleIdx]
    --        local yx = location[sampleIdx] -- height, width
    --        -- (-1,-1) top left corner, (1,1) bottom right corner of image
    --        local y, x = yx:select(1, 1), yx:select(1, 2)
    --        -- (0,0), (1,1)
    --        y, x = (y + 1) / 2, (x + 1) / 2
    --
    --        -- for each depth of glimpse : pad, crop, downscale
    --        local glimpseWidth = self.width
    --        local glimpseHeight = self.height
    --        for depth = 1, self.depth do
    --            local src = gradOutputSample[depth]
    --            if depth > 1 then
    --                glimpseWidth = glimpseWidth * self.scale
    --                glimpseHeight = glimpseHeight * self.scale
    --            end
    --
    --            -- add zero padding (glimpse could be partially out of bounds)
    --            local padWidth = math.floor((glimpseWidth - 1) / 2)
    --            local padHeight = math.floor((glimpseHeight - 1) / 2)
    --            self._pad:resize(input:size(2), input:size(3) + padHeight * 2, input:size(4) + padWidth * 2):zero()
    --
    --            local h, w = self._pad:size(2) - glimpseHeight, self._pad:size(3) - glimpseWidth
    --            local y, x = math.min(h, math.max(0, y * h)), math.min(w, math.max(0, x * w))
    --            local pad = self._pad:narrow(2, y + 1, glimpseHeight):narrow(3, x + 1, glimpseWidth)
    --
    --            -- upscale glimpse for different depths
    --            if depth == 1 then
    --                pad:copy(src)
    --            else
    --                self._crop:resize(input:size(2), glimpseHeight, glimpseWidth)
    --
    --                if torch.type(self.module) == 'nn.SpatialAveragePooling' then
    --                    local poolWidth = glimpseWidth / self.width
    --                    assert(poolWidth % 2 == 0)
    --                    local poolHeight = glimpseHeight / self.height
    --                    assert(poolHeight % 2 == 0)
    --                    self.module.kW = poolWidth
    --                    self.module.kH = poolHeight
    --                    self.module.dW = poolWidth
    --                    self.module.dH = poolHeight
    --                end
    --
    --                pad:copy(self.module:updateGradInput(self._crop, src))
    --            end
    --
    --            -- copy into gradInput tensor (excluding padding)
    --            gradInputSample:add(self._pad:narrow(2, padHeight + 1, input:size(3)):narrow(3, padWidth + 1, input:size(4)))
    --        end
    --    end

    self.gradInput[1] = self:fromBatch(gradInput, 1)
    self.gradInput[2] = self:fromBatch(gradLocation, 1)

    return self.gradInput
end
