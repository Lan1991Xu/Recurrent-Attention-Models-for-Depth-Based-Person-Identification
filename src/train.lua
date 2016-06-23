require 'dp'
require 'rnn'
require 'hdf5'
torch = require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local opts = require 'opts'

local opt = opts.parse(arg)
print(opt)

-- Configure GPU/CPU
if opt.gpuid >= 0 then
    print('Loading Torch GPU libraries...')
    local ok, _ = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('Using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1
    end
end

-- Load the encoder model
local f = io.open(opt.encoder, 'r')
if f ~= nil then
    print('Loading encoder model from: ' .. opt.encoder)
    encoder = torch.load(opt.encoder, 'ascii')
else
    print('Error: Unable to load encoder model from: ' .. opt.encoder)
    print('Terminating.')
    do return end
end

print(encoder)

--[[data]] --
if opt.dataset == 'TranslatedMnist' then
    ds = torch.checkpoint(paths.concat(dp.DATA_DIR, 'checkpoint/dp.TranslatedMnist.t7'),
        function() return dp[opt.dataset]() end,
        opt.overwrite)
else
    ds = dp[opt.dataset]()
end

--[[Model]] --
-- glimpse network (rnn input layer)
locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2))
locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
locationSensor:add(nn[opt.transfer]())

glimpseSensor = nn.Sequential()
glimpseSensor:add(nn.SpatialGlimpse(encoder, opt.patch_width, opt.n_patch, opt.glimpse_blur):float())
glimpseSensor:add(nn.Linear(opt.embedding_size, opt.glimpseHiddenSize))
glimpseSensor:add(nn[opt.transfer]())

glimpse = nn.Sequential()
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1, 1))
glimpse:add(nn.Linear(opt.glimpseHiddenSize + opt.locatorHiddenSize, opt.imageHiddenSize))
glimpse:add(nn[opt.transfer]())
glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

-- rnn recurrent layer
if opt.FastLSTM then
    recurrent = nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)
else
    recurrent = nn.Linear(opt.hiddenSize, opt.hiddenSize)
end


-- recurrent neural network
rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn[opt.transfer](), 99999)

-- actions (locator)
locator = nn.Sequential()
locator:add(nn.Linear(opt.hiddenSize, 2))
locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
locator:add(nn.ReinforceNormal(2 * opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
assert(locator:get(3).stochastic == opt.stochastic, "Please update the dpnn package : luarocks install dpnn")
locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
locator:add(nn.MulConstant(opt.unitPixels * 2 / ds:imageSize("h")))

attention = nn.RecurrentAttention(rnn, locator, opt.rho, { opt.hiddenSize })

-- model is a reinforcement learning agent
agent = nn.Sequential()
agent:add(nn.Convert(ds:ioShapes(), 'bchw'))
agent:add(attention)

-- classifier :
agent:add(nn.SelectTable(-1))
agent:add(nn.Linear(opt.hiddenSize, #ds:classes()))
agent:add(nn.LogSoftMax())

-- add the baseline reward predictor
seq = nn.Sequential()
seq:add(nn.Constant(1, 1))
seq:add(nn.Add(1))
concat = nn.ConcatTable():add(nn.Identity()):add(seq)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

-- output will be : {classpred, {classpred, basereward}}
agent:add(concat2)

if opt.uniform > 0 then
    for k, param in ipairs(agent:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
    end
end

--[[Propagators]] --
opt.decayFactor = (opt.minLR - opt.learningRate) / opt.saturateEpoch

train = dp.Optimizer {
    loss = nn.ParallelCriterion(true):add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())):add(nn.ModuleCriterion(nn.VRClassReward(agent, opt.rewardScale), nil, nn.Convert())),
    epoch_callback = function(model, report) -- called every epoch
    if report.epoch > 0 then
        opt.learningRate = opt.learningRate + opt.decayFactor
        opt.learningRate = math.max(opt.minLR, opt.learningRate)
        if not opt.silent then
            print("learningRate", opt.learningRate)
        end
    end
    end,
    callback = function(model, report)
        if opt.cutoffNorm > 0 then
            local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
            opt.meanNorm = opt.meanNorm and (opt.meanNorm * 0.9 + norm * 0.1) or norm
            if opt.lastEpoch < report.epoch and not opt.silent then
                print("mean gradParam norm", opt.meanNorm)
            end
        end
        model:updateGradParameters(opt.momentum) -- affects gradParams
        model:updateParameters(opt.learningRate) -- affects params
        model:maxParamNorm(opt.maxOutNorm) -- affects params
        model:zeroGradParameters() -- affects gradParams
    end,
    feedback = dp.Confusion { output_module = nn.SelectTable(1) },
    sampler = dp.ShuffleSampler {
        epoch_size = opt.trainEpochSize,
        batch_size = opt.batchSize
    },
    progress = opt.progress
}


valid = dp.Evaluator {
    sampler = dp.Sampler { epoch_size = opt.validEpochSize, batch_size = opt.batchSize },
    progress = opt.progress
}
if not opt.noTest then
    tester = dp.Evaluator {
        sampler = dp.Sampler { epoch_size = opt.validEpochSize, batch_size = opt.batchSize },
        progress = opt.progress
    }
end

--[[Experiment]] --

xp = dp.Experiment {
    model = agent,
    optimizer = train,
    validator = valid,
    tester = tester,
    observer = {
        ad,
        dp.FileLogger(),
    },
    random_seed = os.time(),
    max_epoch = opt.maxEpoch
}

--[[GPU or CPU]] --
if opt.cuda then
    print "Using CUDA"
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.useDevice)
    xp:cuda()
else
    xp:float()
end

xp:verbose(not opt.silent)
if not opt.silent then
    print "Agent :"
    print(agent)
end

xp.opt = opt

if checksum then
    assert(math.abs(xp:model():parameters()[1]:sum() - checksum) < 0.0001, "Loaded model parameters were changed???")
end
xp:run(ds)
