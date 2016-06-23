local M = {}

function M.parse(arg)

    --[[command line arguments]] --
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Train a Recurrent Model for Visual Attention')
    cmd:text('Example:')
    cmd:text('$> th rnn-visual-attention.lua > results.txt')
    cmd:text('Options:')
    cmd:option('-gpuid', -1, 'GPU ID to use. Default: -1 means CPU mode.')

    -- Encoder settings
    -- Model settings
    cmd:option('-embedding_size', 128, 'Size of the embedding (or bottleneck layer)')
    cmd:option('-patch_width', 8, 'Size of inner-most glimpse patch')
    cmd:option('-n_patch', 5, 'Number of patches per glimpse')
    cmd:option('-glimpse_blur', 2, 'Blur scale for each glimpse further from center. Uses: 1 / (glimpse# ^ glimpse_blur)')
    cmd:option('-encoder', './encoder/encoder_gw40_np5_i1900_gpu.bin', 'Binary file of the model saved by encoder/train.lua')

    cmd:option('-learningRate', 0.01, 'learning rate at t=0')
    cmd:option('-minLR', 0.00001, 'minimum learning rate')
    cmd:option('-saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
    cmd:option('-momentum', 0.9, 'momentum')
    cmd:option('-maxOutNorm', -1, 'max norm each layers output neuron weights')
    cmd:option('-cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
    cmd:option('-batchSize', 20, 'number of examples per batch')
    cmd:option('-maxEpoch', 2000, 'maximum number of epochs to run')
    cmd:option('-maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
    cmd:option('-transfer', 'ReLU', 'activation function')
    cmd:option('-uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
    cmd:option('-xpPath', '', 'path to a previously saved model')
    cmd:option('-progress', true, 'print progress bar')
    cmd:option('-silent', false, 'dont print anything to stdout')

    --[[ reinforce ]] --
    cmd:option('-rewardScale', 1, "scale of positive reward (negative is 0)")
    cmd:option('-unitPixels', 125, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
    cmd:option('-locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
    cmd:option('-stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

    --[[ glimpse layer ]] --
    cmd:option('-glimpseHiddenSize', 128, 'size of glimpse hidden layer')
    cmd:option('-locatorHiddenSize', 128, 'size of locator hidden layer')
    cmd:option('-imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')

    --[[ recurrent layer ]] --
    cmd:option('-rho', 7, 'back-propagate through time (BPTT) for rho time-steps')
    cmd:option('-hiddenSize', 256, 'number of hidden units used in Simple RNN.')
    cmd:option('-dropout', false, 'apply dropout on hidden neurons')

    --[[ data ]] --
    cmd:option('-dataset', 'Dpit', 'which dataset to use : Mnist | Dpit | Biwi | etc')
    cmd:option('-trainEpochSize', -1, 'number of train examples seen between each epoch')
    cmd:option('-validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
    cmd:option('-noTest', false, 'dont propagate through the test set')
    cmd:option('-overwrite', false, 'overwrite checkpoint')

    cmd:text()
    local opt = cmd:parse(arg or {})
    return opt
end

return M
