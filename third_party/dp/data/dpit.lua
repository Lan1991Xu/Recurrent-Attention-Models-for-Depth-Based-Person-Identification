------------------------------------------------------------------------
--[[ DPI-T Dataset Loader ]] --
-- Albert Haque, 2015-10-16
------------------------------------------------------------------------
require 'image'

local Dpit, DataSource = torch.class("dp.Dpit", "dp.DataSource")
Dpit.isDpit = true

Dpit._name = 'Dpit'
Dpit._image_size = { 320, 240, 1 }
Dpit._image_axes = 'bhwc'
Dpit._feature_size = 1 * 320 * 240
Dpit._classes = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }

function Dpit:__init(config)
    config = config or {}
    assert(torch.type(config) == 'table' and not config[1],
        "Constructor requires key-value arguments")
    local args, load_all, input_preprocess, target_preprocess
    args, self._valid_ratio, self._train_file, self._test_file,
    self._data_path, self._scale, self._binarize, self._shuffle,
    self._download_url, load_all, input_preprocess,
    target_preprocess
    = xlua.unpack({ config },
        'Dpit',
        'Dpit reidentification problem' ..
                'Note: Train and valid sets are already shuffled.',
        {
            arg = 'valid_ratio',
            type = 'number',
            default = 1 / 6,
            help = 'proportion of training set to use for cross-validation.'
        },
        {
            arg = 'train_file',
            type = 'string',
            default = 'train.th7',
            help = 'name of training file'
        },
        {
            arg = 'test_file',
            type = 'string',
            default = 'test.th7',
            help = 'name of test file'
        },
        {
            arg = 'data_path',
            type = 'string',
            default = '../datasets/',
            help = 'path to data repository'
        },
        {
            arg = 'scale',
            type = 'table',
            help = 'bounds to scale the values between. [Default={0,1}]'
        },
        {
            arg = 'binarize',
            type = 'boolean',
            help = 'binarize the inputs (0s and 1s)',
            default = false
        },
        {
            arg = 'shuffle',
            type = 'boolean',
            help = 'shuffle different sets',
            default = false
        },
        {
            arg = 'download_url',
            type = 'string',
            default = '',
            help = 'URL from which to download dataset if not found on disk.'
        },
        {
            arg = 'load_all',
            type = 'boolean',
            help = 'Load all datasets : train, valid, test.',
            default = true
        },
        {
            arg = 'input_preprocess',
            type = 'table | dp.Preprocess',
            help = 'to be performed on set inputs, measuring statistics ' ..
                    '(fitting) on the train_set only, and reusing these to ' ..
                    'preprocess the valid_set and test_set.'
        },
        {
            arg = 'target_preprocess',
            type = 'table | dp.Preprocess',
            help = 'to be performed on set targets, measuring statistics ' ..
                    '(fitting) on the train_set only, and reusing these to ' ..
                    'preprocess the valid_set and test_set.'
        })
    if (self._scale == nil) then
        self._scale = { 0, 1 }
    end
    if load_all then
        print('Loading training data...')
        self:loadTrainValid()
        print('Loading test set data...')
        self:loadTest()
    end
    DataSource.__init(self, {
        train_set = self:trainSet(),
        valid_set = self:validSet(),
        test_set = self:testSet(),
        input_preprocess = input_preprocess,
        target_preprocess = target_preprocess
    })
end

function Dpit:loadTrainValid()
    --Data will contain a tensor where each row is an example, and where
    --the last column contains the target class.
    --local data = self:loadData(self._train_file, self._download_url)
    local data = self:loadLocalData('train')
    -- train
    local start = 1
    local size = math.floor(data[1]:size(1) * (1 - self._valid_ratio))
    self:trainSet(self:createDataSet(data[1]:narrow(1, start, size), data[2]:narrow(1, start, size), 'train'))
    -- valid
    if self._valid_ratio == 0 then
        print "Warning : No Valid Set due to valid_ratio == 0"
        return
    end
    start = size
    size = data[1]:size(1) - start
    self:validSet(self:createDataSet(data[1]:narrow(1, start, size), data[2]:narrow(1, start, size), 'valid'))
    return self:trainSet(), self:validSet()
end

function Dpit:loadTest()
    --local test_data = self:loadData(self._test_file, self._download_url)
    local test_data = self:loadLocalData('test')
    self:testSet(self:createDataSet(test_data[1], test_data[2], 'test'))
    return self:testSet()
end

--Creates an Dpit Dataset out of inputs, targets and which_set
function Dpit:createDataSet(inputs, targets, which_set)
    if self._shuffle then
        local indices = torch.randperm(inputs:size(1)):long()
        inputs = inputs:index(1, indices)
        targets = targets:index(1, indices)
    end
    if self._binarize then
        DataSource.binarize(inputs, 128)
    end
    if self._scale and not self._binarize then
        DataSource.rescale(inputs, self._scale[1], self._scale[2])
    end
    -- class 0 will have index 1, class 1 index 2, and so on.
    targets:add(1)
    -- construct inputs and targets dp.Views
    local input_v, target_v = dp.ImageView(), dp.ClassView()
    input_v:forward(self._image_axes, inputs)
    target_v:forward('b', targets)
    target_v:setClasses(self._classes)
    -- construct dataset
    local ds = dp.DataSet { inputs = input_v, targets = target_v, which_set = which_set }
    ds:ioShapes('bhwc', 'b')
    return ds
end

function Dpit:loadData(file_name, download_url)
    local path = DataSource.getDataPath {
        name = self._name,
        url = download_url,
        decompress_file = file_name,
        data_dir = self._data_path
    }
    -- backwards compatible with old binary format
    local status, data = pcall(function() return torch.load(path, "ascii") end)
    if not status then
        return torch.load(path, "binary")
    end
    return data
end


--[[
tableToTensor(table)
  Takes a lua table and converts it to a torch.Tensor

Args:
  table (table): Lua table, can be any number of dimensions

Returns:
  tensor (torch.Tensor): Original lua table converted to a torch.Tensor
]]
function Dpit:tableToTensor(table)
    local tensorSize = table[1]:size()
    local tensorSizeTable = { -1 }
    for i = 1, tensorSize:size(1) do
        tensorSizeTable[i + 1] = tensorSize[i]
    end
    merge = nn.Sequential():add(nn.JoinTable(1)):add(nn.View(unpack(tensorSizeTable)))
    tensor = merge:forward(table)
    return tensor
end

function Dpit:loadLocalData(mode)
    filename = self._data_path .. '/DPI-T_' .. mode .. '_depth_map.h5'
    local input_file = hdf5.open(filename, 'r')
    local data = input_file:read('/'):all()
    input_file:close()

    local n_examples = 0
    for p_id, person in pairs(data) do
        for v_id, video in pairs(person) do
            n_examples = n_examples + video:size()[1]
        end
    end
    idx = 1
    local X = torch.Tensor(n_examples, Dpit._image_size[2], Dpit._image_size[1], Dpit._image_size[3])
    local y = torch.Tensor(n_examples)
    for p_id, person in pairs(data) do
        label = tonumber(p_id) + 1 -- Label must start at 1
        for v_id, video in pairs(person) do
            local n_frames = video:size()[1]
            for i = 1, n_frames do
                X[{ idx, {}, {}, 1 }] = video[i]
                y[idx] = label
                idx = idx + 1
            end
        end
    end

    result = { X, y }
    return result
end

function Dpit:scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls "' .. directory .. '"'):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end
