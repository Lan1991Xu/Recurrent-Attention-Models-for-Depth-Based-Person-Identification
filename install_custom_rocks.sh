#!/usr/bin/env bash
luarocks install torch
luarocks install torchx
luarocks install nn
luarocks install image
luarocks install moses
cd ./third_party/dpnn
luarocks make rocks/dpnn-scm-1.rockspec
cd ../dp
luarocks make rocks/dp-scm-1.rockspec
cd ../rnn
luarocks make rocks/rnn-scm-1.rockspec
cd ../torch-hdf5
luarocks install hdf5-0-0.rockspec
cd ../..
