# Recurrent Attention Models for Person Identification

<img src="attention.gif" width="100%">


**Recurrent Attention Models for Depth-Based Person Identification** [[website](https://www.albert.cm/projects/ram_person_id/)] [[arxiv](http://arxiv.org/abs/1603.07076)] [[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Haque_Recurrent_Attention_Models_CVPR_2016_paper.pdf)]
<br>
[Albert Haque](https://www.albert.cm), [Alexandre Alahi](http://www.ivpe.com/), [Li Fei-Fei](http://vision.stanford.edu/feifeili/)
<br>
[CVPR 2016](http://cvpr2016.thecvf.com)

## Install the Dependencies

1. Clone the repo: `git clone https://github.com/ahaque/ram_person_id.git`

2. (Optional) If you are using GPU/CUDA:

        luarocks install cutorch
        luarocks install cunn
        luarocks install cunnx

3. Install [HDF5](https://www.hdfgroup.org/HDF5/): `sudo apt install libhdf5-serial-dev hdf5-tools`

4. Install more Lua packages (nn, dpnn, image, etc.): `./install_custom_rocks.sh`

5. Confirm that the custom rocks were installed correctly by running torch and checking if `dp.Dpit` exists. From the bash/command line:


        $ th
        th> require 'dp'
        th> dp.Dpit
        table: 0x41d27e80       [0.0001s]

    If you see an error, you may need to refresh the lua package cache. Run: `package.loaded` from the torch command line.

## Datasets

The datasets can be downloaded from the publishers' websites:

* **DPI-T**: Depth-Based Person Identification from Top View [[website](https://www.albert.cm/projects/ram_person_id/#dataset)]
* **BIWI**: BIWI RGBD-ID Dataset [[website](http://robotics.dei.unipd.it/reid/index.php/8-dataset/2-overview-biwi)]
* **IAS**: IAS-Lab RGBD-ID [[website](http://robotics.dei.unipd.it/reid/index.php/8-dataset/5-overview-iaslab)]
* **IIT PAVIS**: RGB-D person Re-Identification Dataset [[website](https://www.iit.it/en/datasets/rgbdid)]

To automatically download the DPI-T dataset, run: `./download_datasets.sh`.

## Training

### Encoder: Convolutional Autoencoder

1. Navigate to [src/encoder](src/encoder).

2. Train the encoder:

        th train.lua
        th train.lua -gpuid 0

3. The model will be saved to `opt.dir` every `opt.save_interval` epochs.

### Recurrent Attention Model

Note: You must have a saved encoder model before training the recurrent attention model.

1. Navigate to the [src](src/) folder. The file [opts.lua](src/opts.lua) contains the training options, model architecture definition, and optimization settings.

        th train.lua
        th train.lua -gpuid 0

2. Done.

## References

Haque, A., Alahi, A., Fei-Fei, L.: Recurrent attention models for depth-based person identification. CVPR, 2016.

Bibtex:

    @inproceedings{haque2016cvpr,
        author = {Haque, Albert and Alahi, Alexandre and Fei-Fei, Li},
        title = {Recurrent Attention Models for Depth-Based Person Identification},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2016}
    }
