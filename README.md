## Overview

This is the PyTorch code for the paper "MaskVD: Region Masking for Efficient Video Object Detection".



## Running Scripts

Scripts should be run from the repo's base directory using the respective configuration files present in `configs`.

```
./scripts/evaluate/vitdet_vid_mask.py ./configs/evaluate/vitdet_vid/base_672_maskvd.yml
```


## Data

For `ImageNet-VID`, please follow the instructions [here](https://github.com/WISION-Lab/eventful-transformer)

VID requires a manual download. Download `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) and place it at `./data/vid/data.tar`. The VID class will take care of unpacking and preparing the data on first use.

For `KITTI`, use util/convert_kittitrack_to_coco.py to convert annotations to COCO format. 
For more information, please follow the instructions in [CenterTrack](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md)

## Other Setup

Scripts assume that the current working directory is on the Python path. In the Bash shell, run
```
export PYTHONPATH="$PYTHONPATH:."
```


## Acknowledgement

This code repo is based on [Eventful-Transformer](https://github.com/WISION-Lab/eventful-transformer)
