## Overview

This is the PyTorch code for the paper [MaskVD: Region Masking for Efficient Video Object Detection](https://arxiv.org/abs/2407.12067).

## Environment Setup

To create environment, run
```
conda env create -f environment.yml
```

## Running Scripts

Scripts should be run from the repo's base directory using the respective configuration files present in `configs`.

```
python scripts/evaluate/vitdet_vid_mask.py base_672_maskvd
```



## Data

For `ImageNet-VID`, please follow the instructions [here](https://github.com/WISION-Lab/eventful-transformer).

VID requires a manual download. Download `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) and place it at `./data/vid/data.tar`. The VID class will take care of unpacking and preparing the data on first use.

For `KITTI`, use util/convert_kittitrack_to_coco.py to convert annotations to COCO format. 
For more information, please follow the instructions in [CenterTrack](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md).

## Weights
The pre-trained `ImageNet-VID` weights can be downloaded from [here](https://drive.google.com/drive/folders/1Lv2DO0CA7GY_PrYxNJNl_TwKtkW2ksEv?usp=sharing) and has to be placed in the `weights/` folder.

## Other Setup

Scripts assume that the current working directory is on the Python path. In the Bash shell, run
```
export PYTHONPATH="$PYTHONPATH:."
```


## Acknowledgement

This code repo is adapted from [Eventful-Transformer](https://github.com/WISION-Lab/eventful-transformer).

## Citation
If you find this repo useful for your research, please consider citing the following work:
```
@article{sarkar2024maskvd,
  title={MaskVD: Region Masking for Efficient Video Object Detection},
  author={Sarkar, Sreetama and Datta, Gourav and Kundu, Souvik and Zheng, Kai and Bhattacharyya, Chirayata and Beerel, Peter A},
  journal={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2025}
}