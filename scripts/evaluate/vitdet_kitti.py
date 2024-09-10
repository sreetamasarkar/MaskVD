#!/usr/bin/env python3

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

import sys


from datasets.kitti_tracking import build
from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict


def evaluate_vitdet_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    print(data)
    print(len(data))
    print("##########################")
    n_items = config.get("n_items", len(data))
    count = 0

    vid_item = DataLoader(data, batch_size=1)
    n_frames += len(vid_item)
    model.reset()
    #print(vid_item)
    #print("##########################")
    for _, (frame, annotations) in  tqdm(zip(range(5), vid_item), total=n_items, ncols=0):
        with torch.inference_mode():
            outputs.extend(model(frame.to(device)))
        labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))

    # for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
    #     count += 1
    #     vid_item = DataLoader(vid_item, batch_size=1)
    #     n_frames += len(vid_item)
    #     model.reset()
    #     #print(vid_item)
    #     #print("##########################")
    #     for frame, annotations in vid_item:
    #         # print(f'{count}###')
    #         # count += 1
    #         #print(frame)
    #         #print(annotations)
    #         #print("##########################")
    #         #break
    #         with torch.inference_mode():
    #             outputs.extend(model(frame.to(device)))
    #         labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
        #break
    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}

def main():
    config = initialize_run(config_location=Path("configs", "evaluate", "vitdet_vid"))
    long_edge = max(config["model"]["input_shape"][-2:])
    data = build("val", args=None)
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)

if __name__ == "__main__":
    main()