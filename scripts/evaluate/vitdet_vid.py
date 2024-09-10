#!/usr/bin/env python3

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict, tee_print

def collate_fn(batch):
    batch = list(zip(*batch))
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    # return tuple(batch)
    batch[0] = torch.stack(batch[0])
    return tuple(batch)

def evaluate_vitdet_metrics(device, model, data, config, output_file):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    latency = 0
    memory = 0
    count = 0

    # Latency calculation
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # --------------- GPU Warmup for correct latency calculation ---------------
    vid_item = DataLoader(data[0], batch_size=1, collate_fn=collate_fn)
    model.reset()
    # model.backbone.reset_self()
    for i, (frame, annotations) in enumerate(vid_item):
        with torch.inference_mode():
            results = model(frame.to(device))
    # -------------------------------------------------------------------------

    for _, vid_item in tqdm(zip(range(50), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        n_frames += len(vid_item)
        model.reset()
        # model.backbone.reset_self()
        for frame, annotations in vid_item:
            with torch.inference_mode():
                starter.record()
                results, _ = model(frame.to(device))
                # outputs.extend(model(frame.to(device)))
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)

                outputs.extend(results)
                count += 1
                latency += curr_time
                MB = 1024 * 1024
                memory += torch.cuda.max_memory_allocated() / MB

            # labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
            labels.extend([dict_to_device(annotation, device) for annotation in annotations])

    tee_print(f'Latency: {latency / count} ms', output_file)
    tee_print(f'Memory: {memory / count} MB', output_file)

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
    data = VID(
        Path("/data1", "vid_data"),
        split=config["split"],
        tar_path=Path("/data1", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()
