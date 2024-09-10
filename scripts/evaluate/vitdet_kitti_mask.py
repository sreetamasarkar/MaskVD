#!/usr/bin/env python3

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from datasets.kitti_tracking import build
from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict
from utils.misc import tee_print
from detectron2.utils.events import EventStorage
from detectron2.data.detection_utils import annotations_to_instances, BoxMode
from backbones.base import dict_string
from util.lr_sched import LR_Scheduler
from datetime import datetime
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.misc import dict_to_device, squeeze_dict, get_pytorch_device, set_policies
from backbones.policies import TokenNormTopK
import torch.nn.functional as F
import numpy as np
from utils.image import pad_to_size
from torchprofile import profile_macs


def get_region_mask_dynamic(results, image_shape, conf_threshold=0.5, region_size=16, margin=0):
    mask = torch.zeros(image_shape)
    for result in results:
        bb = False
        for i, bbox in enumerate(result['boxes']):
            if result['scores'][i] > conf_threshold:
                bb = True
                x1, y1, x2, y2 = bbox
                mask[int(y1)-margin:int(y2)+margin, int(x1)-margin:int(x2)+margin] = 1
    if not bb:
        mask = torch.ones(image_shape)
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index = torch.nonzero(y.flatten() > 0).reshape(1, -1)
    window_size = 14
    weight_win = torch.ones(1, 1, region_size*window_size, region_size*window_size)
    y_win = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight_win, stride=region_size*window_size)
    window_index = torch.nonzero(y_win.flatten() > 0).squeeze(-1)
    sparsity = 1 - mask_index.shape[1] / y.numel()
    return mask_index, window_index, sparsity, y.numel()

def topkmask(feature, sparsity=0.5):
    # import pdb; pdb.set_trace()
    feature_flat = feature.flatten(1)
    n_tokens = feature_flat.shape[1]
    left_tokens = int(n_tokens * sparsity)
    value, idx = torch.topk(feature_flat, left_tokens, dim=1, sorted=False)
    # min_value = value[:,-1].unsqueeze(-1).unsqueeze(-1).expand(-1, N, C) # min value to keep for each feature map
    # return (feature > min_value) + 0
    return idx

def get_region_mask_static(region_size=16, region_sparsity=0.5):
    with open('heatmap_kitti.npy', 'rb') as f:
        heat_map = np.load(f)
    heatmap = torch.tensor(heat_map/heat_map.max(), dtype=torch.float)
    heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), (370, 672), mode='bilinear')
    heatmap = pad_to_size(heatmap, (672, 672))
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(heatmap, weight, stride=region_size)
    index = topkmask(y, sparsity=region_sparsity)
    # scores = y.flatten(1).softmax(1)
    return index

def val_pass(device, model, data, config, output_file):
    model.counting()
    model.clear_counts()
    model.eval()
    n_frames = 0
    outputs = []
    labels = []
    total_sparsity = 0 
    latency = 0
    memory = 0
    n_items = config.get("n_items", len(data))
    count = 0

    vid_item = DataLoader(data, batch_size=1)
    n_frames += len(vid_item)
    model.reset()

    # Latency calculation
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # --------------- GPU Warmup for correct latency calculation ---------------
    for _, (frame, annotations) in  tqdm(zip(range(10), vid_item), total=n_items, ncols=0):
        with torch.inference_mode():
            # images, x = model.pre_backbone(frame.to(device))
            results, _ = model(frame.to(device))
    model.clear_counts()
    # -------------------------------------------------------------------------
    if config["sparsity"] < 1.0:
        mask_index_static = get_region_mask_static(region_sparsity=1 - config["sparsity"]) # sparsity is keep rate 

    for _, (frame, annotations) in  tqdm(zip(range(n_items), vid_item), total=n_items, ncols=0):
        if annotations['frame_id'] == 1:
            model.reset()
            step = 0
        with torch.inference_mode():
            if step % config["period"] == 0:
                mask_index = None
                sparsity = 0
                window_index = None
            else:
                mask_index, window_index, sparsity, n_tokens = get_region_mask_dynamic(results, (672,672), conf_threshold=config["conf"], region_size=16, margin=config["margin"])
                # mask_index, sparsity, n_tokens = get_original_mask(annotations[0], (672,672), region_size=16)
                # combine with static mask
                if config["sparsity"] < 1.0:
                    # get union of static and dynamic mask preserving shape
                    mask_index = torch.unique(torch.cat((mask_index, mask_index_static), dim=1)).reshape(1, -1)
                keep_rate = mask_index.shape[1] / n_tokens
                sparsity = 1 - keep_rate
                mask_index = mask_index.to(device)

            starter.record()
            results, _ = model(frame.to(device), mask_index)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            outputs.extend(results)

            step += 1
            count += 1
            total_sparsity += sparsity
            latency += curr_time
            MB = 1024 * 1024
            memory += torch.cuda.max_memory_allocated() / MB

        labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))


    counts = model.total_counts() / n_frames

    tee_print(f'Sparsity: {total_sparsity / count}', output_file)
    tee_print(f'Latency: {latency / count} ms', output_file)
    tee_print(f'Memory: {memory / count} MB', output_file)
    if config["model"]["backbone_config"]["backbone"] == "windowed":
        tee_print(f'GFLOPs: {sum([value for _, value in counts.items()]) / 1e9}', output_file)
    else:
        # G
        mask_index_test = get_region_mask_static(region_sparsity=1 - total_sparsity/count) # sparsity is keep rate
        _, x = model.pre_backbone(frame.to(device))
        macs = profile_macs(model.backbone, (x, None, 12, mask_index_test))
        tee_print(f'GFLOPs (torchprofile): {macs / 1e9}', output_file)

    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def main():
    config = initialize_run(config_location=Path("configs", "evaluate", "vitdet_kitti"))
    long_edge = max(config["model"]["input_shape"][-2:])
    n_classes = config["nb_classes"]
    # train_data = build("train", n_classes, args=None)
    val_data = build("val", n_classes, args=None)
    
    # run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)
    output_dir = Path(config["_output"])
    output_file = open(output_dir / "output.txt", 'a')

    device = get_pytorch_device()
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    # Load and set up the model.
    config["model"]["classes"] = n_classes # modify number of classes for Kitti, do not include Misc and Don't Care
    config["model"]["mask"] = True
    model = ViTDet(**(config["model"]))
    # set_policies(model, TokenNormTopK, k=512)
    # msg = model.load_state_dict(torch.load(config["weights"]), strict=False)
    # Load model
    ckpt = torch.load(config["weights"])

    # Delete parameters for mismatch due to num_class
    # for key in list(ckpt.keys()):
    #     if "roi_heads.box_predictor.cls_score.weight" in key or "roi_heads.box_predictor.cls_score.bias" in key or "roi_heads.box_predictor.bbox_pred.weight" in key or "roi_heads.box_predictor.bbox_pred.bias" in key:
    #         del ckpt[key]
    # original_keys = list(ckpt.keys())
    # for key in original_keys:
    #     if "input_layer_norm" in key:
    #         new_key = key.replace("input_layer_norm", "norm1")
    #         ckpt[new_key] = ckpt.pop(key)
    #     elif "qkv" in key:
    #         new_key = key.replace("qkv", "attn.qkv")
    #         ckpt[new_key] = ckpt.pop(key)
    #     elif "mlp_layer_norm" in key:
    #         new_key = key.replace("mlp_layer_norm", "norm2")
    #         ckpt[new_key] = ckpt.pop(key)
    #     elif "mlp_1" in key:
    #         new_key = key.replace("mlp_1", "mlp.fc1")
    #         ckpt[new_key] = ckpt.pop(key)
    #     elif "mlp_2" in key:
    #         new_key = key.replace("mlp_2", "mlp.fc2")
    #         ckpt[new_key] = ckpt.pop(key)
    #     elif "position_encoding" in key:
    #         new_key = key.replace("position_encoding.encoding", "pos_embed.encoding")
    #         ckpt[new_key] = ckpt.pop(key)
    #     elif "projection" in key:
    #         new_key = key.replace("projection", "attn.proj")
    #         ckpt[new_key] = ckpt.pop(key)
    #     elif "embedding" in key:
    #         new_key = key.replace("embedding.conv", "backbone.patch_embed.proj")
    #         ckpt[new_key] = ckpt.pop(key)
    # # interpolate_pos_embed(model.backbone, ckpt, key="backbone.pos_embed")
    msg = model.load_state_dict(ckpt, strict=False)
    tee_print(msg, output_file)

    # TODO: map backbone weights to timm ViT model
    model = model.to(device)

    tee_print("Evaluating....", output_file)
    results = val_pass(device, model, val_data, config, output_file)
    
    # Print and save results.
    if isinstance(results, dict):
        for key, val in results.items():
            tee_print(key.capitalize(), output_file)
            tee_print(dict_string(val), output_file)
    else:
        tee_print(results, output_file)
    tee_print("", output_file) 

    output_file.close()


if __name__ == "__main__":
    main()