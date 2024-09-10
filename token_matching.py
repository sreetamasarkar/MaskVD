#!/usr/bin/env python3

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet_new import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict, get_pytorch_device
from torch import optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from detectron2.data.detection_utils import annotations_to_instances, BoxMode
from detectron2.utils.events import EventStorage
from utils.misc import tee_print
from backbones.base import dict_string
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

def collate_fn(batch):
    batch = list(zip(*batch))
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    # return tuple(batch)
    batch[0] = torch.stack(batch[0])
    return tuple(batch)

def train_pass(config, device, epoch, model, optimizer, data, tensorboard, output_file):
    model.train()
    n_items = config.get("n_items", len(data))
    step = 0
    total_loss = 0
    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        for frame, annotations in vid_item:
            step += 1
            # Convert annotations to instances
            annotation_list = []
            gt_instances = []
            for annotation in annotations:
                for i, bbox in enumerate(annotation['boxes']):
                    annotation_dict = {}
                    annotation_dict['bbox'] = bbox
                    annotation_dict['category_id'] = annotation['labels'][i]
                    annotation_dict['bbox_mode'] = BoxMode.XYXY_ABS
                    annotation_list.append(annotation_dict)
                gt_instance = annotations_to_instances(annotation_list, frame.shape[-2:], frame.shape[-2:])
                gt_instances.append(gt_instance.to(device))
            optimizer.zero_grad()
            with EventStorage() as storage:
                images, x = model.pre_backbone(frame.to(device))
                mask_index = model.get_region_mask_static()
                x = model.backbone(x, mask_id=mask_index)
                # x = model.backbone(x)
                x = x.transpose(-1, -2)
                x = x.view(x.shape[:-1] + model.backbone_input_size)
                x = model.pyramid(x)

                # Compute region proposals and bounding boxes.
                x = dict(zip(model.proposal_generator.in_features, x))

                proposals, proposal_losses = model.proposal_generator(images, x, gt_instances)
                _, detector_losses = model.roi_heads(images, x, proposals, gt_instances)
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            loss = sum(losses.values())
            # x = self.backbone.get_intermediate_layers(x, 1)[0] # get output of last layer 
            # results, losses = model.post_backbone(images, x)
            loss.backward()
            # total_loss += loss.item()
            if step % 8 == 0:
                tee_print(f'Loss: {loss.item()}', output_file)
                optimizer.step()
            # total_loss = 0
            if tensorboard is not None:
                tensorboard.add_scalar("train/loss", loss.item())
    
def val_pass(device, model, data, config):
    model.counting()
    model.clear_counts()
    model.eval()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    for _, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        vid_item = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
        n_frames += len(vid_item)
        model.reset()
        # model.backbone.reset_self()
        for frame, annotations in vid_item:
            with torch.inference_mode():
                results, _ = model(frame.to(device))
                outputs.extend(results)
            # labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
            labels.extend([dict_to_device(annotation, device) for annotation in annotations])

    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def train_vitdet(config, model_class, train_data, val_data, output_file):
    
    # Set up the optimizer.
    optimizer_class = getattr(optim, config["optimizer"])
    optimizer = optimizer_class(model.parameters(), **config["optimizer_kwargs"])

    # Set up TensorBoard logging.
    if "tensorboard" in config:
        base_name = config["tensorboard"]
        now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tensorboard = SummaryWriter(f"{base_name}_{now_str}")
    else:
        tensorboard = None

    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        tee_print(f"\nEpoch {epoch + 1}/{n_epochs}", flush=True, file=output_file)
        train_pass(config, device, epoch + 1, model, optimizer, train_data, tensorboard, output_file)
        results = val_pass(device, model, val_data, config)
       
        # Print and save results.
        if isinstance(results, dict):
            for key, val in results.items():
                tee_print(key.capitalize(), output_file)
                tee_print(dict_string(val), output_file)
        else:
            tee_print(results, output_file)
        tee_print("", output_file) 
        
    if tensorboard is not None:
        tensorboard.close()

    # Save the final weights.
    weight_path = config["output_weights"]
    torch.save(model.state_dict(), weight_path)
    tee_print(f"Saved weights to {weight_path}", flush=True, file=output_file)


def maximum_bipartite_matching(input1, input2):
    # Calculate pairwise similarity between tokens in input1 and input2
    similarities = torch.matmul(F.normalize(input1, dim=-1), F.normalize(input2, dim=-1).transpose(1, 2))
    
    # Convert similarities to a numpy array
    similarity_matrix = similarities.squeeze().cpu().detach().numpy()

    # Perform linear sum assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Minimization problem, hence negate similarity matrix

    # Create the matching matrix
    matching_matrix = np.zeros_like(similarity_matrix, dtype=np.float32)
    matching_matrix[row_ind, col_ind] = 1.0

    return matching_matrix, row_ind, col_ind


def main():
    config = initialize_run(config_location=Path("configs", "train", "vitdet_vid"))
    long_edge = max(config["model"]["input_shape"][-2:])
    train_data = VID(
        Path("/data1", "vid_data"),
        split="vid_train",
        tar_path=Path("/data1", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    val_data = VID(
        Path("/data1", "vid_data"),
        split="vid_val",
        tar_path=Path("/data1", "vid_data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )

    device = get_pytorch_device()
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    # Load and set up the model.
    model = ViTDet(**(config["model"]))
    # msg = model.load_state_dict(torch.load(config["weights"]), strict=False)
    # print(msg)
    # Load model
    # from eventful_transformer.backbone_vit import interpolate_pos_embed
    ckpt = torch.load(config["weights"])
    original_keys = list(ckpt.keys())
    for key in original_keys:
        if "input_layer_norm" in key:
            new_key = key.replace("input_layer_norm", "norm1")
            ckpt[new_key] = ckpt.pop(key)
        elif "qkv" in key:
            new_key = key.replace("qkv", "attn.qkv")
            ckpt[new_key] = ckpt.pop(key)
        elif "mlp_layer_norm" in key:
            new_key = key.replace("mlp_layer_norm", "norm2")
            ckpt[new_key] = ckpt.pop(key)
        elif "mlp_1" in key:
            new_key = key.replace("mlp_1", "mlp.fc1")
            ckpt[new_key] = ckpt.pop(key)
        elif "mlp_2" in key:
            new_key = key.replace("mlp_2", "mlp.fc2")
            ckpt[new_key] = ckpt.pop(key)
        elif "position_encoding" in key:
            new_key = key.replace("position_encoding.encoding", "pos_embed.encoding")
            ckpt[new_key] = ckpt.pop(key)
        elif "projection" in key:
            new_key = key.replace("projection", "attn.proj")
            ckpt[new_key] = ckpt.pop(key)
        elif "embedding" in key:
            new_key = key.replace("embedding.conv", "backbone.patch_embed.proj")
            ckpt[new_key] = ckpt.pop(key)
    # interpolate_pos_embed(model.backbone, ckpt, key="backbone.pos_embed")
    msg = model.load_state_dict(ckpt, strict=False)
    print(msg)

    # TODO: map backbone weights to timm ViT model
    model = model.to(device)

    # Dataloader
    val_data.video_info.sort(key=lambda v: v["video_id"])
    vid_item = val_data[0]
    vid_loader = DataLoader(vid_item, batch_size=1, collate_fn=collate_fn)
    frame1, annotations1 = next(iter(vid_loader))
    vid_loader_iter = iter(vid_loader)
    for i in range(50):
        batch = next(vid_loader_iter)
    frame2, annotations2 = batch

    image1, x = model.pre_backbone(frame1.to(device))
    im1_embedding = model.backbone.patch_embed(x)

    image2, x = model.pre_backbone(frame2.to(device))
    im2_embedding = model.backbone.patch_embed(x)

    region_size = 16
    rmask1 = torch.zeros(x.shape[-2:])
    for annotation in annotations1:
        for i, bbox in enumerate(annotation['boxes']):
            x1, y1, x2, y2 = bbox
            rmask1[int(y1):int(y2), int(x1):int(x2)] = 1
    plt.imsave('rmask1.png', rmask1.detach().cpu().numpy())
    weight = torch.ones(1, 1, region_size, region_size)
    y = F.conv2d(rmask1.unsqueeze(0).unsqueeze(0), weight, stride=region_size)
    mask_index1 = torch.nonzero(y.squeeze().flatten() > 0)
    print(mask_index1)

    rmask2 = torch.zeros(x.shape[-2:])
    for annotation in annotations2:
        for i, bbox in enumerate(annotation['boxes']):
            x1, y1, x2, y2 = bbox
            rmask2[int(y1):int(y2), int(x1):int(x2)] = 1      
    plt.imsave('rmask2.png', rmask2.detach().cpu().numpy())

    std = model.preprocessing.normalization.std
    mean = model.preprocessing.normalization.mean
    frame1_img = image1[0].cpu().numpy().transpose(1, 2, 0)
    frame2_img = image2[0].cpu().numpy().transpose(1, 2, 0)
    max_val, min_val = frame1_img.max(), frame1_img.min()
    plt.imsave('frame1.png', (frame1_img - frame1_img.min())/(max_val - min_val))
    max_val, min_val = frame2_img.max(), frame2_img.min()
    plt.imsave('frame2.png', (frame2_img - frame2_img.min())/(max_val - min_val))
    
    matching_matrix, row_index, col_index = maximum_bipartite_matching(im1_embedding, im2_embedding)
    print(row_index, col_index)

    mask1_index = mask_index1.squeeze()
    mask2 = torch.zeros((1, im1_embedding.shape[1], 1))
    mask2[:, col_index[mask1_index], :] = 1.0

    # Visualize mask
    rows, cols = image1[0].shape[1:]
    mask_2d = mask2.view(1, rows//region_size, cols//region_size)
    region_mask = torch.repeat_interleave(torch.repeat_interleave(mask_2d, region_size, dim=1), region_size, dim=2)
    plt.imsave('region_mask_tokenmatch.png', region_mask.squeeze().cpu().numpy())
    # run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)
    # output_dir = Path(config["_output"])
    # output_file = open(output_dir / "output.txt", 'a')
    # train_vitdet(config, ViTDet, train_data, val_data, output_file)
    # output_file.close()

if __name__ == "__main__":
    main()
