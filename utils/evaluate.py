from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones.base import dict_csv_header, dict_csv_line, dict_string
from backbones.policies import (
    TokenNormThreshold,
    TokenNormTopK,
    TokenNormTopFraction,
)
from utils.misc import (
    TopKAccuracy,
    get_device_description,
    get_pytorch_device,
    set_policies,
    tee_print,
)


def run_evaluations(config, model_class, data, evaluate_function):
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    # Load and set up the model.
    model = model_class(**(config["model"]))
    msg = model.load_state_dict(torch.load(config["weights"]), strict=False)
    print(msg)
    # Load model
    # from backbones.backbone_vit import interpolate_pos_embed
    # ckpt = torch.load(config["weights"])
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
    # msg = model.load_state_dict(ckpt, strict=False)
    # print(msg)

    # TODO: map backbone weights to timm ViT model
    model = model.to(device)

    completed = []
    output_dir = Path(config["_output"])

    def do_evaluation(title):
        with open(output_dir / "output.txt", "a") as tee_file:
            # Run the evaluation.
            model.eval()
            results = evaluate_function(device, model, data, config, tee_file)

            # Print and save results.
            tee_print(title, tee_file)
            tee_print(get_device_description(device), tee_file)
            if isinstance(results, dict):
                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
                for key, val in results.items():
                    tee_print(key.capitalize(), tee_file)
                    tee_print(dict_string(val), tee_file)
            else:
                tee_print(results, tee_file)
            tee_print("", tee_file)
            completed.append(title)

    # Evaluate the model.
    if config.get("vanilla", False):
        do_evaluation("Vanilla")
    for k in config.get("token_top_k", []):
        set_policies(model, TokenNormTopK, k=k)
        do_evaluation(f"Token top k={k}")
    for fraction in config.get("token_top_fraction", []):
        set_policies(model, TokenNormTopFraction, fraction=fraction)
        do_evaluation(f"Token top {fraction * 100:.1f}%")
    for threshold in config.get("token_thresholds", []):
        set_policies(model, TokenNormThreshold, threshold=threshold)
        do_evaluation(f"Token threshold {threshold}")


def save_csv_results(results, output_dir, first_run=False):
    for key, val in results.items():
        with open(output_dir / f"{key}.csv", "a") as csv_file:
            if first_run:
                print(dict_csv_header(val), file=csv_file)
            print(dict_csv_line(val), file=csv_file)
