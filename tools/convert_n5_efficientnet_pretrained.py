import argparse
import os

import torch

from models import EfficientNetB0


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def convert_shared_backbone_to_three_planes(state_dict, model):
    model_state = model.state_dict()
    converted = {}

    for key, value in state_dict.items():
        if key.startswith("backbone."):
            suffix = key[len("backbone."):]
            for plane in ("axial", "coronal", "sagittal"):
                target_key = f"{plane}.{suffix}"
                if target_key in model_state and hasattr(value, "shape") and model_state[target_key].shape == value.shape:
                    converted[target_key] = value
            continue

        if key in model_state and hasattr(value, "shape") and model_state[key].shape == value.shape:
            converted[key] = value

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert N5 EfficientNet-B0 pretrained checkpoint to this repo format.")
    parser.add_argument("--input", type=str, required=True, help="Path to N5 checkpoint (.pth)")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path (.pth)")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state_dict = strip_module_prefix(state_dict)

    model = EfficientNetB0()
    converted = convert_shared_backbone_to_three_planes(state_dict, model)

    if not converted:
        raise RuntimeError("No compatible tensors were converted. Check input checkpoint architecture.")

    missing, unexpected = model.load_state_dict(converted, strict=False)
    print(f"Converted tensors: {len(converted)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": converted,
            "source_checkpoint": args.input,
            "format": "efficientnetb0_three_planes_partial",
        },
        args.output,
    )
    print(f"Saved converted checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
