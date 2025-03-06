import torch
import torch.nn as nn
from model.network import Net
import os

def print_model_input_size(model_path, model_class=None, model_args={}):
    """
    Load a .pth checkpoint and print its input size.

    If the checkpoint is a dictionary and contains an "input_size" key,
    that value will be printed.

    If the checkpoint is a state_dict only (or a checkpoint dict with extra keys)
    and model_class is provided, the function will instantiate the model and load
    the state_dict. It then checks if the model has an "input_size" attribute,
    or attempts to infer the expected number of input channels from the first Conv2d layer.

    Parameters:
        model_path (str): Path to the .pth file.
        model_class (class, optional): The model class to instantiate if needed.
        model_args (dict, optional): Keyword arguments for model instantiation.
    """
    # Use GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Case 1: Check if checkpoint has metadata for input size.
    if isinstance(checkpoint, dict) and "input_size" in checkpoint:
        print("Model input size (from checkpoint metadata):", checkpoint["input_size"])
        return

    # Case 2: If checkpoint is a dict but no explicit input_size metadata,
    # then check if it's a checkpoint dict with extra keys.
    if isinstance(checkpoint, dict):
        # If the checkpoint dict has a "state_dict" key, extract it.
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
        print(f"state :{state}")
        if model_class is None:
            print("Checkpoint is a state_dict and no model_class was provided.")
            return

        model = model_class(**model_args)
        model.load_state_dict(state)
    else:
        # Case 3: The checkpoint is a complete model object.
        model = checkpoint

    # Move the model to GPU (if available).
    model.to(device)

    # Try to print an "input_size" attribute if available.
    if hasattr(model, "input_size"):
        print("Model input size (from model attribute):", model.input_size)
        return

    # Otherwise, attempt to infer expected input channels by checking the first Conv2d layer.
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            print("Model appears to expect input with", module.in_channels, "channels.")
            return

    print("Could not infer input size from the model or checkpoint.")

# Example usage:
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "results", "vis_line", "model_best.pth")
    model_args = {"numAngle": 320, "numRho": 320, "backbone": "resnet50"}
    print_model_input_size("results/vis_line/model_best.pth", model_class=Net, model_args=model_args)

    # Example 2: If your .pth file is a state_dict and you need to instantiate the model.
    # from model.network import Net  # Replace with your model's import.
    # model_args = {"numAngle": 320, "numRho": 320, "backbone": "resnet50"}
    # print_model_input_size(model_path, model_class=Net, model_args=model_args)

