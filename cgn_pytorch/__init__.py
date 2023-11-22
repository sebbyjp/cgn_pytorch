from .contact_graspnet import CGN
import torch
import cgn_pytorch.util.config_utils as config_utils
from importlib.resources import files


def from_pretrained(cpu: bool = False, checkpoint_path: str = None) -> tuple[CGN, torch.optim.Adam, dict]:
    '''Loads a pretrained model and optimizer.

    Args:

      cpu (bool, optional): Whether to force use the cpu or not.
      checkpoint_path (str, optional): The path to the checkpoint file. If None,
       a pretrained model based on https://github.com/NVlabs/contact_graspnet
         will be loaded.

    Returns:
        tuple[CGN, torch.optim.Adam, dict]: CGN model, optimizer and config dict.
    '''
    print("initializing net")
    torch.cuda.empty_cache()
    config_dict = config_utils.load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not cpu else "cpu")
    model = CGN(config_dict, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if checkpoint_path is None:
        checkpoint_path = files("cgn_pytorch").joinpath("checkpoints/current.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer, config_dict


__all__ = ["CGN", "from_pretrained"]
