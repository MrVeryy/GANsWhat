import torch

def list_available_devices(verbose: bool = True):
    """
    Return a list of available torch devices in priority order.
    Example: ["cuda:0", "cuda:1", "mps", "cpu"]
    """
    devices = ["cpu"]

    # Apple Silicon MPS
    if torch.backends.mps.is_available():
        devices.append("mps")

    # NVIDIA CUDA GPUs
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    if verbose:
        print("Available devices:", devices)
    return devices


def get_recommended_device() -> torch.device:
    """
    Return the best available device:
    1. CUDA GPU (cuda:0)
    2. Apple MPS
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
