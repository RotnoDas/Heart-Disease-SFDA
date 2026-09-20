import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(
            "Using CUDA GPU:",
            torch.cuda.get_device_name(0)
        )

        return device

    print("CUDA unavailable. Using CPU.")

    return torch.device("cpu")