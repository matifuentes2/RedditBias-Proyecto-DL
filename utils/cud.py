import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Print CUDA device count
    print(f"Using CUDA with {torch.cuda.device_count()} GPU(s)")
    # Print CUDA device name
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")