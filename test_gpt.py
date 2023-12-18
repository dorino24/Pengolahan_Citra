import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Print the list of available GPUs
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Choose the GPU to use (e.g., choosing the first GPU)
    chosen_gpu = 0

    # Set the chosen GPU
    torch.cuda.set_device(chosen_gpu)

    # Print a message indicating the selected GPU
    print(f"Using GPU: {torch.cuda.get_device_name(chosen_gpu)}")
    device = torch.device("cuda")
else:
    # If CUDA is not available, use the CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
