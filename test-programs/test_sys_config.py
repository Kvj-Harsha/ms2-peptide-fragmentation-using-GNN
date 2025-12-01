import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("CUDA version:", torch.version.cuda)
print("Device capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")
print("Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else "N/A")
