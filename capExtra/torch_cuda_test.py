import torch

def check_available_devices():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f'GPU is available with {device_count} device(s).')
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f'Device {i}: {device_name}')
    else:
        print('No GPU available, using CPU.')
        print(f'Current device: {torch.device("cpu")}')

if __name__ == "__main__":
    check_available_devices()
