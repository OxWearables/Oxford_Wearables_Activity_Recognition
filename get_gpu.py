""" Script to get a free MIG instance on an NVIDIA A100 GPU """
import os
import pynvml


def get_free_mig_uuid():

    # Initialize NVML
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    free_mig_uuid = None

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        # Check if MIG mode is enabled
        is_mig_enabled = pynvml.nvmlDeviceGetMigMode(handle)[0]
        if is_mig_enabled:
            # Get the number of MIG instances
            mig_instance_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)

            for mig_index in range(mig_instance_count):
                try:
                    # Get the MIG instance handle and UUID
                    mig_device = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(
                        handle, mig_index)
                    mig_uuid = pynvml.nvmlDeviceGetUUID(mig_device)

                    # Get the list of processes on this MIG instance
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(
                        mig_device)

                    if len(processes) == 0:
                        # If there are no processes, the MIG instance is free
                        free_mig_uuid = mig_uuid
                        print(f"Found free MIG instance: UUID = {mig_uuid}")
                        break
                except pynvml.NVMLError_NotFound:
                    # No more MIG instances
                    continue

        if free_mig_uuid:
            break

    pynvml.nvmlShutdown()  # Shutdown NVML after use
    return free_mig_uuid


def get_gpu():

    # Get a free MIG instance UUID
    free_mig_uuid = get_free_mig_uuid()

    if free_mig_uuid:
        # Set CUDA_VISIBLE_DEVICES to the free MIG instance UUID
        os.environ['CUDA_VISIBLE_DEVICES'] = free_mig_uuid
        print(f"Using MIG instance with UUID: {free_mig_uuid}")
    else:
        print("No free MIG instance found. Falling back to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # Apparently, we need to set the environ BEFORE importing torch
    import torch

    # Check if CUDA is available and set the device accordingly
    if torch.cuda.is_available() and os.environ['CUDA_VISIBLE_DEVICES']:
        device = torch.device("cuda")
        print(f"Using {device} device")
    else:
        device = torch.device("cpu")
        print(f"Using {device} device")

    return device


if __name__ == '__main__':
    get_gpu()
