from voicebot.vmcheck import check_gpu, check_disk
HEADER = "| NVIDIA-SMI 580.178   Driver Version: 580.178   CUDA Version: 13.0 |"

def test_mismatch_says_reboot():                                         # Review Focus 2
    p = check_gpu(18, "Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 580.178", "")
    assert p == ["NVIDIA driver/library version mismatch: reboot the VM (sudo reboot), then rerun."]

def test_healthy_5090(): assert check_gpu(0, "NVIDIA GeForce RTX 5090, 580.178", HEADER) == []

def test_old_driver_cuda_and_wrong_gpu():
    p = check_gpu(0, "NVIDIA GeForce RTX 4060, 550.54", HEADER.replace("13.0", "12.4"))
    assert any("older than 570" in x for x in p) and any("older than 12.8" in x for x in p) and any("RTX 5090" in x for x in p)

def test_disk(): assert check_disk(44 * 1024**3) and not check_disk(46 * 1024**3)
