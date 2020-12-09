import os

def getFreeId(minRatio=70):
    """
    Get free indices of the GPUs which have free ratio > minRatio
    :param minRatio: minimal ratio of the GPU utilization to be called free GPU
    :return: gpu indices string, e.g. "4,5,6,7"
    """
    import pynvml
    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)>minRatio:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    """
    set GPUs by indices
    :param gpuinput: e.g. "1,4,5", if not specified, all free gpus will be used.
    :return: number of GPU being set

    example:
        n_gpu = setgpu("5, 6, 7")
    """
    minRatio = 70
    freeids = getFreeId(minRatio)
    if gpuinput=="all":
        gpus = freeids
    else:
        gpus = gpuinput.replace(" ", "")
        gpu_in_use = [g for g in gpus.split(",") if g not in freeids]
        if len(gpu_in_use) > 0:
            raise ValueError("gpu{:s} is being used! Availabel gpus: gpu {:s}".format(",".join(gpu_in_use), freeids))
    print("using gpu "+gpus)
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus
    return len(gpus.split(","))