import torch
import numpy as np
from losses.IW_SSIM_PyTorch import IW_SSIM as iwssim_class
from skimage.color import rgb2gray
from PIL import Image

PATCH_SIZE = 4
STRIDE = 11
RESIZE_SIZE = (256, 256)
TARGET_SHAPE = (256, 256, 3)

def resize_similarity_map(map, target_size):
    return torch.nn.functional.interpolate(map, size=target_size[:2], mode='bilinear', align_corners=False)

def iwssim_map(data, outputs, num_scales=2):  # Add a parameter for the number of scales
    batch_size = data.shape[0]
    anomaly_maps = []

    iw_ssim = iwssim_class()

    for i in range(batch_size):
        imgo = data[i]
        imgd = outputs[i]

        imgo_gray = rgb2gray(imgo)
        imgd_gray = rgb2gray(imgd)

        imgo = torch.tensor(imgo_gray, dtype=torch.float32)
        imgd = torch.from_numpy(imgd_gray)
        imgo = (imgo / 127.5) - 1
        imgd = (imgd / 127.5) - 1
        
        # Use PIL.Image.resize with a different interpolation method
        # Use PIL.Image.resize instead of skimage.transform.resize
        imgo_resized = Image.fromarray(imgo.numpy()).resize(RESIZE_SIZE, Image.LANCZOS) # Add .numpy() here
        imgd_resized = Image.fromarray(imgd.numpy()).resize(RESIZE_SIZE, Image.LANCZOS) # Add .numpy() here

        imgopr, imgdpr = iw_ssim.get_pyrd(np.array(imgo_resized), np.array(imgd_resized))
        iw_map = iw_ssim.info_content_weight_map(imgopr, imgdpr)

        # Use only the desired number of scales from iw_map
        selected_scales = sorted(iw_map.keys())[-num_scales:]
        resized_maps = {scale: resize_similarity_map(map, TARGET_SHAPE) for scale, map in iw_map.items() if scale in selected_scales}

        iw_map_tensors = [map.squeeze() for map in resized_maps.values()]

        similarity_map = torch.stack(iw_map_tensors, dim=-1)
        anomaly_score = 1 - similarity_map
        
        # Normalize similarity maps using softmax instead of min-max scaling
        # normalized_anomaly_score = torch.softmax(anomaly_score, dim=-1)
        # Normalize similarity maps between 0 and 1
        min_val = anomaly_score.min()
        max_val = anomaly_score.max()
        normalized_anomaly_score = (anomaly_score - min_val) / torch.clamp(max_val - min_val, min=0.0026)
        anomaly_maps.append(normalized_anomaly_score)

    # Convert the list of anomaly maps to a NumPy array
    anomaly_maps = torch.stack(anomaly_maps).detach().cpu().numpy()
    # print(anomaly_maps)
    # print('anomaly_maps', anomaly_maps.shape)
    return anomaly_maps









