from DM.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch
import numpy as np


if __name__ == '__main__':
    pth_path = r'./experiments/yyyy-mm-dd-hh-mm-ss/results/model-20.pt'
    save_path = pth_path[:-3]
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(channels=4)

    diffusion = GaussianDiffusion(
        model,
        image_size=512,
        timesteps=1000,
        sampling_timesteps=10,
        objective='pred_v',
    ).to(device)

    datstate_dict = torch.load(pth_path, map_location=device, weights_only=True)
    diffusion.load_state_dict(datstate_dict['model'])

    ds_test = Dataset(data_path=r'./data_npz', source_modality='T1N', target_modality='T2W', image_size=512)

    repeat_num = 8
    batch_size = 1
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    with torch.inference_mode():
        for cond_img, _, patient, fname in tqdm(dl_test):
            cond_img = cond_img.to(device)
            cond_img = cond_img.repeat(repeat_num, 1, 1, 1)
            predict_images = diffusion.sample(cond_img, batch_size=repeat_num * batch_size).cpu()
            for i in range(batch_size):
                save_predict_images = predict_images[i*batch_size:(i+1)*batch_size]
                predict_image_avg = torch.mean(save_predict_images, dim=0)
                predict_image_std = torch.std(save_predict_images, dim=0)
                predict_image_avg[predict_image_avg < 0] = 0
                predict_image_avg[predict_image_avg > 1] = 1
                npz_save_path = os.path.join(save_path, patient[i], fname[i])
                os.makedirs(os.path.dirname(npz_save_path), exist_ok=True)
                np.savez_compressed(
                    npz_save_path,
                    avg=predict_image_avg.squeeze().cpu().numpy(),
                    std=predict_image_std.squeeze().cpu().numpy()
                )
