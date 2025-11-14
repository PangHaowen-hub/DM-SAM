import os
import logging
import time
from DM.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


if __name__ == '__main__':
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    now_time_path = os.path.join('experiments', now_time)
    os.makedirs(now_time_path, exist_ok=True)
    logger = logging.getLogger()
    logfile = '{}.log'.format(now_time)
    logfile = os.path.join(now_time_path, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

    model = Unet(channels=4)

    diffusion = GaussianDiffusion(
        model,
        image_size=512,
        timesteps=1000,
        sampling_timesteps=10,
        objective='pred_v',
    )

    trainer = Trainer(
        diffusion,
        data_path='data_npz',
        source_modality='T1N',
        target_modality='T2W',
        train_batch_size=16,
        train_lr=8e-5,
        train_num_steps=200000,
        save_and_sample_every=10000,
        num_samples=16,
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=False,  # whether to calculate fid during training
        results_folder=os.path.join(now_time_path, 'results'),
    )
    trainer.train()
