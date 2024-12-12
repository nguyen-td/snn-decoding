import snntorch as snn

import torch
import torch.nn as nn

from pathlib import Path
import os

from modules import SNN, Trainer

# load input spike data
train_spikes = torch.load(Path('data') / 'train_spikes_gratings.pt', weights_only=True)
val_spikes = torch.load(Path('data') / 'val_spikes_gratings.pt', weights_only=True)

# load images
train_images = torch.load(Path('data') / 'train_images_gratings.pt', weights_only=True)
val_images = torch.load(Path('data') / 'val_images_gratings.pt', weights_only=True)

# set device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device (CPU or GPU): ', device)

n_inputs = train_spikes.shape[0] # train on all channels
n_pixels = train_images[0].shape[0] * train_images[0].shape[0]
n_hidden = 256

# hyperparameters
beta = 0.9 
tau_pre = 2
tau_post = 2
n_steps = 100
n_epochs = 30
# n_steps = 1
# n_epochs = 2
lr = 1e-2

model_save_dir = Path('model')

# set up spike, use spikes of a selected big image, concatenate the trials
spk_in = train_spikes.permute(1, 0, 2, 3)
print(spk_in.shape)

# train network
trainer = Trainer(n_inputs=n_inputs, 
                  n_hidden=n_hidden, 
                  n_pixels=n_pixels, 
                  beta=beta, 
                  n_steps=n_steps, 
                  n_epochs=n_epochs, 
                  device=device, 
                  tau_pre=tau_pre,
                  tau_post=tau_post, 
                  lr=lr,
                  model_save_dir=model_save_dir)
loss_hist_train, decoded_image, spk_rec, mem_rec, network = trainer.train(spk_in, train_images)

# validate network
spk_in = val_spikes.permute(1, 0, 2, 3)
loss, decoded_image_val, spk_rec_val, mem_rec_val = trainer.eval(network, spk_in, val_images)

# save outputs
if not os.path.exists(Path('outputs')):
    os.makedirs(Path('outputs'))

torch.save(loss_hist_train, Path('outputs') / 'loss_hist_train.pt')
torch.save(loss, Path('outputs') / 'loss_val.pt')
torch.save(decoded_image_val, Path('outputs') / 'decoded_image_val.pt')
torch.save(spk_rec_val, Path('outputs') / 'spk_rec_val.pt')
torch.save(mem_rec_val, Path('outputs') / 'mem_rec_val.pt')