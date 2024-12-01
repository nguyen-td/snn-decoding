import snntorch as snn

import torch
import torch.nn as nn

import scipy.io
from pathlib import Path
import os

from modules import SNN, Trainer

# load input spike data
mat = scipy.io.loadmat(Path('data') / '01.mat') # animal 01
spike_train_all = mat['resp_train'] # spike train of all neurons, neurons x image x trials x milliseconds
print(spike_train_all.shape)

# get indices of all small natural images
idx_small_nat_images = torch.zeros(spike_train_all.shape[1])
idx_small_nat_images[:539:2] = 1

# get indices of all big natural images
idx_big_nat_images = torch.ones(spike_train_all.shape[1])
idx_big_nat_images[:539:2] = 0
idx_big_nat_images[540:] = 0

# get indices of all gratings
idx_gratings = torch.zeros(spike_train_all.shape[1])
idx_gratings[540:] = 1

# only keep well-centered channels
indcent = mat['INDCENT'].squeeze() # indicates if an stimulus was centered inside the neuron's RF and if the spikes were sorted
spike_train_cent = torch.tensor(spike_train_all[indcent == 1]).float()
spike_train_cent.shape

# set device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device (CPU or GPU): ', device)

# layer parameters
images_all = mat['images'].squeeze()

n_inputs = spike_train_cent.shape[0] # train on all channels
n_pixels = images_all[0].shape[0] * images_all[0].shape[0]
n_hidden = 256

# hyperparameters
beta = 0.9 
tau_pre = 2
tau_post = 2
n_steps = 200
n_epochs = 5000
lr = 1e-2

# set up spike, use spikes of a selected big image, concatenate the trials
spk_in = spike_train_cent[:, 600, :, :].squeeze()
print(spk_in.shape)

target = torch.tensor(images_all[600])

# train network
# TODO: Find better intialization scheme
trainer = Trainer(n_inputs=n_inputs, 
                  n_hidden=n_hidden, 
                  n_pixels=n_pixels, 
                  beta=beta, 
                  n_steps=n_steps, 
                  n_epochs=n_epochs, 
                  device=device, 
                  tau_pre=tau_pre,
                  tau_post=tau_post, 
                  lr=lr)
loss_hist, decoded_image, spk_rec, mem_rec = trainer.train(spk_in, target)

# save outputs
if not os.path.exists(Path('outputs')):
    os.makedirs(Path('outputs'))

torch.save(loss_hist, Path('outputs') / 'loss_hist.pt')
torch.save(decoded_image, Path('outputs') / 'decoded_image.pt')
torch.save(spk_rec, Path('outputs') / 'spk_rec.pt')
torch.save(mem_rec, Path('outputs') / 'mem_rec.pt')