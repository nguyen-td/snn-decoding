import snntorch as snn
from snntorch.functional.stdp_learner import STDPLearner
from snntorch import utils

import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
import warnings
from pathlib import Path
from more_itertools import chunked

class Trainer:
    '''
    Class consisting of methods to train a spiking neural network.

    Inputs:
    -------
        n_inputs: Scalar
            Number of input channels
        n_hidden: Scalar
            Number of hidden neurons (not really used in SNN-CNNs)
        n_pixels: Scalar
            Number of pixels of the target image, flattened to a vector
        beta: Scalar
            Beta value for STDP
        n_steps: Scalar
            Number of simulation steps
        n_epochs: Scalar
            Number of epochs
        device: String
            Either 'cpu' or 'cuda:0'
        model_save_dir: String
            Directory where the model should be saved at.
        net_name: String
            Name of the network to train, either 'SNN-CNN' or 'SNN'. Default is 'SNN-CNN'.
        tau_pre: Scalar
            Trace of the pre-synaptic neuron (cf. https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/stdp.html#stdp-spike-timing-dependent-plasticity). Default is 2.
        tau_post: Scalar
            Trace of the post-synaptic neuron (cf. https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/stdp.html#stdp-spike-timing-dependent-plasticity). Default is 2.
        lr: Scalar
            Learning rate for the Adam optimizer, default is 1e-4.
        mini_batch_size: Scalar
            Mini-batch size for training, default is 32.

    '''
    def __init__(self, n_inputs, n_hidden, n_pixels, beta, n_steps, n_epochs, device, model_save_dir, net_name='SNN-CNN', tau_pre=2, tau_post=2, lr=1e-4, mini_batch_size=32):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_pixels = n_pixels
        self.beta = beta
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        
        self.model_save_dir = model_save_dir
        self.net_name = net_name
        self.device = device

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _choose_network(self):
        '''
        Choose which network to train and define or call it.

        Output:
        -------
            network: torch.nn class
                Neural network class definition
        '''

        if self.net_name == 'SNN-CNN':
            network = nn.Sequential(OrderedDict([
                # ('lgn', nn.Conv2d(in_channels=self.n_inputs, out_channels=16, kernel_size=3, stride=1, padding=1)),
                # ('lif1', snn.Leaky(beta=self.beta, init_hidden=True)),
                ('v1_simple', nn.Conv2d(in_channels=self.n_inputs, out_channels=16, kernel_size=5, stride=1, padding=1)),
                ('lif2', snn.Leaky(beta=self.beta, init_hidden=True)),
                ('v1_complex', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('pool', nn.AdaptiveMaxPool2d(1)),
                ('flat', nn.Flatten()),
                ('fc1', nn.Linear(16, self.n_pixels)),
                ('lif3', snn.Leaky(beta=self.beta, init_hidden=True, output=True))
            ]))
            network = network.to(self.device)
            network.apply(self._init_weights)

            # initialize STDPLearner for each layer
            # self.stdp_lgn = STDPLearner(synapse=network.lgn, sn=network.lif1, tau_pre=self.tau_pre, tau_post=self.tau_post)
            self.stdp_v1 = STDPLearner(synapse=network.v1_simple, sn=network.lif2, tau_pre=self.tau_pre, tau_post=self.tau_pre)

            # enable STDP for all layers
            # self.stdp_lgn.enable()
            self.stdp_v1.enable()

            # only store gradients for the last layer
            for name, param in network.named_parameters():
                if 'fc1' not in name:  # exclude last layer
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # keep last layer trainable
        else:
            warnings.warn("The 'net_name' parameter only accepts 'SNN-CNN' so far.")

        return network
    
    def _forward(self, network, data):
        '''
        Forward pass through the network.

        Inputs:
        -------
            network: torch.nn class
                Neural network
            n_steps: Scalar
                Number of steps
            data: (n_batch, n_channels, n_trials, n_time) Torch tensor
                Input data in the correct shape

        Outputs:
        --------
            spk_rec: (n_steps, n_batch, n_pixels) Torch tensor
                Spike outputs
            mem_rec: (n_steps, n_batch, n_pixels) Torch tensor
                Membrane output current
        '''
        
        mem_rec = []
        spk_rec = []
        utils.reset(network)  # resets hidden states for all LIF neurons in network

        for step in range(self.n_steps):
            spk_out, mem_out = network(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
            
        return torch.stack(spk_rec), torch.stack(mem_rec)

    def train(self, spk_in, target):
        '''
        Simulate the selected network.

        Inputs:
        ------
            spk_in: (n_batch, n_channels, n_trials, n_time) Torch tensor
                Spiking input of the corresponding image.
            target: (n_batch, n_pix, n_pix) Torch tensor
                Target image, where n_pixels = n_pix x n_pix

        Outputs:
        --------
            loss_hist: (n_epochs, ) Torch tensor
                Loss over epochs
            decoded_image: (n_pix, n_pix) Torch tensor
                Decoeded image
            spk_rec: (n_steps, n_batch, n_pixels) Torch tensor
                Spiking output over steps of the last layer. Used to decode the image, where each neuron codes for one pixel using rate coding.
            mem_rec: (n_steps, n_batch, n_pixels) Torch tensor
                Membrane potential over steps of the last layer.
            network: torch.nn class
                Neural network
        '''

        # initialize network
        network = self._choose_network()
        print(network)
        print()

        if self.net_name == 'SNN-CNN':
            # gradient descent on last layer
            optimizer = torch.optim.SGD(network.fc1.parameters(), lr=self.lr)
            mse_loss = torch.nn.MSELoss()

            loss_hist = torch.zeros(self.n_epochs)
            utils.reset(network)  # resets hidden states for all LIF neurons in network

            # make mini-batches
            train_batch = list((chunked(spk_in, self.mini_batch_size)))
            target_batch = list((chunked(target, self.mini_batch_size)))
            n_batches = len(train_batch)

            # STDP on all layers but the last FC
            for epoch in range(self.n_epochs):
                print(f'Epoch {epoch}')

                train_loss = 0
                for batch in range(n_batches):
                    optimizer.zero_grad()  
                    # data = spk_in.to(self.device)
                    data = torch.as_tensor(np.array(train_batch[batch]), device=self.device)
                    target = torch.as_tensor(np.array(target_batch[batch]), device=self.device)
                    # data = spk_in.unsqueeze(0).to(self.device)

                    # skip last batch if minibatch size does not match
                    # TODO: change later and add proper oversampling
                    if not data.shape[0] == self.mini_batch_size:
                        continue

                    # forward pass
                    network.train()
                    spk_rec, mem_rec = self._forward(network, data)

                    # apply STDP updates (on weights only, no gradients involved)
                    # self.stdp_lgn.step(on_grad=True)
                    self.stdp_v1.step(on_grad=True)

                    # clamp weights to prevent instability after STDP update
                    with torch.no_grad():
                        # network.lgn.weight.data.clamp_(-1.0, 1.0)
                        network.v1_simple.weight.data.clamp_(-1.0, 1.0)

                    # decode image using rate code, i.e., each output neuron codes for a pixel
                    decoded_image = (spk_rec.sum(dim=0).reshape(target.shape) / self.n_steps)  # firing rates normalized

                    # compute loss
                    loss = mse_loss(decoded_image.float(), target.float()) 
                    train_loss += loss.item()

                    # backpropagation for classification layer
                    loss.backward(retain_graph=True)  
                    optimizer.step()  

            loss_hist[epoch] = train_loss / n_batches
            
            # disable STDPLearner after training
            # self.stdp_lgn.disable()
            self.stdp_v1.disable()

        # save trained model
        torch.save(network.state_dict(), Path(self.model_save_dir) / 'snn_gratings.pt')

        return loss_hist, decoded_image, spk_rec, mem_rec, network
    
    def eval(self, network, spk_in, target):
        '''
        Evaluate the selected network (do not compute gradients).

        Inputs:
        ------
            network: torch.nn class
                Neural network
            spk_in: (n_channels, n_trials, n_time) Torch tensor
                Spiking input of the corresponding image.
            target: (n_pix, n_pix) Torch tensor
                Target image, where n_pixels = n_pix x n_pix

        Outputs:
        --------
            loss: Torch tensor
                Validation loss
            decoded_image: (n_pix, n_pix) Torch tensor
                Decoeded image
            spk_rec: (n_steps, n_batch, n_pixels) Torch tensor
                Spiking output over steps of the last layer. Used to decode the image, where each neuron codes for one pixel using rate coding.
            mem_rec: (n_steps, n_batch, n_pixels) Torch tensor
                Membrane potential over steps of the last layer.
        '''

        if self.net_name == 'SNN-CNN':
            mse_loss = torch.nn.MSELoss()

            with torch.no_grad():
                data = spk_in.to(self.device)

                # forward pass
                network.eval()
                spk_rec, mem_rec = self._forward(network, data)

                # decode image using rate code, i.e., each output neuron codes for a pixel
                decoded_image = (spk_rec.sum(dim=0).reshape(target.shape) / self.n_steps)  # firing rates normalized

                # compute loss
                loss = mse_loss(decoded_image.float(), target.float()) 

        return loss.item(), decoded_image, spk_rec, mem_rec


