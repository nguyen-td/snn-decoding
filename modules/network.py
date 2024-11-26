import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Network
class SNN(nn.Module):
    def __init__(self, n_inputs, beta, n_steps, n_pixels):
        super().__init__()

        self.n_inputs = n_inputs
        self.beta = beta
        self.n_steps = n_steps
        self.n_pixels = n_pixels

        # initialize layers
        self.lgn = nn.Conv2d(in_channels=n_inputs, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=self.beta)

        self.v1_simple = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.v1_complex = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif2 = snn.Leaky(beta=self.beta)

        self.v2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=7, stride=1, padding=3)
        self.v2_complex = nn.AdaptiveMaxPool2d((1)) 
        self.lif3 = snn.Leaky(beta=self.beta)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(8, n_pixels)
        self.lif4 = snn.Leaky(beta=self.beta)

    def forward(self, x):

        # initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # record the final layer
        spk_rec = []
        mem_rec = []

        batch_size = x.shape[0]

        for step in range(self.n_steps):
            cur1 = self.lgn(x)  # LGN layer
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.v1_complex(self.v1_simple(spk1)) # V1 layer
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.v2_complex(self.v2(spk2)) # V2 layer
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc1(self.flat(spk3.view(batch_size, 8, -1))) # decoding layer
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_rec.append(spk4)
            mem_rec.append(mem4)    

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)