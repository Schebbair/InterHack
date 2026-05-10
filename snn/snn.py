import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNN(nn.Module):
    def __init__(self, num_inputs=5, num_hidden=64, num_outputs=4, beta=0.9):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        #x shape (time_steps, batch_size, num_inputs)
        
        #mem potential
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        #voltajes
        mem2_rec = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_rec.append(mem2)

        out_mem = torch.stack(mem2_rec, dim=0)[-1]
        
        return out_mem
    

    