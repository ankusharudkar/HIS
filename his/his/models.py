from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
    def grad_norm(self):
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm
    
class ConvRNNCell(Model):
    def __init__(self, in_channels, height, width, cell_unit, out_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.out_channels = out_channels

        # hidden state
        # self.hidden = torch.randn(in_channels, height, width)
        self.register_buffer('hidden', torch.ones(1, in_channels, height, width))
        self.hidden = self.hidden.to("cuda")
        
        # output generation
        self.Wya = cell_unit(in_channels, out_channels).to("cuda")
        
        # hidden state processing  
        self.Waa = cell_unit(in_channels, in_channels).to("cuda")
        self.Wax = cell_unit(in_channels, in_channels).to("cuda")
                
    def forward(self, x):
        # calculate state
        hidden_batched = self.hidden.expand(x.shape[0], -1, -1, -1)  
        # hidden_batched = self.hidden
        
        self.hidden = nn.functional.relu(self.Waa(hidden_batched) + self.Wax(x))
       
        # calculate output
        x = nn.functional.relu(self.Wya(self.hidden))
        
        return x
    
    def resetState(self):
        self.hidden = torch.ones(1, self.in_channels, self.height, self.width).to("cuda")
        
class ConvLSTMCell(Model):
    def __init__(self, in_channels, height, width, cell_unit, out_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.out_channels = out_channels

        # hidden and cell states
        # self.hidden = torch.randn(in_channels, height, width)
        self.register_buffer('hidden_state', torch.ones(1, in_channels, height, width))
        self.register_buffer('cell_state', torch.ones(1, in_channels, height, width))
        self.hidden_state = self.hidden_state.to("cuda")
        self.cell_state = self.cell_state.to("cuda")
        
        # forget gate
        self.Wf = cell_unit(in_channels, out_channels)
        self.Uf = cell_unit(in_channels, out_channels)
        
        # input gate
        self.Wi = cell_unit(in_channels, out_channels)
        self.Ui = cell_unit(in_channels, out_channels)
        
        # output gate
        self.Wo = cell_unit(in_channels, out_channels)
        self.Uo = cell_unit(in_channels, out_channels)
        
        # cell state
        self.Wc = cell_unit(in_channels, out_channels)
        self.Uc = cell_unit(in_channels, out_channels)
        
        # activations
        self.sig1 = nn.Tanh()
        self.sig2 = nn.Tanh()
        self.sig3 = nn.Tanh()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
               
        
    def forward(self, x):
        # forget gate
        f_t = self.sig1(self.Wf(x) + self.Uf(self.hidden_state))
        
        # input gate
        i_t = self.sig2(self.Wi(x) + self.Ui(self.hidden_state))
        
        # output gate
        o_t = self.sig3(self.Wo(x) + self.Uo(self.hidden_state))
        
        # cell state
        c_t_prime =self.relu1(self.Wc(x) + self.Uc(self.hidden_state))
        self.cell_state = f_t * self.cell_state + i_t * c_t_prime
        
        # hidden state
        self.hidden_state = o_t * self.relu2(self.cell_state)
        
        return o_t
    
    def resetState(self):
        self.hidden_state = torch.ones(1, self.in_channels, self.height, self.width).to("cuda")
        self.cell_state = torch.ones(1, self.in_channels, self.height, self.width).to("cuda")
        

class ConvGRUCell(Model):
    def __init__(self, in_channels, height, width, cell_unit, out_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.out_channels = out_channels

        # hidden and cell states
        # self.hidden = torch.randn(in_channels, height, width)
        self.register_buffer('hidden_state', torch.ones(1, in_channels, height, width))
        self.hidden_state = self.hidden_state.to("cuda")
        
        # update gate
        self.Wz = cell_unit(in_channels, out_channels)
        self.Uz = cell_unit(in_channels, out_channels)
        
        # reset gate
        self.Wr = cell_unit(in_channels, out_channels)
        self.Ur = cell_unit(in_channels, out_channels)
        

        # output generation
        self.Wo = cell_unit(in_channels, out_channels)
        self.Uo = cell_unit(in_channels, out_channels)
    
               
        self.Wf = cell_unit(in_channels, out_channels)
        
    def forward(self, x):
        # update gate
        u = nn.functional.sigmoid(self.Wz(x) + self.Uz(self.hidden_state))
        
        # input gate
        r = nn.functional.sigmoid(self.Wr(x) + self.Ur(self.hidden_state))
        
        c_t_prime = nn.functional.relu(self.Wo(x) + self.Uo(r*self.hidden_state))
        
        # hidden state
        self.hidden_state = u * c_t_prime + (1-u) * self.hidden_state
        
        # output
        output = self.Wf(self.hidden_state)
    
        return output
    
    def resetState(self):
        self.hidden_state = torch.ones(1, self.in_channels, self.height, self.width).to("cuda")