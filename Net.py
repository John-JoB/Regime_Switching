from typing import Callable
import torch as pt
from dpf.model import *
from numpy import sqrt
from dpf.utils import nd_select, normalise_log_quantity, batched_select         

class Markov_Switching(pt.nn.Module):
    def __init__(self, n_models:int, switching_diag: float, switching_diag_1: float, dyn = 'Boot', device:str ='cuda'):
        super().__init__()
        self.device=device
        self.dyn = dyn
        self.n_models = n_models
        tprobs = pt.ones(n_models) * ((1 - switching_diag - switching_diag_1)/(n_models - 2))
        tprobs[0] = switching_diag
        tprobs[1] = switching_diag_1
        self.switching_vec = pt.log(tprobs).to(device=device)
        self.dyn = dyn
        if dyn == 'Uni' or dyn == 'Deter':
            self.probs = pt.ones(n_models)/n_models
        else:
            self.probs = tprobs

    def init_state(self, batches, n_samples):
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=self.device).tile((batches, n_samples//self.n_models)).unsqueeze(2)
        return pt.multinomial(pt.ones(self.n_models), batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)

    def forward(self, x_t_1, t):
        if self.dyn == 'Deter':
            return pt.arange(self.n_models, device=self.device).tile((x_t_1.size(0), x_t_1.size(1)//self.n_models)).unsqueeze(2) 
        shifts = pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0),x_t_1.size(1)])
        new_models = pt.remainder(shifts + x_t_1[:, :, 1], self.n_models)
        return new_models.unsqueeze(2)
    
    def get_log_probs(self, x_t, x_t_1):
        shifts = (x_t[:,:,1] - x_t_1[:,:,1])
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return self.switching_vec[shifts]


class Polya_Switching(pt.nn.Module):
    def __init__(self, n_models, dyn, device:str='cuda') -> None:
        super().__init__()
        self.device = device
        self.dyn = dyn
        self.n_models = n_models
        self.ones_vec = pt.ones(n_models)
        
    def init_state(self, batches, n_samples):
        self.scatter_v = pt.zeros((batches, n_samples, self.n_models), device=self.device)
        i_models = pt.multinomial(self.ones_vec, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        return pt.concat((i_models, pt.ones((batches, n_samples, self.n_models), device=self.device)), dim=2)

    def forward(self, x_t_1, t):
        self.scatter_v.zero_()
        self.scatter_v.scatter_(2, x_t_1[:,:,1].unsqueeze(2).to(int), 1)
        c = x_t_1[:,:,2:] + self.scatter_v
        if self.dyn == 'Uni':
            return pt.concat((pt.multinomial(self.ones_vec,  x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0),x_t_1.size(1), 1]), c), dim=2)
        return pt.concat((pt.multinomial(c.reshape(-1, self.n_models), 1, True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        probs = x_t[:, :, 2:]
        probs /= pt.sum(probs, dim=2, keepdim=True)
        s_probs = batched_select(probs, x_t_1[:, :, 1].to(int))
        return pt.log(s_probs)

class NN_Switching(pt.nn.Module):

    def __init__(self, n_models, recurrent_length, dyn, device):
        super().__init__()
        self.device = device
        self.r_length = recurrent_length
        self.n_models = n_models
        self.forget = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.self_forget = pt.nn.Sequential(pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Sigmoid())
        self.scale = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid())
        self.to_reccurrent = pt.nn.Sequential(pt.nn.Linear(n_models, recurrent_length), pt.nn.Tanh())
        self.output_layer = pt.nn.Sequential(pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Tanh(), pt.nn.Linear(recurrent_length, n_models))
        self.probs = pt.ones(n_models)/n_models
        self.dyn = dyn

    def init_state(self, batches, n_samples):
        i_models = pt.multinomial(self.probs, batches*n_samples, True).reshape((batches, n_samples, 1)).to(device=self.device)
        if self.r_length > 0:
            return pt.concat((i_models, pt.zeros((batches, n_samples, self.r_length), device=self.device)), dim=2)
        else:
            return i_models

    def forward(self, x_t_1, t):
        old_model = x_t_1[:, :, 1].to(int).unsqueeze(2)
        one_hot = pt.zeros((old_model.size(0), old_model.size(1), self.n_models), device=self.device)
        one_hot = pt.scatter(one_hot, 2, old_model, 1)
        old_recurrent = x_t_1[:, :, 2:]
        c = old_recurrent * self.self_forget(old_recurrent)
        c *= self.forget(one_hot)
        c += self.scale(one_hot) * self.to_reccurrent(one_hot)
        if self.dyn == 'Boot':
            probs = pt.abs(self.output_layer(c))
            probs = probs / pt.sum(probs, dim=2, keepdim=True)
            return pt.concat((pt.multinomial(probs.reshape(-1, self.n_models), 1, True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
        return pt.concat((pt.multinomial(self.probs, x_t_1.size(0)*x_t_1.size(1), True).to(self.device).reshape([x_t_1.size(0), x_t_1.size(1), 1]), c), dim=2)
    
    def get_log_probs(self, x_t, x_t_1):
        models = x_t[:,:,1].to(int)
        probs = pt.abs(self.output_layer(x_t[:, :, 2:])) + 1e-7
        probs = probs / pt.sum(probs, dim=2, keepdim=True)
        log_probs = batched_select(probs.reshape(-1, self.n_models), models.flatten()).reshape(x_t.size(0), x_t.size(1))
        return pt.log(log_probs+1e-7)
    
class Likelihood_NN(pt.nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.net = pt.nn.Sequential(pt.nn.Linear(input, hidden), pt.nn.Tanh(), pt.nn.Linear(hidden, hidden), pt.nn.Tanh(), pt.nn.Linear(hidden, output))

    def forward(self, in_vec):
        return self.net(in_vec.unsqueeze(1)).squeeze()


class Simple_NN(pt.nn.Module):
    def __init__(self, input, hidden, output, layers):
        super().__init__()
        nn_layers = [pt.nn.Linear(input, hidden), pt.nn.Tanh()]
        for i in range(layers-2):
            nn_layers += [pt.nn.Linear(hidden, hidden), pt.nn.Tanh()]
        nn_layers += [pt.nn.Linear(hidden, output)]
        self.net = pt.nn.Sequential(*tuple(nn_layers))

    def forward(self, in_vec):
        return self.net(in_vec.unsqueeze(1)).squeeze()

class PF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, a:list[int], b:list[int], var_s:float, switching_dyn:pt.nn.Module, dyn ='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.tensor(a, device = device)
        self.b = pt.tensor(b, device = device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.var_factor = -1/(2*var_s)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches:int, n_samples: int):
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)
                                      
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device)
        new_models = self.switching_dyn(x_t_1, t)
        index = new_models[:,:,0].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = ((scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        locs = (scaling*pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias)

        return self.var_factor * ((self.y[t] - locs)**2)
    
    def observation_generation(self, x_t):
        noise = self.y_dist.sample((x_t.size(0), 1)).to(device=self.device)
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = ((scaling * pt.sqrt(pt.abs(x_t[:, :, 0])) + bias).unsqueeze(2) + noise)
        return new_pos


class RSDBPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, init_scale = 1, layers=2, hidden_size = 8, dyn='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        self.switching_dyn = switching_dyn
        for p in self.parameters():
            p =  p * init_scale
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.pi_fact = (1/2)* pt.log(pt.tensor(2*pt.pi))
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2) + 1e-6)
        self.pre_factor = -(1/2)*pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device) * self.sd_d
        new_models = self.switching_dyn(x_t_1, t)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=self.device)
        index = new_models[:, :, 0].to(int)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.dyn_models[m](x_t_1[:,:,0][mask])
        new_pos = (locs.unsqueeze(2) + noise)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2) + self.pre_factor
    

class DBPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, init_scale = 1, layers = 2, hidden_size = 5, device:str = 'cuda'):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_model = Simple_NN(1, hidden_size, 1, layers)
        self.obs_model = Simple_NN(1, hidden_size, 1, layers)
        for p in self.parameters():
            p  = p*init_scale
        self.sd_d = pt.nn.Parameter(pt.rand(1, device=self.device)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1, device=self.device)*0.4 + 0.1)
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.alg = self.PF_Type.Bootstrap

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        return init_locs           
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device) * self.sd_d
        locs = self.dyn_model(x_t_1)
        new_pos = locs.unsqueeze(2) + noise
        return new_pos
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return pt.zeros([x_t.size(0), x_t.size(1)], device=self.device)

    def log_f_t(self, x_t, t: int):
        locs = self.obs_model(x_t)
        return self.var_factor * (self.y[t] - locs)**2

class MAPF(SSM):
    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, init_scale = 1, layers = 2, hidden_size = 8, device:str = 'cuda'):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)])
        for p in self.parameters():
            p  = p*init_scale
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.alg = self.PF_Type.Bootstrap

    def M_0_proposal(self, batches:int, n_samples: int):
        self.var_factor = -1/(2*(self.sd_o**2))
        init_locs = self.init_x_dist.sample([batches, n_samples]).to(device=self.device).unsqueeze(2)
        particles_per_model = (n_samples // self.n_models) + 1
        init_regimes = pt.arange(self.n_models, dtype=pt.float32, device=self.device).tile((batches, particles_per_model))[:, :n_samples]
        return pt.cat((init_locs, init_regimes.unsqueeze(2)), dim = 2)
    
    def M_t_proposal(self, x_t_1, t: int):
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device) * self.sd_d
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=self.device)
        models = x_t_1[:, :, 1:2]
        index = models[:, :, 0].to(int)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.dyn_models[m](x_t_1[:,:,0][mask])
        new_pos = (locs.unsqueeze(2) + noise)
        return pt.cat((new_pos, models), dim = 2)
    
    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        for m in range(self.n_models):
            mask = (index == m)
            locs[mask] = self.obs_models[m](x_t[:,:,0][mask])
        return self.var_factor * ((self.y[t] - locs)**2)

class Redefined_RSDBPF(SSM):

    def set_observations(self, get_observation: Callable, t: int):
        self.y = self.reindexed_array(t-1, [get_observation(t-1), get_observation(t)])

    def __init__(self, n_models, switching_dyn:pt.nn.Module, dyn='Boot', device:str = 'cuda'):
        super().__init__(device)
        self.n_models = n_models
        self.dyn_models = pt.nn.ModuleList([Likelihood_NN(1, 8, 1) for _ in range(n_models)])
        self.obs_models = pt.nn.ModuleList([Likelihood_NN(1, 8, 1) for _ in range(n_models)])
        self.switching_dyn = switching_dyn
        for p in self.parameters():
            p  = p/5
        self.sd_d = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1)*0.4 + 0.1)
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        if dyn == 'Boot':
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided
        self.pi_fact = (1/2)* pt.log(pt.tensor(2*pt.pi))

    def M_0_proposal(self, batches:int, n_samples: int):
        self.obs_var_factor = -1/(2*(self.sd_o**2) + 1e-7)
        self.dyn_var_factor = -1/(2*(self.sd_d**2) + 1e-7)
        self.obs_pre_factor = -(1/2)*pt.log(self.sd_o**2 + 1e-7) - self.pi_fact
        self.dyn_pre_factor = -(1/2)*pt.log(self.sd_d**2 + 1e-7) - self.pi_fact
        init_locs = pt.zeros((batches, n_samples, 1), device = self.device)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim = 2)                   
    
    def M_t_proposal(self, x_t_1, t: int):
        new_pos = pt.zeros((x_t_1.size(0), x_t_1.size(1), 1), device=self.device)
        new_models = self.switching_dyn(x_t_1, t)
        return pt.cat((new_pos, new_models), dim = 2)
    
    def log_eta_t(self, x_t, t: int):
        pass

    def log_R_0(self, x_0):
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        return self.switching_dyn.get_log_probs(x_t, x_t_1)

    def log_f_t(self, x_t, t: int):
        index = x_t[:, :, 1].to(int)
        o_probs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        d_probs = pt.zeros_like(o_probs)
        for m in range(self.n_models):
            mask = (index == m)
            o_loc = self.obs_models[m](self.y[t][:, 1])
            o_probs = pt.where(mask, (self.obs_var_factor*((self.y[t][:, 0] - o_loc) ** 2) + self.obs_pre_factor).unsqueeze(1), o_probs)
            if t != 0:
                d_loc = self.dyn_models[m](self.y[t-1][:, 1])
                d_probs = pt.where(mask, (self.dyn_var_factor*((self.y[t][:, 1] - d_loc) ** 2) + self.dyn_pre_factor).unsqueeze(1), d_probs)
        return o_probs + d_probs

class LSTM(pt.nn.Module):

    def __init__(self, obs_dim, hid_dim, state_dim, n_layers, device='cuda') -> None:
        super().__init__()
        self.lstm = pt.nn.LSTM(obs_dim, hid_dim, n_layers, True, True, 0.0, False, state_dim, device)

    def forward(self, y_t):
            return self.lstm(y_t)[0]
        

class Transformer(pt.nn.Module):

    def __init__(self, obs_dim, hid_dim, state_dim, T:int = 50, device ='cuda'):
        super().__init__()
        self.encoder_layer = pt.nn.TransformerEncoderLayer(hid_dim, 1, hid_dim, 0.1, batch_first=True, device=device)
        self.transformer = pt.nn.TransformerEncoder(self.encoder_layer, 2)
        self.encoding = pt.nn.Linear(obs_dim, hid_dim, device=device)
        self.decoding = pt.nn.Linear(hid_dim, state_dim, device=device)
        self.relu = pt.nn.ReLU()
        self.mask = pt.tril(pt.ones((T+1, T+1), device=device))
    
    def forward(self, y_t):
        t = self.encoding(y_t)
        t = self.relu(t)
        t = self.transformer(t, mask = self.mask, is_causal = True)
        return self.decoding(t)