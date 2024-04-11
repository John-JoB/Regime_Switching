import torch as pt
from dpf.simulation import Differentiable_Particle_Filter
from tqdm import tqdm
from typing import Iterable
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from dpf.loss import Loss
from dpf import model
from dpf import loss as losses
from dpf import results


def _test(
        DPF: Differentiable_Particle_Filter, 
        loss: Loss, 
        T: int, 
        data: pt.utils.data.DataLoader, 
        ):
    DPF.eval()
    with pt.inference_mode():
        for i, simulated_object in enumerate(data):
            loss.clear_data()
            loss.register_data(truth=simulated_object)
            DPF(simulated_object, T, loss.get_reporters())
            loss_t = loss.per_step_loss()
            loss_t = loss_t.to(device ='cpu').detach().numpy()
    print(f'Test loss: {np.mean(loss_t)}')
    return np.array([np.mean(loss_t)]), np.mean(loss_t, axis = 0)

def e2e_likelihood_train(
        DPF: Differentiable_Particle_Filter,
        DPF_valid: Differentiable_Particle_Filter,
        opt: pt.optim.Optimizer,
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: Iterable[int], 
        set_fractions: Iterable[float], 
        epochs: int,
        test_scaling: float=1,
        opt_schedule: pt.optim.lr_scheduler.LRScheduler=None,
        verbose:bool=True,
        clip:float = pt.inf,
        lamb: float = 1.
        ):
    train_set, valid_set, test_set = pt.utils.data.random_split(data, set_fractions)
    if batch_size[0] == -1:
        batch_size[0] = len(train_set)
    if batch_size[1] == -1:
        batch_size[1] = len(valid_set)
    if batch_size[2] == -1:
        batch_size[2] = len(test_set)

    train = pt.utils.data.DataLoader(train_set, batch_size[0], shuffle=True, collate_fn=data.collate, num_workers= data.workers)
    valid = pt.utils.data.DataLoader(valid_set, min(batch_size[1], len(valid_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers)
    test = pt.utils.data.DataLoader(test_set, min(batch_size[2], len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
    loss =  losses.Magnitude_Loss(results.Log_Likelihood_Factors(), sign=-1)
    valid_loss_f = losses.Supervised_L2_Loss(function=lambda x: x[:, :, 0].unsqueeze(2))
    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    min_valid_loss = pt.inf
    DPF.model.dyn_models = DPF_valid.model.dyn_models 
    DPF.model.obs_models = DPF_valid.model.obs_models
    DPF.model.sd_o = DPF_valid.model.sd_o
    DPF.model.sd_d = DPF_valid.model.sd_d
    DPF.model.switching_dyn = DPF_valid.model.switching_dyn
    for epoch in range(epochs):
        DPF.train()
        try:
            DPF_valid.model.switching_dyn.dyn = 'Uni'
            DPF_valid.model.alg = DPF.model.PF_Type.Guided
        except:
            pass
        train_it = enumerate(train)
        for b, simulated_object in train_it:
            temp_object = model.Observation_Queue(simulated_object.state, pt.concat((simulated_object.observations, simulated_object.state[:, :, :1]), dim = 2))
            opt.zero_grad()
            loss.clear_data()
            loss.register_data(truth=temp_object)
            valid_loss_f.clear_data()
            valid_loss_f.register_data(truth=simulated_object)
            DPF(temp_object, T, loss.get_reporters())
            DPF_valid(simulated_object, T, valid_loss_f.get_reporters())
            loss_1 = loss()
            loss_2 = valid_loss_f()
            (loss_1 + lamb*loss_2).backward()
            for p in DPF.parameters():
                p.grad = pt.clamp(p.grad, -clip, clip)
            opt.step()
            DPF.model.sd_d.data.clamp_(-1, 1)
            DPF.model.sd_o.data.clamp_(-1, 1)
            train_loss[b + len(train)*epoch] = loss.item()
        if opt_schedule is not None:
            opt_schedule.step()
        DPF_valid.eval()
        try:
            DPF_valid.model.switching_dyn.dyn = 'Boot'
            DPF_valid.model.alg = DPF.model.PF_Type.Bootstrap
        except:
            pass
        with pt.inference_mode():
            for simulated_object in valid:
                valid_loss_f.clear_data()
                valid_loss_f.register_data(truth=simulated_object)
                DPF_valid(simulated_object, T, valid_loss_f.get_reporters())
                test_loss[epoch] += valid_loss_f().item()
            test_loss[epoch] /= len(valid)

        if test_loss[epoch].item() < min_valid_loss:
            min_valid_loss = test_loss[epoch].item()
            best_dict = deepcopy(DPF_valid.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(train_loss[epoch*len(train):(epoch+1)*len(train)])}')
            print(f'Validation loss: {test_loss[epoch]}\n')

    DPF_valid.load_state_dict(best_dict)
    DPF_valid.n_particles *= test_scaling
    DPF_valid.ESS_threshold *= test_scaling
    return _test(DPF_valid, valid_loss_f, T, test)

def test(DPF: Differentiable_Particle_Filter,
        loss: Loss, 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size:int, 
        fraction:float, 
        ):
        if fraction == 1:
            test_set = data
        else:
            test_set, _ =  pt.utils.data.random_split(data, [fraction, 1-fraction])
        if batch_size == -1:
            batch_size = len(test_set)
        test = pt.utils.data.DataLoader(test_set, min(batch_size, len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
        return _test(DPF, loss, T, test)

def e2e_train(
        DPF: Differentiable_Particle_Filter,
        opt: pt.optim.Optimizer,
        loss: Loss, 
        T: int, 
        data: pt.utils.data.Dataset, 
        batch_size: Iterable[int], 
        set_fractions: Iterable[float], 
        epochs: int,
        test_scaling: float=1,
        opt_schedule: pt.optim.lr_scheduler.LRScheduler=None,
        verbose:bool=True,
        clip:float = pt.inf
        ):
    try:
        train_set, valid_set, test_set = pt.utils.data.random_split(data, set_fractions)
        if batch_size[0] == -1:
            batch_size[0] = len(train_set)
        if batch_size[1] == -1:
            batch_size[1] = len(valid_set)
        if batch_size[2] == -1:
            batch_size[2] = len(test_set)

        train = pt.utils.data.DataLoader(train_set, batch_size[0], shuffle=True, collate_fn=data.collate, num_workers= data.workers)
        valid = pt.utils.data.DataLoader(valid_set, min(batch_size[1], len(valid_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers)
        test = pt.utils.data.DataLoader(test_set, min(batch_size[2], len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
    except:
        train, valid, test = set_fractions
    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    min_valid_loss = pt.inf
    
    for epoch in range(epochs):
        DPF.train()
        try:
            DPF.model.switching_dyn.dyn = 'Uni'
            DPF.model.alg = DPF.model.PF_Type.Guided
        except:
            pass
        train_it = enumerate(train)
        for b, simulated_object in train_it:
            opt.zero_grad()
            loss.clear_data()
            loss.register_data(truth=simulated_object)
            DPF(simulated_object, T, loss.get_reporters())

            loss()
            loss.backward()
            pt.nn.utils.clip_grad_value_(DPF.parameters(), clip)
            opt.step()
            train_loss[b + len(train)*epoch] = loss.item()
        if opt_schedule is not None:
            opt_schedule.step()
        DPF.eval()
        try:
            DPF.model.switching_dyn.dyn = 'Boot'
            DPF.model.alg = DPF.model.PF_Type.Bootstrap
        except:
            pass
        
        with pt.inference_mode():
            for simulated_object in valid:
                loss.clear_data()
                loss.register_data(truth=simulated_object)
                DPF(simulated_object, T, loss.get_reporters())
                test_loss[epoch] += loss().item()
            test_loss[epoch] /= len(valid)

        if test_loss[epoch].item() < min_valid_loss:
            min_valid_loss = test_loss[epoch].item()
            best_dict = deepcopy(DPF.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(train_loss[epoch*len(train):(epoch+1)*len(train)])}')
            print(f'Validation loss: {test_loss[epoch]}\n')

    DPF.load_state_dict(best_dict)
    DPF.n_particles *= test_scaling
    DPF.ESS_threshold *= test_scaling
    return _test(DPF, loss, T, test)

def train_s2s(NN: pt.nn.Module, opt: pt.optim.Optimizer, data: pt.utils.data.Dataset, batch_size: Iterable[int], 
        set_fractions: Iterable[float], 
        epochs: int,
        opt_schedule: pt.optim.lr_scheduler.LRScheduler=None,
        verbose:bool=True,
        clip:float = pt.inf
        ):
    try:
        train_set, valid_set, test_set = pt.utils.data.random_split(data, set_fractions)
        if batch_size[0] == -1:
            batch_size[0] = len(train_set)
        if batch_size[1] == -1:
            batch_size[1] = len(valid_set)
        if batch_size[2] == -1:
            batch_size[2] = len(test_set)

        train = pt.utils.data.DataLoader(train_set, batch_size[0], shuffle=True, collate_fn=data.collate, num_workers= data.workers)
        valid = pt.utils.data.DataLoader(valid_set, min(batch_size[1], len(valid_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers)
        test = pt.utils.data.DataLoader(test_set, min(batch_size[2], len(test_set)), shuffle=False, collate_fn=data.collate, num_workers= data.workers, drop_last=True)
    except:
        train, valid, test = set_fractions
    train_loss = np.zeros(len(train)*epochs)
    test_loss = np.zeros(epochs)
    min_valid_loss = pt.inf
    
    for epoch in range(epochs):
        NN.train()
        train_it = enumerate(train)
        for b, simulated_object in train_it:
            opt.zero_grad()
            x = NN(simulated_object.observations)
            loss = pt.mean((x - simulated_object.state[:, :, 0:1])**2)
            loss.backward()
            pt.nn.utils.clip_grad_value_(NN.parameters(), clip)
            opt.step()
            train_loss[b + len(train)*epoch] = loss.item()
        if opt_schedule is not None:
            opt_schedule.step()
        NN.eval()
        for simulated_object in valid:
            x = NN(simulated_object.observations)
            loss = pt.mean((x - simulated_object.state[:, :, 0:1])**2)
            test_loss[epoch] += loss.item()
        test_loss[epoch] /= len(valid)

        if test_loss[epoch] < min_valid_loss:
            min_valid_loss = test_loss[epoch]
            best_dict = deepcopy(NN.state_dict())

        if verbose:
            print(f'Epoch {epoch}:')
            print(f'Train loss: {np.mean(train_loss[epoch*len(train):(epoch+1)*len(train)])}')
            print(f'Validation loss: {test_loss[epoch]}\n')

    NN.load_state_dict(best_dict)
    for simulated_object in test:
        x = NN(simulated_object.observations)
        loss = pt.mean((x - simulated_object.state[:, :, 0:1])**2, dim=2)
        loss = loss.to(device ='cpu').detach().numpy()
        print(f'Test loss: {np.mean(loss)}')
        return np.array([np.mean(loss)]), np.mean(loss, axis = 0)