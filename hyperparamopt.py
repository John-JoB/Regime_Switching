import hyperopt as hypo
import argparse
from Net import PF, RSDBPF, Redefined_RSDBPF, Markov_Switching, Polya_Switching, NN_Switching, DBPF, LSTM, MAPF, Transformer
from dpf.model import Simulated_Object, State_Space_Dataset
from trainingRS import test, e2e_train, e2e_likelihood_train, train_s2s
from dpf.simulation import Differentiable_Particle_Filter
from simulationRS import MADPF
from dpf.resampling import Soft_Resampler_Systematic
from dpf.loss import Supervised_L2_Loss
import torch as pt
from dpf.utils import aggregate_runs, fix_rng
import pickle
import numpy as np

def optimise(function, space, max_evals):
    t = hypo.Trials()
    best = hypo.fmin(fn=function, space=space, algo=hypo.tpe.suggest, max_evals=max_evals, trials=t)
    return best

transformer_space = {'lr': hypo.hp.lognormal('_lr', -3, 1),
         'w_decay': hypo.hp.lognormal('_w_decay', -3, 1),
         'lr_gamma': hypo.hp.uniform('_lr_gamma', 0.5, 1),
         'clip': hypo.hp.loguniform('_clip', 0, 3),
         'hidden_size': hypo.hp.quniform('_hid_size', 4, 30, 1),
         'T': hypo.hp.quniform('_T', 5, 60, 1),
         'layers': hypo.hp.quniform('_layers', 1, 5, 1)}

RLPF_space = {'lr': hypo.hp.lognormal('_lr', -3, 1),
         'w_decay': hypo.hp.lognormal('_w_decay', -3, 1),
         'lr_gamma': hypo.hp.uniform('_lr_gamma', 0.5, 1),
         'clip': hypo.hp.loguniform('_clip', 0, 3),
         'init_scale': hypo.hp.lognormal('_init_scale', 0, 1),
         'lamb': hypo.hp.normal('_lamb', 2, 1.5),
         'soft_choice': hypo.hp.choice('_soft_choice', [{'softness' : 1}, {'softness': hypo.hp.uniform('_softness', 0.3, 1)}]),
         'grad_decay': hypo.hp.uniform('_grad_decay', 0, 1),
         'layers_info': hypo.hp.choice('_layers_info', [{'layers' : 2,  'hid_size': hypo.hp.qnormal('_hid_size_1', 20, 10, 1)}, {'layers' : 3, 'hid_size' : hypo.hp.quniform('_hid_size_2', 4, 15, 1)}])}

def runRLPF(param_dict):
    print(param_dict)
    


    data = State_Space_Dataset(f'./data/hyp_opt', lazy = False, device='cuda', num_workers=0)
    model = RSDBPF(8, NN_Switching(8, 8, 'Uni', 'cuda'), param_dict['init_scale'], int(param_dict['layers_info']['layers']), int(abs(param_dict['layers_info']['hid_size'])), 'Uni', 'cuda')
    re_model = Redefined_RSDBPF(8, NN_Switching(8, 8, 'Uni', 'cuda'), 'Uni', 'cuda')
    DPF = Differentiable_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, param_dict['grad_decay']), 200, 'cuda')
    opt = pt.optim.AdamW(params=DPF.parameters(), lr = param_dict['lr'], weight_decay=param_dict['w_decay'])
    DPF_ELBO = Differentiable_Particle_Filter(re_model, 200, Soft_Resampler_Systematic(param_dict['soft_choice']['softness'], 1), 200, 'cuda')
    opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, [10, 20, 30, 40], param_dict['lr_gamma'])
    _, loss = e2e_likelihood_train(DPF_ELBO, DPF, opt, 50, data, [100, -1, -1], [0.5, 0.25, 0.25], 20, 10, opt_sch, False, param_dict['clip'], abs(param_dict['lamb']))
    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
        return 10000
    return np.mean(loss)

def run_transformer(param_dict):
    


    data = State_Space_Dataset(f'./data/hyp_opt', lazy = False, device='cuda', num_workers=0)
    NN = Transformer(1, int(param_dict['hidden_size']), 1, int(param_dict['T']), 'cuda', int(param_dict['layers']))
    opt = pt.optim.AdamW(params=NN.parameters(), lr = param_dict['lr'], weight_decay=param_dict['w_decay'])
    opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, [10, 20, 30, 40], param_dict['lr_gamma'])
    _, loss = train_s2s(NN, opt, data, [100, -1, -1], [0.5, 0.25, 0.25], 50, opt_sch, False, param_dict['clip'])
    if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
        return 10000
    return np.mean(loss)



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--alg', dest='alg', type=str, default='RSPF', choices=['RLPF', 'RSDBPF', 'RSPF', 'DBPF', 'LSTM', 'MADPF', 'Transformer'], help='algorithm to use')
    parser.add_argument('--evals', dest='evals', type=int, default=50, help='number of evals')
    args = parser.parse_args()
    if args.alg == 'RLPF':
        fun = runRLPF
        space = RLPF_space
    if args.alg == 'Transformer':
        fun = run_transformer
        space = transformer_space
    print(optimise(fun, space, args.evals))



main()