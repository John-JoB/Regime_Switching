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

def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--device', dest='device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device to use')
    parser.add_argument('--alg', dest='alg', type=str, default='RSPF', choices=['RLPF', 'RSDBPF', 'RSPF', 'DBPF', 'LSTM', 'MADPF', 'Transformer'], help='algorithm to use')
    parser.add_argument('--train_alg', dest='train_alg', type=str, default='Lambda', choices=['Lambda', 'MSE'], help='training method to use, only relavent for the RSDBPF and RLPF')
    parser.add_argument('--experiment', dest='experiment', type=str, default='Markov', choices=['Markov', 'Polya'], help='Experiment to run')
    parser.add_argument('--lr', dest='lr', type=float, default=0.05, help='Initial max learning rate')
    parser.add_argument('--opt', dest='opt', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='optimiser')
    parser.add_argument('--w_decay', dest='w_decay', type=float, default=0.05, help='Weight decay strength')
    parser.add_argument('--lr_steps', dest='lr_steps', nargs='+', type=int, default=[], help='steps to decrease the lr')
    parser.add_argument('--lr_gamma', dest='lr_gamma', type=float, default=0.5, help='learning rate decay per step')
    parser.add_argument('--clip', dest='clip', type=float, default=pt.inf,  help='Value to clip the gradient at')
    parser.add_argument('--init_scale', dest='init_scale', type=float, default=1, help='Rescale initilialised parameters')
    parser.add_argument('--lamb', dest='lamb', type=float, default=1, help='Ratio of ELBO to MSE loss')
    parser.add_argument('--store_loc', dest='store_loc', type=str, default='temp', help='File in the results folder to store the results dictionary')
    parser.add_argument('--n_runs', dest='n_runs', type=int, default=20, help='Number of runs to average')
    parser.add_argument('--softness', dest='softness', type=float, default=0.5, help='Softness of resampling for the ELBO loss')
    parser.add_argument('--layers', dest='layers', type=int, default=2, help='Number of fully connected layers in neural networks')
    parser.add_argument('--hid_size', dest='hidden_size', type=int, default=8, help='Number of nodes in hidden layers')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='temp', help='Data directory')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Number of epochs to train for')
    
    args = parser.parse_args()
    def create_data():
        nonlocal args
        if args.experiment == 'Markov':
            model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Markov_Switching(8, 0.8, 0.15, 'Boot', device=args.device), 'Boot', args.device)
        else:
            model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Polya_Switching(8, 'Boot', device=args.device), 'Boot', args.device)
        sim_obj = Simulated_Object(model, 100, 100, 1, args.device)
        sim_obj.save(f'./data/{args.data_dir}', 50, 20, '', bypass_ask=True)

    if args.alg == 'RSPF':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)
            if args.experiment == 'Markov':
                model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Markov_Switching(8, 0.8, 0.15, 'Boot', device=args.device), 'Boot', args.device)
            else:
                model = PF([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], [0, -2, 2, -4, 0, 2, -2, 4], 0.1, Polya_Switching(8, 'Boot', device=args.device), 'Boot', args.device)
            DPF = Differentiable_Particle_Filter(model, 2000, Soft_Resampler_Systematic(1, 0), 2001, args.device)
            loss = Supervised_L2_Loss(function=lambda x : x[:, :, 0:1])
            return test(DPF, loss, 50, data, -1, 0.25)

    if args.alg == 'RLPF' or args.alg == 'RSDBPF':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)
            if args.alg == 'RLPF':
                model = RSDBPF(8, NN_Switching(8, 8, 'Uni', args.device), args.init_scale, args.layers, args.hidden_size, 'Uni', args.device)
                re_model = Redefined_RSDBPF(8, NN_Switching(8, 8, 'Uni', args.device), 'Uni', args.device)
            elif args.experiment == 'Markov':
                model = RSDBPF(8, Markov_Switching(8, 0.8, 0.15, 'Boot', device=args.device), args.init_scale, args.layers, args.hidden_size, 'Boot', args.device)
                re_model = Redefined_RSDBPF(8, Markov_Switching(8, 0.8, 0.15, 'Boot', device=args.device), 'Boot', args.device)
            else: 
                model = RSDBPF(8, Polya_Switching(8, 'Boot', device=args.device), args.init_scale, args.layers, args.hidden_size, 'Boot', args.device)
                re_model = Redefined_RSDBPF(8, Polya_Switching(8, 'Boot', device=args.device), 'Boot', args.device)
            
            DPF = Differentiable_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, 0), 200, args.device)
            if args.opt == 'AdamW':
                opt = pt.optim.AdamW(params=DPF.parameters(), lr = args.lr, weight_decay=args.w_decay)
            else:
                opt = pt.optim.SGD(params= DPF.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.w_decay)
            if len(args.lr_steps) > 0:
                opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, args.lr_steps, args.lr_gamma)
            else:
                opt_sch = None
            if args.train_alg == 'Lambda':
                DPF_ELBO = Differentiable_Particle_Filter(re_model, 200, Soft_Resampler_Systematic(args.softness, 1), 200, args.device)
                return e2e_likelihood_train(DPF_ELBO, DPF, opt, 50, data, [100, -1, -1], [0.5, 0.25, 0.25], args.epochs, 10, opt_sch, True, args.clip, args.lamb)
            return e2e_train(DPF, opt, Supervised_L2_Loss(function=lambda x : x[:, :, 0].unsqueeze(2)), 50, data, [100, -1, -1], [0.5, 0.25, 0.25], args.epochs, 10, opt_sch, True, args.clip)
        
    if args.alg == 'DBPF' or args.alg == 'MADPF':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)
            if args.alg == 'DBPF':
                model = DBPF(8, args.init_scale, args.layers, args.hidden_size, args.device)
                DPF = Differentiable_Particle_Filter(model, 200, Soft_Resampler_Systematic(1, 0), 200, args.device)
            else:
                model = MAPF(8, args.init_scale, args.layers, args.hidden_size, args.device)
                DPF = MADPF(model, 200, Soft_Resampler_Systematic(1, 0), 201, args.device, 0.2)
            
            if args.opt == 'AdamW':
                print('AdamW')
                opt = pt.optim.AdamW(params=DPF.parameters(), lr = args.lr, weight_decay=args.w_decay)
            else:
                opt = pt.optim.SGD(params= DPF.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.w_decay)
            if len(args.lr_steps) > 0:
                opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, args.lr_steps, args.lr_gamma)
            else:
                opt_sch = None
            return e2e_train(DPF, opt, Supervised_L2_Loss(function=lambda x : x[:, :, 0].unsqueeze(2)), 50, data, [100, -1, -1], [0.5, 0.25, 0.25], args.epochs, 10, opt_sch, True, args.clip)

    if args.alg == 'LSTM' or args.alg == 'Transformer':
        def train_test():
            nonlocal args
            data = State_Space_Dataset(f'./data/{args.data_dir}', lazy = False, device=args.device, num_workers=0)
            if args.alg == 'LSTM':
                NN = LSTM(1, 20, 1, 1, args.device)
            else:
                NN = Transformer(1, 10, 1, 50, 'cuda')
            if args.opt == 'AdamW':
                print('AdamW')
                opt = pt.optim.AdamW(params=NN.parameters(), lr = args.lr, weight_decay=args.w_decay)
            else:
                opt = pt.optim.SGD(params= NN.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.w_decay)
            if len(args.lr_steps) > 0:
                opt_sch = pt.optim.lr_scheduler.MultiStepLR(opt, args.lr_steps, args.lr_gamma)
            else:
                opt_sch = None
            return train_s2s(NN, opt, data, [100, -1, -1], [0.5, 0.25, 0.25], args.epochs, opt_sch, True, args.clip)
        
    fix_rng(0) 
    def run():
        nonlocal create_data
        nonlocal train_test
        create_data()
        return train_test()
    
    dic = aggregate_runs(run, args.n_runs, ['loss', 'per_step_loss'])
    print(dic)
    with open(f'./results/{args.store_loc}.pickle', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__== '__main__':
    main()



