#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb
from datetime import datetime

from helpers.datasets import partition_data
from helpers.synthesizers import AdvSynthesizer
from helpers.utils import average_weights, DatasetSplit, KLDiv, setup_seed, test, kldiv
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100
from models.resnet import resnet18, resnet50
from models.vit import deit_tiny_patch16_224

warnings.filterwarnings('ignore')


def build_optimizer(params, args):
    opt = (args.optimizer or 'sgd').lower()
    wd = getattr(args, 'weight_decay', 0.0)
    if opt == 'sgd':
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=wd)
    if opt == 'adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=wd)
    if opt == 'adamw':
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {args.optimizer}. Use one of: sgd|adam|adamw")


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def update_weights(self, model, client_id, test_loader):
        model.train()
        optimizer = build_optimizer(model.parameters(), self.args)
        for iter in tqdm(range(self.args.local_ep)):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                output = model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
        
        acc, test_loss = test(model, test_loader)
        print('client_{}_accuracy: '.format(client_id), acc)
        return model.state_dict(), acc


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (L2 penalty)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')
    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float, help='temperature for KD')
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    parser.add_argument('--distill_steps', default=[10], nargs='+', type=int, help='number of distillation steps (single value or one per checkpoint)')
    parser.add_argument('--distill_steps_final', default=150, type=int, help='number of distillation steps for the initial checkpoint')
    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')
    # parser.add_argument('--other', default="", type=str,
    #                     help='seed for initializing training.')
    parser.add_argument('--alg', default="ours", type=str,
                        help='algorithm.')
    parser.add_argument('--checkpoint_steps', nargs='+', type=int, default=None,
                        help='steps for checkpoints')
    parser.add_argument('--kl_threshold', type=float, default=0.0,
                        help='KL divergence threshold for checkpoint selection (default: 0.0, no filtering)')
    
    # Paths
    parser.add_argument('--data_dir', default='/data/ccy/datasets', type=str, help='path to dataset')
    parser.add_argument('--weights_dir', default='weights', type=str, help='directory to save/load weights')
    parser.add_argument('--checkpoint_dir', default='df_ckpt', type=str, help='directory to save checkpoints')
    
    # Experiment
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb logging')
    parser.add_argument('--skip_eval', action='store_true', help='skip evaluation during training')
    parser.add_argument('--smooth_transition', action='store_true', help='enable smooth transition (interpolation) between checkpoints')
    parser.add_argument('--output_dir', default='experiments', type=str, help='root directory for experiment outputs')
    parser.add_argument('--run_name', default=None, type=str, help='optional name for the run')
    parser.add_argument('--continuous_teacher', action='store_true', help='maintain a single teacher model instance and update weights')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer: sgd|adam|adamw')

    args = parser.parse_args()
    return args


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def kd_train_ensemble(synthesizer, model, criterion, optimizer, test_loader):
    student, teacher = model
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct = 0.0

    for idx, (images, label) in enumerate(synthesizer.get_data()):
        optimizer.zero_grad()
        images = images.cuda()
        with torch.no_grad():
            t_out = teacher(images)
        s_out = student(images.detach())
        loss_s = criterion(s_out, t_out.detach())

        loss_s.backward()
        optimizer.step()

        total_loss += loss_s.detach().item()
        pred = s_out.argmax(dim=1)
        target = t_out.argmax(dim=1)
        correct += pred.eq(target.view_as(pred)).sum().item()


def kd_train(synthesizer, model, criterion, optimizer, test_loader, ensemble_model, traindata_cls_counts, num_class):
    student, teacher_list = model
    student.train()
    ensemble_model.eval()
    total_loss = 0.0
    correct = 0.0

    for idx, (images, label) in enumerate(synthesizer.get_data()):
        # ensemble_pred = ensemble_model(images.cuda())
        # teacher_idx=0
        images = images.cuda()
        
        for teacher in teacher_list:
            optimizer.zero_grad()
            with torch.no_grad():
                # what's meaning of this line?
                # ensemble_pred = ensemble_model(images)
                t_out = teacher(images)

            mask = (t_out.max(1)[1] == label.cuda()).float()

            s_out = student(images.detach())

            loss_s = (F.cross_entropy(s_out, label.cuda(), reduction='none') * mask).mean()
            loss_s += (kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()


def calculate_kl_divergence(model1, model2, anchor_loader):
    """
    Calculate KL divergence between two models on an anchor set.
    Score = KL(P(y|x; model2) || P(y|x; model1))
    model1: Previous model (W_last)
    model2: Current model (W_t)
    """
    model1.eval()
    model2.eval()
    kl_div_sum = 0.0
    count = 0
    
    with torch.no_grad():
        for images, _ in anchor_loader:
            images = images.cuda()
            
            # Get logits
            logits1 = model1(images)
            logits2 = model2(images)
            
            # Softmax to get probabilities
            probs1 = F.softmax(logits1, dim=1)
            log_probs1 = F.log_softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            log_probs2 = F.log_softmax(logits2, dim=1)
            
            # KL Divergence: sum(p(x) * (log(p(x)) - log(q(x))))
            # Here we use PyTorch's F.kl_div which expects input to be log-probabilities
            # and target to be probabilities.
            # KL(P || Q) -> input=log_Q, target=P
            # We want KL(P(model2) || P(model1)) -> input=log_probs1, target=probs2
            
            batch_kl = F.kl_div(log_probs1, probs2, reduction='batchmean')
            kl_div_sum += batch_kl.item() * images.size(0)
            count += images.size(0)
            
    return kl_div_sum / count


def interpolate_weights(w_prev, w_curr, alpha):
    """
    Interpolate between two state_dicts: (1 - alpha) * w_prev + alpha * w_curr
    """
    w_new = {}
    for key in w_curr.keys():
        w_new[key] = (1 - alpha) * w_prev[key] + alpha * w_curr[key]
    return w_new


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().cuda()
    elif args.model == "res":
        num_classes = 200
        if args.dataset == "eurosat":
            num_classes = 10
        elif args.dataset == "nwpu":
            num_classes = 45
        elif args.dataset == "siri-whu":
            num_classes = 12
        global_model = resnet18(num_classes=num_classes).cuda()
    elif args.model == "res50":
        num_classes = 200
        if args.dataset == "eurosat":
            num_classes = 10
        elif args.dataset == "nwpu":
            num_classes = 45
        elif args.dataset == "siri-whu":
            num_classes = 12
        global_model = resnet50(num_classes=num_classes).cuda()
    elif args.model == "vit":
        num_classes = 10
        if args.dataset == "tiny":
            num_classes = 200
        elif args.dataset == "cifar100":
            num_classes = 100
        elif args.dataset == "nwpu":
            num_classes = 45
        elif args.dataset == "siri-whu":
            num_classes = 12
        elif args.dataset == "eurosat":
            num_classes = 10

        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, num_classes)
        global_model = global_model.cuda()
        global_model = torch.nn.DataParallel(global_model)
    else:
        # Default fallback or error handling
        global_model = CNNCifar().cuda()
    return global_model


def main():
    args = args_parser()
    
    # Setup WandB
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(config=args,
                   project="ont-shot distillation experiment",
                   name=(args.dataset + "-" + str(args.beta) + "-" + args.alg + "-client" + str(args.num_users)))

    setup_seed(args.seed)
    
    # Setup Experiment Directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = args.run_name + "_" + run_id
    
    # Create a unique directory for this run
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Update paths to be inside the run directory
    args.save_dir = os.path.join(run_dir, 'synthesis')
    args.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    
    # Ensure directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Experiment outputs will be saved to: {run_dir}")

    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta, num_users=args.num_users, data_dir=args.data_dir)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                              shuffle=False, num_workers=4)
    
    global_model = get_model(args)
    bst_acc = -1
    local_weights = []
    global_model.train()
    
    if args.type == "pretrain":
        # In pretrain mode, we save weights to the run directory
        weights_save_dir = os.path.join(run_dir, 'weights')
        os.makedirs(weights_save_dir, exist_ok=True)
        print(f"Pretrained weights will be saved to: {weights_save_dir}")

        # Create anchor set for KL divergence calculation if needed
        anchor_loader = None
        if args.kl_threshold > 0:
            # Use a small subset of test data as anchor set (e.g., 64 images)
            anchor_indices = np.random.choice(len(test_dataset), 64, replace=False)
            anchor_subset = torch.utils.data.Subset(test_dataset, anchor_indices)
            anchor_loader = torch.utils.data.DataLoader(anchor_subset, batch_size=32, shuffle=False)
            print(f"Created anchor set with {len(anchor_subset)} images for KL filtering (threshold={args.kl_threshold})")

        saved_epochs = []
        client_last_saved_weights = [None] * args.num_users
        client_save_counts = [0] * args.num_users

        for i in range(args.epochs):
            round_accuracies = []
            for idx in range(args.num_users):
                print("client {}".format(idx))
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
                if i == 0:
                    w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx, test_loader)
                else:
                    global_model.load_state_dict(local_weights[idx])
                    w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx, test_loader)

                round_accuracies.append(local_acc)
                if i == 0:
                    local_weights.append(copy.deepcopy(w))
                else:
                    local_weights[idx] = w

            avg_client_acc = float(np.mean(round_accuracies)) if round_accuracies else 0.0
            print(f"Round {i+1} Average Client Accuracy: {avg_client_acc:.4f}")

            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            
            if not args.skip_eval:
                print("avg acc:")
                test(global_model, test_loader)
            
            # Create ensemble model for testing and KL check
            model_list = []
            for idx_client in range(len(local_weights)):
                net = copy.deepcopy(global_model)
                net.load_state_dict(local_weights[idx_client])
                model_list.append(net)
            ensemble_model = Ensemble(model_list)
            
            if not args.skip_eval:
                print("ensemble acc:")
                test(ensemble_model, test_loader)

            # Checkpoint Filtering Logic
            should_save = False
            save_data = local_weights # Default to saving current weights
            current_step = args.local_ep * (i + 1)
            
            # Checkpoint Filtering Logic
            clients_to_save = []
            
            if args.kl_threshold > 0:
                updated_clients_count = 0
                client_kl_details = []
                
                for k in range(args.num_users):
                    update_k = False
                    kl_val = 0.0
                    
                    if client_last_saved_weights[k] is None:
                        update_k = True
                        kl_val = float('inf')
                    else:
                        # Instantiate models for KL calculation
                        net_prev = copy.deepcopy(global_model)
                        net_prev.load_state_dict(client_last_saved_weights[k])
                        
                        net_curr = copy.deepcopy(global_model)
                        net_curr.load_state_dict(local_weights[k])
                        
                        kl_val = calculate_kl_divergence(net_prev, net_curr, anchor_loader)
                        
                        if kl_val >= args.kl_threshold:
                            update_k = True
                    
                    # Always update on the last round to ensure final state is saved
                    if i == args.epochs - 1:
                        update_k = True
                        
                    if update_k:
                        client_last_saved_weights[k] = copy.deepcopy(local_weights[k])
                        clients_to_save.append(k)
                        updated_clients_count += 1
                        client_save_counts[k] += 1
                        client_kl_details.append(f"C{k}:YES({kl_val:.4f})")
                    else:
                        client_kl_details.append(f"C{k}:NO({kl_val:.4f})")
                
                if updated_clients_count > 0:
                    print(f"Round {i+1} (Step {current_step}): {updated_clients_count}/{args.num_users} clients updated. Saving.")
                    print(f"Details: {', '.join(client_kl_details)}")
                else:
                    print(f"Round {i+1} (Step {current_step}): No clients updated (KL < {args.kl_threshold}). Skipping.")

            else:
                # Logic for kl_threshold <= 0 (Interval or List based saving)
                should_save_step = False
                if i == args.epochs - 1:
                    should_save_step = True # Always save last
                    print(f"Always save the last checkpoint: Round {i+1}")
                elif args.checkpoint_steps is None:
                    should_save_step = True # Default: save every round
                elif len(args.checkpoint_steps) == 1:
                    interval = args.checkpoint_steps[0]
                    if (i + 1) % interval == 0:
                        should_save_step = True
                        print(f"Saving checkpoint at interval {interval}: Round {i+1}")
                else:
                    if (i + 1) in args.checkpoint_steps:
                        should_save_step = True
                        print(f"Saving checkpoint at specified round: {i+1}")
                
                if should_save_step:
                    clients_to_save = list(range(args.num_users))
                    # Update save counts for all clients
                    for k in range(args.num_users):
                        client_save_counts[k] += 1
            
            if clients_to_save:
                saved_epochs.append(current_step)
                for k in clients_to_save:
                    file_name = '{}_{}clients_{}_{}epoch_{}seed_client{}.pkl'.format(
                        args.dataset, args.num_users, args.beta, current_step, args.seed, k)
                    file_path = os.path.join(weights_save_dir, file_name)
                    torch.save(local_weights[k], file_path)
            
            print(f"Round {i+1} Client Save Counts: {client_save_counts}")
        
        print("Pretraining finished.")
        print(f"Saved checkpoints at steps: {saved_epochs}")
        print(f"Client save counts: {client_save_counts}")
        print(f"Checkpoints saved to: {weights_save_dir}")

    elif args.type == "fedavg":
        print(f"Starting FedAvg training for {args.epochs} rounds...")
        fedavg_save_dir = os.path.join(run_dir, 'fedavg_checkpoints')
        os.makedirs(fedavg_save_dir, exist_ok=True)
        
        fedavg_acc_list = []
        
        for epoch in range(args.epochs):
            local_weights = []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            
            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
                w, local_acc = local_model.update_weights(copy.deepcopy(global_model), idx, test_loader)
                local_weights.append(copy.deepcopy(w))
            
            # update global weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            
            # Test global model
            test_acc, test_loss = test(global_model, test_loader)
            fedavg_acc_list.append(test_acc)
            
            if test_acc > bst_acc:
                bst_acc = test_acc
                torch.save(global_model.state_dict(), os.path.join(fedavg_save_dir, 'best_model.pth'))
            
            print(f"Round {epoch+1}, Test Acc: {test_acc}, Best Acc: {bst_acc}")
            wandb.log({'accuracy': test_acc, 'round': epoch})

        np.save(os.path.join(run_dir, "fedavg_acc_{}.npy".format(args.dataset)), np.array(fedavg_acc_list))

    else:
        # In distillation mode, we load weights from args.weights_dir
        # User must specify the correct directory containing the .pkl files
        print(f"Loading pretrained weights from: {args.weights_dir}")
        
        file_path_template = os.path.join(args.weights_dir, '{}_{}clients_{}_{}epoch_{}seed_client{}.pkl')
        
        # Initialize local_weights with current global model state (as fallback)
        local_weights = [copy.deepcopy(global_model.state_dict()) for _ in range(args.num_users)]
        
        # Try to load initial weights (Step = args.local_ep, i.e., Round 1)
        initial_step = args.local_ep
        loaded_count = 0
        for k in range(args.num_users):
            file_path = file_path_template.format(
                args.dataset, args.num_users, args.beta, initial_step, args.seed, k)
            
            if os.path.exists(file_path):
                local_weights[k] = torch.load(file_path)
                loaded_count += 1
        
        # Fallback to monolithic file if no individual files found
        if loaded_count == 0:
            monolithic_file = os.path.join(args.weights_dir, '{}_{}clients_{}_{}epoch_{}seed.pkl'.format(
                args.dataset, args.num_users, args.beta, initial_step, args.seed))
            
            if os.path.exists(monolithic_file):
                print(f"Found monolithic checkpoint for initial step {initial_step}. Loading...")
                all_weights = torch.load(monolithic_file)
                if isinstance(all_weights, list) and len(all_weights) == args.num_users:
                    local_weights = all_weights
                    loaded_count = args.num_users
                else:
                    print(f"Error: Monolithic file format mismatch.")

        if loaded_count == 0:
            print(f"Warning: No client weights found for initial step {initial_step}. Using random initialization.")
        else:
            print(f"Loaded {loaded_count}/{args.num_users} client weights for initial step {initial_step}.")

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        print("avg acc:")
        test(global_model, test_loader)
        
        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
        ensemble_model = Ensemble(model_list)
        print("ensemble acc:")
        test(ensemble_model, test_loader)
        
        global_model = get_model(args)

        # data generator
        nz = args.nz
        nc = 3 if "cifar" in args.dataset or args.dataset == "svhn" or args.dataset == "eurosat" or args.dataset == "nwpu" or args.dataset == "siri-whu" else 1
        img_size = 32 if "cifar" in args.dataset or args.dataset == "svhn" else 28
        if args.dataset == "tiny" or args.dataset == "eurosat":
            img_size = 64
        elif args.dataset == "nwpu" or args.dataset == "siri-whu":
             img_size = 224
        
        generator1 = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()
        generator2 = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda() if args.alg == "DFDG" else None
        
        args.cur_ep = 0
        img_size2 = (3, 32, 32) if "cifar" in args.dataset or args.dataset == "svhn" else (1, 28, 28)
        if args.dataset == "tiny" or args.dataset == "eurosat":
            img_size2 = (3, 64, 64)
        elif args.dataset == "nwpu" or args.dataset == "siri-whu":
            img_size2 = (3, 224, 224)
        
        num_class = 100 if args.dataset == "cifar100" else 10
        if args.dataset == "tiny":
            num_class = 200
        elif args.dataset == "nwpu":
            num_class = 45
        elif args.dataset == "siri-whu":
            num_class = 12
            
        synthesizer = AdvSynthesizer(ensemble_model, model_list, global_model, generator1, generator2,
                                     nz=nz, num_classes=num_class, img_size=img_size2,
                                     iterations=args.g_steps, lr_g=args.lr_g,
                                     synthesis_batch_size=args.synthesis_batch_size,
                                     sample_batch_size=args.batch_size,
                                     adv=args.adv, bn=args.bn, oh=args.oh,
                                     save_dir=args.save_dir, dataset=args.dataset, alg=args.alg)
        
        criterion = KLDiv(T=args.T)
        optimizer = build_optimizer(global_model.parameters(), args)
        global_model.train()
        distill_acc = []
        
        if args.checkpoint_steps is not None:
            checkpoint_list = sorted(args.checkpoint_steps)
        else:
            # Scan directory for checkpoints matching the current configuration
            checkpoint_list = set()
            client_checkpoint_counts = [0] * args.num_users
            import re
            # Pattern: {dataset}_{num_users}clients_{beta}_{step}epoch_{seed}seed_client{k}.pkl
            pattern = re.compile(rf"^{re.escape(args.dataset)}_{args.num_users}clients_{args.beta}_(\d+)epoch_{args.seed}seed_client(\d+)\.pkl$")
            # Pattern for monolithic: {dataset}_{num_users}clients_{beta}_{step}epoch_{seed}seed.pkl
            pattern_mono = re.compile(rf"^{re.escape(args.dataset)}_{args.num_users}clients_{args.beta}_(\d+)epoch_{args.seed}seed\.pkl$")
            
            if os.path.exists(args.weights_dir):
                files = os.listdir(args.weights_dir)
                for f in files:
                    match = pattern.match(f)
                    if match:
                        step = int(match.group(1))
                        client_id = int(match.group(2))
                        checkpoint_list.add(step)
                        if 0 <= client_id < args.num_users:
                            client_checkpoint_counts[client_id] += 1
                    else:
                        match_mono = pattern_mono.match(f)
                        if match_mono:
                            step = int(match_mono.group(1))
                            checkpoint_list.add(step)
            
            checkpoint_list = sorted(list(checkpoint_list))
            print(f"Client Checkpoint Counts: {client_checkpoint_counts}")
            
            if not checkpoint_list:
                print(f"No checkpoints found in {args.weights_dir} matching pattern.")
                if args.alg != "ours":
                     checkpoint_list = [args.local_ep]
                else:
                     checkpoint_list = []
        
        print(f"Checkpoints to distill: {checkpoint_list}")

        # Test the last checkpoint (reconstructed from latest available client files)
        if checkpoint_list:
            last_step = checkpoint_list[-1]
            print(f"\n[Test] Evaluating Last Checkpoint (Step {last_step})")
            print("Reconstructing state by scanning backwards for latest client weights...")
            
            test_local_weights = [None] * args.num_users
            file_path_template = os.path.join(args.weights_dir, '{}_{}clients_{}_{}epoch_{}seed_client{}.pkl')
            
            for k in range(args.num_users):
                found = False
                # Search backwards from the last step
                for step in reversed(checkpoint_list):
                    # Try individual file
                    fpath = file_path_template.format(args.dataset, args.num_users, args.beta, step, args.seed, k)
                    if os.path.exists(fpath):
                        test_local_weights[k] = torch.load(fpath)
                        found = True
                        break
                    
                    # Try monolithic file
                    mono_path = os.path.join(args.weights_dir, '{}_{}clients_{}_{}epoch_{}seed.pkl'.format(
                        args.dataset, args.num_users, args.beta, step, args.seed))
                    if os.path.exists(mono_path):
                        try:
                            all_weights = torch.load(mono_path)
                            if isinstance(all_weights, list) and len(all_weights) > k:
                                test_local_weights[k] = all_weights[k]
                                found = True
                                break
                        except:
                            pass
                
                if not found:
                    print(f"Client {k}: No weights found in checkpoint list. Using random initialization.")
                    test_local_weights[k] = copy.deepcopy(global_model.state_dict())

            # Test Avg Model
            test_global_weights = average_weights(test_local_weights)
            test_global_model = copy.deepcopy(global_model)
            test_global_model.load_state_dict(test_global_weights)
            print("Last Checkpoint Avg Model Acc:")
            test(test_global_model, test_loader)
            
            # Test Ensemble Model
            test_model_list = []
            for w in test_local_weights:
                net = copy.deepcopy(global_model)
                net.load_state_dict(w)
                test_model_list.append(net)
            test_ensemble_model = Ensemble(test_model_list)
            print("Last Checkpoint Ensemble Model Acc:")
            test(test_ensemble_model, test_loader)
            
            # Cleanup to free memory
            del test_local_weights, test_global_model, test_ensemble_model, test_model_list
            torch.cuda.empty_cache()
            print("[Test] Evaluation finished.\n")

        last_local_weights = None
        total_checkpoints = len(checkpoint_list)

        for ckpt_idx, step in enumerate(checkpoint_list):
            print(f"\n[Distillation Progress] Checkpoint {ckpt_idx + 1}/{total_checkpoints} (Step {step}) | Remaining: {total_checkpoints - ckpt_idx - 1}")
            
            # Load individual client files for this step
            updated_count = 0
            file_path_template = os.path.join(args.weights_dir, '{}_{}clients_{}_{}epoch_{}seed_client{}.pkl')
            
            for k in range(args.num_users):
                file_path = file_path_template.format(
                    args.dataset, args.num_users, args.beta, step, args.seed, k)
                
                if os.path.exists(file_path):
                    local_weights[k] = torch.load(file_path)
                    updated_count += 1
            
            # Fallback to monolithic file if no individual files found
            if updated_count == 0:
                monolithic_file = os.path.join(args.weights_dir, '{}_{}clients_{}_{}epoch_{}seed.pkl'.format(
                    args.dataset, args.num_users, args.beta, step, args.seed))
                
                if os.path.exists(monolithic_file):
                    print(f"Found monolithic checkpoint for step {step}. Loading...")
                    all_weights = torch.load(monolithic_file)
                    if isinstance(all_weights, list) and len(all_weights) == args.num_users:
                        local_weights = all_weights
                        updated_count = args.num_users
            
            if updated_count == 0:
                 print(f"Warning: No client updates found for step {step}. Using previous weights.")

            model_list = []
             
            for i in range(len(local_weights)):
                net = copy.deepcopy(global_model)
                net.load_state_dict(local_weights[i])
                model_list.append(net)
            ensemble_model = Ensemble(model_list)
            
            synthesizer.set_teacher(ensemble_model, model_list, step, args.save_dir)
            
            if len(args.distill_steps) > 1:
                # Use array-based steps if more than one value provided
                if ckpt_idx < len(args.distill_steps):
                    temp_step = args.distill_steps[ckpt_idx]
                else:
                    temp_step = args.distill_steps[-1]
            else:
                # Default logic for single value
                # Note: args.distill_steps is now a list [10] by default
                val = args.distill_steps[0]
                if step != args.local_ep:
                    temp_step = val
                else:
                    temp_step = args.distill_steps_final
            
            pbar = tqdm(range(int(temp_step)))
            for epoch in pbar:
                # Smooth Transition (Interpolation)
                if args.smooth_transition and last_local_weights is not None:
                    # alpha goes from 1/temp_step to 1.0
                    alpha = (epoch + 1) / temp_step
                    
                    # Interpolate weights for each client model
                    for i in range(len(local_weights)):
                        w_interp = interpolate_weights(last_local_weights[i], local_weights[i], alpha)
                        model_list[i].load_state_dict(w_interp)
                    
                    # Note: ensemble_model uses model_list, so it is automatically updated.
                    # However, synthesizer might need to know about the update if it caches anything.
                    # But synthesizer.gen_data uses self.teacher (which is ensemble_model).
                    # So it should be fine.
                
                # 1. Data synthesis
                synthesizer.gen_data(args.cur_ep)
                args.cur_ep += 1

                if args.alg != "ours":
                    kd_train_ensemble(synthesizer, [global_model, ensemble_model], criterion, optimizer, test_loader)
                else:
                    if args.continuous_teacher:
                        # Use the updated model_list and ensemble_model
                        kd_train(synthesizer, [global_model, [ensemble_model]], criterion, optimizer, test_loader, ensemble_model, traindata_cls_counts, num_class)
                    else:
                        kd_train(synthesizer, [global_model, model_list], criterion, optimizer, test_loader, ensemble_model, traindata_cls_counts, num_class)
                
                if not args.skip_eval:
                    acc, test_loss = test(global_model, test_loader)
                    distill_acc.append(acc)
                    bst_acc = max(acc, bst_acc)
                    pbar.set_description("Ckpt {}/{} | Step {} | Ep {} | Acc: {:.2f} | Best: {:.2f}".format(
                        ckpt_idx + 1, total_checkpoints, step, epoch, acc, bst_acc))
                    wandb.log({'accuracy': acc})
                else:
                    pbar.set_description("Ckpt {}/{} | Step {} | Ep {}".format(
                        ckpt_idx + 1, total_checkpoints, step, epoch))
            
            # Update last_local_weights for next step
            last_local_weights = copy.deepcopy(local_weights)

        np.save(os.path.join(run_dir, "distill_acc_{}.npy".format(args.dataset)), np.array(distill_acc))


if __name__ == '__main__':
    main()
