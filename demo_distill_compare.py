import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Adjust path to import helpers and models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helpers.datasets import partition_data
from helpers.synthesizers import AdvSynthesizer
from helpers.utils import KLDiv, setup_seed, test, kldiv, average_weights
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100
from models.resnet import resnet18, resnet50
from models.vit import deit_tiny_patch16_224

# ==================== Utils & Classes ====================

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

def kd_train(synthesizer, model, criterion, optimizer, ensemble_model):
    student, teacher_list = model
    student.train()
    ensemble_model.eval()
    
    for idx, (images, label) in enumerate(synthesizer.get_data()):
        images = images.cuda()
        
        # Teacher output (Ensemble)
        with torch.no_grad():
             t_out = ensemble_model(images)

        mask = (t_out.max(1)[1] == label.cuda()).float()
        s_out = student(images)
        
        # Standard KD loss KL(S||T)
        loss_s = (kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
        # You can add hard label loss if needed: F.cross_entropy(s_out, label.cuda())

        optimizer.zero_grad()
        loss_s.backward()
        optimizer.step()

# ==================== Main Demo Logic ====================

def main():
    parser = argparse.ArgumentParser(description="Distillation Comparison Demo")
    parser.add_argument('--steps', nargs='+', type=int, required=True, help='List of checkpoint steps (rounds) to compare')
    parser.add_argument('--weights_dir', type=str, required=True, help='Directory containing the weights')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
    parser.add_argument('--model', type=str, default='res', help='model name')
    parser.add_argument('--data_dir', type=str, default='/data/ccy/datasets', help='data directory')
    parser.add_argument('--num_users', type=int, default=5, help='number of clients')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for synthesis')
    parser.add_argument('--synthesis_batch_size', type=int, default=256, help='batch size for synthesis')
    parser.add_argument('--distill_epochs', type=int, default=20, help='distillation epochs for each checkpoint')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for student')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for student')
    parser.add_argument('--lr_g', type=float, default=1e-3, help='learning rate for generator')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer: sgd|adam|adamw')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Data Free params (needed for Synthesizer)
    parser.add_argument('--adv', default=1, type=float, help='scaling factor for adv loss')
    parser.add_argument('--bn', default=1, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=1, type=float, help='scaling factor for one hot loss')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss')
    parser.add_argument('--g_steps', default=20, type=int, help='number of iterations for generation')
    parser.add_argument('--T', default=20, type=float, help='temperature for KD')
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--partition', default='dirichlet', type=str)
    
    args = parser.parse_args()

    setup_seed(args.seed)
    print(f"--- Running Distillation Comparison using {args.dataset} ---")
    
    # Construct checkpoint paths
    checkpoints = []
    # Try pattern: {dataset}_{num_users}clients_{beta}_{step}epoch_{seed}seed_client{k}.pkl
    # Or monolithic: {dataset}_{num_users}clients_{beta}_{step}epoch_{seed}seed.pkl
    
    print(f"Searching for weights in {args.weights_dir}...")
    
    for step in args.steps:
        # First try to gather individual client files
        client_files = []
        missing_clients = False
        for k in range(args.num_users):
            fname = f"{args.dataset}_{args.num_users}clients_{args.beta}_{step}epoch_{args.seed}seed_client{k}.pkl"
            fpath = os.path.join(args.weights_dir, fname)
            if os.path.exists(fpath):
                client_files.append(fpath)
            else:
                missing_clients = True
                break
        
        if not missing_clients and len(client_files) == args.num_users:
            checkpoints.append({'step': step, 'type': 'individual', 'paths': client_files})
            print(f"Found {len(client_files)} individual client files for step {step}")
            continue
            
        # Fallback to monolithic
        fname_mono = f"{args.dataset}_{args.num_users}clients_{args.beta}_{step}epoch_{args.seed}seed.pkl"
        fpath_mono = os.path.join(args.weights_dir, fname_mono)
        if os.path.exists(fpath_mono):
            checkpoints.append({'step': step, 'type': 'monolithic', 'path': fpath_mono})
            print(f"Found monolithic file for step {step}")
        else:
            print(f"Warning: Could not find complete weights for step {step}. Skipping.")

    if not checkpoints:
        print("No valid checkpoints found. Exiting.")
        return

    # 1. Prepare Data (for evaluation)
    print("Loading data...")
    # Using 'iid' partition just to get the test set quickly. 
    # We strictly need test_dataset, traindata_cls_counts is not strictly used unless weighted synthesis.
    _, test_dataset, _, _ = partition_data(args.dataset, args.partition, beta=args.beta, num_users=args.num_users, data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 2. Setup Generator Config
    nz = 256
    img_size = 32
    nc = 3
    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        img_size = 32
        nc = 3
    elif args.dataset in ['mnist', 'fmnist']:
        img_size = 28
        nc = 1
    elif args.dataset in ['tiny', 'eurosat']:
        img_size = 64
        nc = 3
    elif args.dataset in ['nwpu', 'siri-whu']:
        img_size = 224
        nc = 3
    
    img_size_tuple = (nc, img_size, img_size)
    
    # Generator Init
    generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=nc).cuda()

    num_class = 10
    if args.dataset == 'cifar100': num_class = 100
    elif args.dataset == 'tiny': num_class = 200
    elif args.dataset == 'nwpu': num_class = 45
    elif args.dataset == 'eurosat': num_class = 10
    elif args.dataset == 'siri-whu': num_class = 12

    results = {}
    
    # 3. Main Loop
    for ckpt_info in checkpoints:
        step = ckpt_info['step']
        print(f"\n>>> Processing Step: {step} <<<")
        
        client_weights = []
        try:
            if ckpt_info['type'] == 'individual':
                for fpath in ckpt_info['paths']:
                    client_weights.append(torch.load(fpath))
            else:
                # Monolithic
                loaded = torch.load(ckpt_info['path'])
                if isinstance(loaded, list):
                    client_weights = loaded
                elif isinstance(loaded, dict):
                    client_weights = [loaded]
                else:
                    print(f"Unknown format in {ckpt_info['path']}")
                    continue
        except Exception as e:
            print(f"Error loading weights for step {step}: {e}")
            continue

        print(f"Loaded {len(client_weights)} teacher models for step {step}.")

        # Reconstruct Teachers
        teacher_list = []
        for w in client_weights:
            net = get_model(args)
            net.load_state_dict(w)
            teacher_list.append(net)
        
        ensemble_model = Ensemble(teacher_list).cuda()
        ensemble_model.eval()

        # Student (init from scratch each time to compare fairly)
        student = get_model(args).cuda()
        student.train()
        optimizer = build_optimizer(student.parameters(), args)
        
        # Synthesizer
        # Re-init synthesizer to reset generator state or just use fresh one
        # Using 'ours' alg default
        synthesizer = AdvSynthesizer(ensemble_model, teacher_list, student, generator, None,
                                     nz=nz, num_classes=num_class, img_size=img_size_tuple,
                                     iterations=args.g_steps, lr_g=args.lr_g,
                                     synthesis_batch_size=args.synthesis_batch_size, 
                                     sample_batch_size=args.batch_size,
                                     adv=args.adv, bn=args.bn, oh=args.oh,
                                     save_dir="temp_syn", dataset=args.dataset, alg='ours')

        # Distill Loop
        curve = []
        print(f"Distilling student for {args.distill_epochs} epochs...")
        for epoch in range(args.distill_epochs):
            # 1. Synthesize
            synthesizer.gen_data(epoch)
            
            # 2. Train Student
            kd_train(synthesizer, [student, teacher_list], None, optimizer, ensemble_model)
            
            # 3. Evaluate
            acc, _ = test(student, test_loader)
            curve.append(acc)
            # print(f" [Ep {epoch+1}] Acc: {acc:.2f}%")
        
        print(f"Final Acc for Step {step}: {curve[-1]:.2f}%")
        results[f"Step {step}"] = curve

    # 4. Visualization / Summary
    print("\n================ Results Summary ================")
    
    # Save results for re-plotting
    import json
    save_data = {
        "dataset": args.dataset,
        "results": results,
        "steps": args.steps,
        "args": vars(args)
    }
    json_path = f"distill_comparison_{args.dataset}.json"
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"Results saved to {json_path}")

    plt.figure(figsize=(10, 6))
    
    colors = ['#CC5651', '#DC8A00', '#038255', '#76009C']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (name, curve) in enumerate(results.items()):
        final_acc = curve[-1] if curve else 0
        print(f"Checkpoint: {name} | Final Student Acc: {final_acc:.2f}%")
        plt.plot(range(1, len(curve)+1), curve, label=f"{name} (Final: {final_acc:.2f}%)", color=colors[i % len(colors)], marker=markers[i % len(markers)])
    
    plt.xlabel('Distillation Epochs')
    plt.ylabel('Student Accuracy')
    plt.title(f'Student Distillation Progress ({args.dataset})')
    plt.legend()
    plt.grid(True)
    
    out_png = f"distill_comparison_{args.dataset}.png"
    plt.savefig(out_png)
    print(f"Comparison plot saved to {out_png}")
    print("=================================================")

if __name__ == "__main__":
    main()
