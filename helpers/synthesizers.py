import copy
from abc import ABC, abstractclassmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import ipdb
from kornia import augmentation
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils
from helpers.utils import ImagePool, DeepInversionHook, average_weights, kldiv, js_div
from torch.autograd import Variable
import numpy as np
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)


class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)

class Ensemble_A(torch.nn.Module):
    def __init__(self, model_list, teacher_device: Optional[str] = None):
        super(Ensemble_A, self).__init__()
        self.models = model_list
        self.teacher_device = teacher_device  # If set, all teachers are on this device

    def forward(self, x):
        logits_total = 0
        input_device = x.device
        for i in range(len(self.models)):
            # Ensure input is on the same device as the model
            if self.teacher_device is not None:
                model_device = self.teacher_device
            else:
                model_device = next(self.models[i].parameters()).device
            logits = self.models[i](x.to(model_device))
            logits_total += logits.to(input_device)
        logits_e = logits_total / len(self.models)

        return logits_e

class Ensemble_M(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble_M, self).__init__()
        self.models = model_list

    def forward(self, x):
        logits_list = []
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_list.append(logits)
        # 把list送入到mlp中
        logits_e = torch.stack((logits_list[0], logits_list[1],
                                logits_list[2], logits_list[3], logits_list[4]))
        data = logits_e.permute(1, 2, 0)  # [bs,num_cls,5]
        return data


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    """
    输入为logits,输出为weight矩阵
    [[bs,num_cls]*5]=[bs,num_cls,5]  ----> [[bs,1]*5], 搭配上[bs,num_cls]
    给每个样本配上一个权重，应该为[bs,1]*[bs,num_cls]
    """

    def __init__(self, dim_in=500, dim_hidden=100, dim_out=5):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        # --------------------
        bs = x.shape[0]  # x:[bs,num_cls,5]
        ori_data = x.permute(2, 0, 1)  # [5,bs,num_cls]
        logits_total = 0
        # --------------------
        x = x.reshape(-1, x.shape[2] * x.shape[1])  # [bs,num_cls*5]
        x = self.layer_input(x)  # [bs,dim_hidden]
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)  # [bs, dim_out]
        # ----------
        y_prob = F.softmax(x, dim=1)  # [bs, 5]
        for i in range(5):
            tmp = y_prob[:, i].reshape(bs, -1).cuda()
            logits = ori_data[i].mul(tmp)  # [bs,10] [bs,5]取第i列，对应点乘
            logits_total += logits
        logits_final = logits_total / 5.0
        return logits_final

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer, y_input=None, diversity_loss_type=None):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        if diversity_loss_type == 'div2':
            y_input_dist = self.pairwise_distance(y_input, how='l1')
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        if diversity_loss_type == 'div2':
            return torch.exp(-torch.mean(noise_dist * layer_dist * torch.exp(y_input_dist)))
        else:
            return torch.exp(-torch.mean(noise_dist * layer_dist))
        
class AdvSynthesizer():
    def __init__(self, teacher, model_list, student, generator1, generator2, nz, num_classes, img_size,
                 iterations, lr_g,
                 synthesis_batch_size, sample_batch_size,
                 adv, bn, oh, save_dir, dataset, alg,
                 use_amp: bool = False,
                 teacher_device: Optional[str] = None,
                 student_device: Optional[str] = None):
        super(AdvSynthesizer, self).__init__()
        self.student = student
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.save_dir = save_dir
        self.data_pool = ImagePool(root=self.save_dir,cls_num=num_classes)
        self.data_iter = None
        self.teacher = teacher
        self.dataset = dataset
        self.alg = alg
        
        # Large model distillation settings (AMP + Multi-GPU)
        self.use_amp = use_amp
        self.teacher_device = teacher_device
        self.student_device = student_device if student_device else 'cuda:0'
        self.scaler = GradScaler() if use_amp else None
        
        if use_amp:
            print(f"[AdvSynthesizer] AMP enabled with GradScaler")
        if teacher_device:
            print(f"[AdvSynthesizer] Teacher models on {teacher_device}, Student/Generator on {self.student_device}")

        # Move generator to appropriate device (generators are already on the correct device from loop_df_fl.py)
        # Just set to train mode
        self.generator1 = generator1.train()
        self.generator2 = generator2.train() if generator2 != None else None
        self.model_list = model_list

        self.div = 1.0
        self.sim = 0.25
        self.cd = 0.25
        self.diversity_loss = DiversityLoss(metric='l2')

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        # =======================
        if not ("cifar" in dataset):
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])

        # datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        # if len(datasets) != 0:
        #     self.data_loader = torch.utils.data.DataLoader(
        #         datasets, batch_size=self.sample_batch_size, shuffle=True,
        #         num_workers=4, pin_memory=True, )
    def random_choice_y(self, batch_size, label_num=[]):
        if len(label_num) > 0:
            _label_num = label_num.sum(axis=1)
            label_pop = _label_num / sum(_label_num)
            label_cumpop = np.cumsum(label_pop)
            label_cumpop = np.insert(label_cumpop, 0, 0.0)
            r_bz = np.random.random(batch_size)
            y = []
            for r in r_bz:
                for i in range(len(label_cumpop) - 1):
                    if r >= label_cumpop[i] and r <= label_cumpop[i + 1]:
                        y.append(i)
        else:
            y = np.random.choice([i for i in range(self.num_classes)], batch_size)
        return y
    
    def set_teacher(self, teacher, model_list, step, save_dir):
        self.teacher = teacher
        self.model_list = model_list
        self.save_dir = save_dir + "/" + str(step)
        # self.data_pool = ImagePool(root=self.save_dir,cls_num=self.num_classes)

    def gen_data(self, cur_ep):
        self.synthesize(self.teacher, cur_ep)

    def get_data(self):
        datasets = self.data_pool.get_dataset(transform=self.transform)  # 获取程序运行到现在所有的图片
        self.data_loader = torch.utils.data.DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
        return self.data_loader

    def loss_g(self,t_out, dual_t_out, s_out, targets,hooks,global_view,z):

        loss_oh = F.cross_entropy(t_out, targets)  # ce_loss

        mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
        loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation
        
        M = len(self.model_list)
        pred_list = []
        loss_sim = 0.0
        input_device = global_view.device
        for teacher in self.model_list:
            teacher.eval()
            # Move input to teacher's device
            teacher_device = next(teacher.parameters()).device
            teacher_out = teacher(global_view.to(teacher_device))
            # Move output back to input device for consistency
            pred_list.append(teacher_out.to(input_device))

        # print("loss_div: ", loss_div)
        # print("loss_oh: ", loss_oh)
        # print("loss_adv: ", loss_adv)
        # print("loss_sim: ", loss_sim)
        if self.alg == "DENSE":
            # Ensure summation works across devices if using Model Parallelism
            if len(hooks) > 0:
                first_device = hooks[0].r_feature.device
                loss_bn = sum([h.r_feature.to(first_device) for h in hooks])
            else:
                loss_bn = 0
                
            loss_dense = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv 
            return loss_dense
        elif self.alg == "FedDF":
            loss_FedDF = self.oh * loss_oh
            return loss_FedDF
        elif self.alg == "FedFTG":
            loss_md = 0.0
            device = s_out.device
            for i in range(M):
                loss_md += -(kldiv(s_out, pred_list[i].to(device), reduction='none').sum(
                        1)).mean()
                # loss_md += -torch.abs(s_out - pred_list[i].detach()).sum(1).mean()
            loss_md = loss_md / M
            loss_div = self.diversity_loss(noises=z, layer=global_view).to(device)
            loss_FedFTG = self.oh * loss_oh + self.adv * loss_md + self.div * loss_div
            return loss_FedFTG
        elif self.alg == "DFRD":
            mask_dfrd = (s_out.max(1)[1] != targets ).float() * (t_out.max(1)[1] == targets ).float()
            loss_trans = -(kldiv(s_out, t_out, reduction='none').sum(
                        1) * mask_dfrd).mean() 
            loss_div = self.diversity_loss(noises=z, layer=global_view).to(s_out.device)
            loss_DFRD = self.oh * loss_oh + self.div * loss_div + self.adv * loss_trans
            return loss_DFRD
        elif self.alg == "DFDG":
            mask_dfdg = (s_out.max(1)[1] != targets ).float() * (t_out.max(1)[1] == targets ).float()
            loss_trans = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask_dfdg).mean() 
            loss_div = self.diversity_loss(noises=z, layer=global_view).to(s_out.device)
            loss_cd = -(kldiv(t_out, dual_t_out, reduction='none').sum(1)).mean()
            loss_DFDG = self.oh * loss_oh + self.div * loss_div + self.adv * loss_trans
            return loss_DFDG
        else :
            # Ensure summation works across devices if using Model Parallelism
            if len(hooks) > 0:
                first_device = hooks[0].r_feature.device
                loss_bn = sum([h.r_feature.to(first_device) for h in hooks])
            else:
                loss_bn = 0
                
            # Compute loss_sim with cross-device handling
            # Use device of first prediction as target for comparison
            # Or use CPU? Use GPU0 (pred_list[0].device usually)
            base_device = pred_list[0].device
            for i in range(M):
                p_i = pred_list[i].to(base_device)
                for j in range(i+1,M):
                    p_j = pred_list[j].to(base_device)
                    loss_sim += kldiv(p_i, p_j, reduction='none').sum(1).mean()
            loss_sim /= (M*(M-1)/2)
            loss_ours = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.sim * loss_sim
            return loss_ours
    def synthesize(self, net, cur_ep):
        net.eval()
        best_cost = 1e6
        best_inputs = None
        
        # Use appropriate device based on settings
        gen_device = self.student_device if self.teacher_device else 'cuda'
        
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=gen_device)
        z.requires_grad = True
        targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,), device=gen_device)
        targets = targets.sort()[0]
        y = F.one_hot(targets, num_classes=self.num_classes)
        y = y.float()
        reset_model(self.generator1)
        if self.generator2 != None:
            reset_model(self.generator2)

        hooks = []
        #############################################
        dim_in = 500 if "cifar100" == self.dataset else 50
        net = Ensemble_A(self.model_list, teacher_device=self.teacher_device)
        net.eval()
        # net_mlp = MLP(dim_in).cuda()
        # net_mlp.train()
        # optimizer_mlp = torch.optim.SGD(net_mlp.parameters(), lr=0.01,
        #                                 momentum=0.9)
        #############################################
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m))

        # generator1
        optimizer = torch.optim.Adam([{'params': self.generator1.parameters()}, {'params': [z]}], self.lr_g,
                                betas=[0.5, 0.999])
        self.generator1.train()
        if self.generator2 != None:
            self.generator2.eval()
        
        for it in range(self.iterations):
            self.generator1.zero_grad()
            optimizer.zero_grad()
            
            # Use AMP if enabled
            if self.use_amp:
                with autocast():
                    inputs = self.generator1(z)  # bs,nz
                    global_view, _ = self.aug(inputs)  # crop and normalize
                    t_out = net(global_view)
                    dual_t_out = None
                    if self.generator2 != None:
                        dual_inputs = self.generator2(z)
                        dual_global_view, _ = self.aug(dual_inputs)
                        dual_t_out = net(dual_global_view)
                    
                    s_out = self.student(global_view.to(self.student_device) if self.teacher_device else global_view)
                    loss = self.loss_g(t_out, dual_t_out, s_out, targets, hooks, global_view, z)
                
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.float()  # Ensure FP32 for saving
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Original non-AMP path
                inputs = self.generator1(z)  # bs,nz
                global_view, _ = self.aug(inputs)  # crop and normalize
                t_out = net(global_view)
                dual_t_out = None
                if self.generator2 != None:
                    dual_inputs = self.generator2(z)
                    dual_global_view, _ = self.aug(dual_inputs)
                    dual_t_out = net(dual_global_view)

                s_out = self.student(global_view)
                loss = self.loss_g(t_out, dual_t_out, s_out, targets, hooks, global_view, z)
                
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
                    
                with torch.autograd.detect_anomaly(True):
                    loss.backward()
                for name, param in self.generator1.named_parameters():
                    if torch.isnan(param.grad).any():
                        print(f"Detected NaN in gradient for {name}")
                optimizer.step()
            # for state in optimizer.state.values():
            #     if isinstance(state, dict):  # 检查状态字典
            #         for key, value in state.items():
            #             if torch.isnan(value).any() or torch.isinf(value).any():
            #                 print(f"Detected NaN or Inf in optimizer state: {key}")

            # optimizer_mlp.step()
            # t.set_description('iters:{}, loss:{}'.format(it, loss.item()))
        # vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)
        self.data_pool.add(best_inputs, targets.cpu())

        # generator2
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=gen_device)
        z.requires_grad = True
        best_cost = 1e6
        best_inputs = None
        if self.generator2 != None:
            self.generator2.train()
            self.generator1.eval()
            optimizer = torch.optim.Adam([{'params': self.generator2.parameters()}, {'params': [z]}], self.lr_g,
                                betas=[0.5, 0.999])
            
            for it in range(self.iterations):
                self.generator2.zero_grad()
                optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        inputs = self.generator2(z)
                        global_view, _ = self.aug(inputs)
                        t_out = net(global_view)
                        
                        dual_inputs = self.generator1(z)
                        dual_global_view, _ = self.aug(dual_inputs)
                        dual_t_out = net(dual_global_view)
                        
                        s_out = self.student(global_view.to(self.student_device) if self.teacher_device else global_view)
                        loss = self.loss_g(t_out, dual_t_out, s_out, targets, hooks, global_view, z)
                    
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.data.float()
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    inputs = self.generator2(z)
                    global_view, _ = self.aug(inputs)
                    t_out = net(global_view)

                    dual_inputs = self.generator1(z)
                    dual_global_view, _ = self.aug(dual_inputs)
                    dual_t_out = net(dual_global_view)

                    s_out = self.student(global_view)
                    loss = self.loss_g(t_out, dual_t_out, s_out, targets, hooks, global_view, z)
                    
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.data
                        
                    with torch.autograd.detect_anomaly(True):
                        loss.backward()
                    for name, param in self.generator2.named_parameters():
                        if torch.isnan(param.grad).any():
                            print(f"Detected NaN in gradient for {name}")
                    optimizer.step()
                # for state in optimizer.state.values():
                #     if isinstance(state, dict):  # 检查状态字典
                #         for key, value in state.items():
                #             if torch.isnan(value).any() or torch.isinf(value).any():
                #                 print(f"Detected NaN or Inf in optimizer state: {key}")

                # optimizer_mlp.step()
                # t.set_description('iters:{}, loss:{}'.format(it, loss.item()))
        
        # vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)

        # save best inputs and reset data iter
        if best_inputs != None:
            self.data_pool.add(best_inputs, targets.cpu())  # 生成了一个batch的数据
