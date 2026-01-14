import zipfile
import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models, utils
import random
from PIL import Image


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()
        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

def load_data(dataset, data_dir):
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]),download=True)
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]),download=True)
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]),download=True)
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]),download=True)
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(data_dir, split="train",
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]),download=True)
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]),download=True)
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,download=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                             ]))
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]),download=True)
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]),download=True)
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]),download=True)

    elif dataset == "tiny":
        data_transforms = {
        "train": transforms.Compose([transforms.RandomCrop(64, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        train_dataset = TinyImageNet(data_dir, train=True, transform=data_transforms["train"])
        test_dataset = TinyImageNet(data_dir, train=False, transform=data_transforms["val"])
    elif dataset == "eurosat":
        data_transforms = {
            "train": transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        # Only use local folder under data_dir, do not use torchvision.EuroSAT
        # Prefer the explicitly requested folder name 'Eurosat_rgb', but accept common variants
        p = os.path.join(data_dir, 'EuroSAT_RGB')
        if os.path.isdir(p):
            dataset_path = p
        if dataset_path is None:
            raise RuntimeError(
                f"EuroSAT dataset folder not found. Please place it at {os.path.join(data_dir, 'Eurosat_rgb')} "
                f"(or one of {candidate_dirs}). Download via torchvision is disabled per request.")

        # Load with ImageFolder
        train_dataset_full = datasets.ImageFolder(dataset_path, transform=data_transforms["train"]) 
        test_dataset_full = datasets.ImageFolder(dataset_path, transform=data_transforms["val"]) 

        num_len = len(train_dataset_full)
        indices = np.arange(num_len)
        np.random.shuffle(indices)
        split = int(num_len * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]

        train_dataset = torch.utils.data.Subset(train_dataset_full, train_idx)
        test_dataset = torch.utils.data.Subset(test_dataset_full, test_idx)

        all_targets = np.array(train_dataset_full.targets)
        y_train = all_targets[train_idx]
        y_test = all_targets[test_idx]
        X_train, X_test = [], []

    elif dataset == "nwpu":
        data_transforms = {
            "train": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        target_root = os.path.join(data_dir, 'NWPU-RESISC45')
        zip_path = '/data/ccy/datasets/NWPU-RESISC45.zip'
        
        # Check if dataset exists, if not try to unzip
        if not os.path.exists(os.path.join(target_root, 'NWPU-RESISC45')):
            if os.path.exists(zip_path):
                print(f"Extracting {zip_path} to {data_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
            elif not os.path.exists(target_root):
                 raise RuntimeError(f"Dataset not found at {target_root} and no zip at {zip_path}")
        
        # The structure is NWPU-RESISC45/NWPU-RESISC45/<class>
        dataset_path = os.path.join(target_root, 'NWPU-RESISC45')
        
        train_dataset_full = datasets.ImageFolder(dataset_path, transform=data_transforms["train"]) 
        test_dataset_full = datasets.ImageFolder(dataset_path, transform=data_transforms["val"])
        
        num_len = len(train_dataset_full)
        indices = np.arange(num_len)
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(num_len * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]
        
        train_dataset = torch.utils.data.Subset(train_dataset_full, train_idx)
        test_dataset = torch.utils.data.Subset(test_dataset_full, test_idx)
        
        all_targets = np.array(train_dataset_full.targets)
        y_train = all_targets[train_idx]
        y_test = all_targets[test_idx]
        X_train, X_test = [], []

    elif dataset == "siri-whu":
        data_transforms = {
            "train": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        dataset_path = '/data/ccy/datasets/SIRI-WHU_Google_image'
        if not os.path.exists(dataset_path):
             print(f"Warning: {dataset_path} not found. Checking data_dir...")
             dataset_path = os.path.join(data_dir, 'SIRI-WHU_Google_image')
        
        # Specific structure handling for SIRI-WHU: contains 12class_tif folder
        if os.path.exists(os.path.join(dataset_path, '12class_tif')):
            dataset_path = os.path.join(dataset_path, '12class_tif')

        if not os.path.exists(dataset_path):
            raise RuntimeError(f"SIRI-WHU dataset not found at {dataset_path}")

        train_dataset_full = datasets.ImageFolder(dataset_path, transform=data_transforms["train"]) 
        test_dataset_full = datasets.ImageFolder(dataset_path, transform=data_transforms["val"])
        
        num_len = len(train_dataset_full)
        indices = np.arange(num_len)
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(num_len * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]
        
        train_dataset = torch.utils.data.Subset(train_dataset_full, train_idx)
        test_dataset = torch.utils.data.Subset(test_dataset_full, test_idx)
        
        all_targets = np.array(train_dataset_full.targets)
        y_train = all_targets[train_idx]
        y_test = all_targets[test_idx]
        X_train, X_test = [], []

    else:
        raise NotImplementedError
    if dataset == "svhn":
        X_train, y_train = train_dataset.data, train_dataset.labels
        X_test, y_test = test_dataset.data, test_dataset.labels
    elif dataset == "tiny":
        X_train, y_train, X_test, y_test = [], [], [], []
        print(len(train_dataset))
        for idx in range(len(train_dataset)):
            if(idx % 1000 == 0):
                print(idx)
            img, label = train_dataset[idx]
            # X_train.append(img)   
            y_train.append(label)
        print(len(test_dataset))
        for idx in range(len(test_dataset)):
            if(idx % 1000 == 0):
                print(idx)
            img, label = test_dataset[idx]
            # X_test.append(img)
            y_test.append(label)
    elif dataset in ["eurosat", "nwpu", "siri-whu"]:
        # Already prepared y_train/y_test above using ImageFolder/Subset; X_* are placeholders
        pass
    else:
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
    if "cifar10" in dataset or dataset == "svhn" or dataset == "tiny" or dataset == "eurosat" or dataset == "nwpu" or dataset == "siri-whu":
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        X_train = X_train.data.numpy()
        y_train = y_train.data.numpy()
        X_test = X_test.data.numpy()
        y_test = y_test.data.numpy()

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset




def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, partition, beta=0.4, num_users=5, data_dir='./data'):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset, data_dir)
    data_size = y_train.shape[0]

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the labelq
                # random [0.5963643 , 0.03712018, 0.04907753, 0.1115522 , 0.2058858 ]
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts
