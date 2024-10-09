import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import defaultdict
from math import floor
import random
import cv2


class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])
        
    def classes(self):
        return torch.unique(self.targets)
    
    def __add__(self, other): 
        self.targets = torch.cat((self.targets, other.targets), 0)
        self.inputs = torch.cat((self.inputs, other.inputs), 0)
        return self
    
    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)
        
    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target


class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        data_idx = self.idxs[index]
        data, target = self.dataset[data_idx]
        return data, target


def distribute_data(dataset, args, n_classes=10, class_per_agent=10):
    if args.num_agents == 1:
        return {0: range(len(dataset))}
    
    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]
    
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
        
    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size    
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
           
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1

    return dict_users


def get_datasets(data):
    """ Returns train and test datasets """
    data_dir = '../data'

    if data == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    
    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir) 
    
    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)
    
    return train_dataset, test_dataset


def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True), labels.to(device=args.device, non_blocking=True)
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item() * outputs.shape[0]
                        
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    
    return avg_loss, (accuracy, per_class_accuracy)


def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1):
    all_idxs = (dataset.targets == args.base_class).nonzero().flatten().tolist()
    if data_idxs:
        all_idxs = list(set(all_idxs).intersection(data_idxs))
        
    poison_frac = 1 if poison_all else args.poison_frac    
    poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs)))
    
    for idx in poison_idxs:
        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        else:
            clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, args.data, pattern_type=args.pattern_type, agent_idx=agent_idx)
        
        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        
        dataset.targets[idx] = args.target_class
    
    return


def add_pattern_bd(x, dataset='cifar10', pattern_type='plus', agent_idx=-1):
    """ Adds a trojan pattern to the image """
    x = np.array(x.squeeze())
    
    if dataset == 'cifar10' and pattern_type == 'plus':
        start_idx = 5
        size = 6
        if agent_idx == -1:
            for d in range(0, 3):  
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx][d] = 0
            for d in range(0, 3):  
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i][d] = 0
        else:
            if agent_idx % 4 == 0:
                for d in range(0, 3):  
                    for i in range(start_idx, start_idx + size // 2 + 1):
                        x[i, start_idx][d] = 0
            elif agent_idx % 4 == 1:
                for d in range(0, 3):  
                    for i in range(start_idx + size // 2 + 1, start_idx + size + 1):
                        x[i, start_idx][d] = 0
            elif agent_idx % 4 == 2:
                for d in range(0, 3):  
                    for i in range(start_idx - size // 2, start_idx + size // 4 + 1):
                        x[start_idx + size // 2, i][d] = 0
            elif agent_idx % 4 == 3:
                for d in range(0, 3):  
                    for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i][d] = 0

    return x


def print_exp_details(args):
    print('======================================')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print('======================================')
