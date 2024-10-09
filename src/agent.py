import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
            if self.id < args.num_corrupt:
                utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)    
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
            if self.id < args.num_corrupt:
                utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, num_workers=self.args.num_workers, pin_memory=False)
        self.n_data = len(self.train_dataset)
        
    def local_train(self, global_model, criterion):
        initial_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        global_model.train()       

        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(global_model.parameters(), lr=self.args.client_lr, weight_decay=1e-4)

        # Cosine Annealing Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=30)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), labels.to(device=self.args.device, non_blocking=True)
                                               
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                nn.utils.clip_grad_norm_(global_model.parameters(), 10)  # Prevent exploding gradients
                optimizer.step()

                scheduler.step()
            
                if self.args.clip > 0:
                    with torch.no_grad():
                        local_model_params = parameters_to_vector(global_model.parameters())
                        update = local_model_params - initial_global_model_params
                        clip_denom = max(1, torch.norm(update, p=2) / self.args.clip)
                        update.div_(clip_denom)
                        vector_to_parameters(initial_global_model_params + update, global_model.parameters())
                            
        with torch.no_grad():
            update = parameters_to_vector(global_model.parameters()).double() - initial_global_model_params
            return update
  
    
