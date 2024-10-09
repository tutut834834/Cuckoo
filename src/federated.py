import torch
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import torch.nn as nn
from datetime import datetime

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    utils.print_exp_details(args)

    # Logging and recorders
    file_name = f"""time_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_clip_val_{args.clip}_noise_std_{args.noise}""" + \
                f"""_aggr_{args.aggr}_s_lr_{args.server_lr}_num_cor_{args.num_corrupt}_pttrn_{args.pattern_type}"""
    writer = SummaryWriter('logs/' + file_name)
    cum_poison_acc_mean = 0

    # Early stopping settings
    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0

    # Arrays to store intermediate values
    val_acc_arr = []
    val_loss_arr = []
    poison_acc_arr = []
    poison_loss_arr = []

    # Load datasets and user groups
    train_dataset, val_dataset = utils.get_datasets(args.data)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    if args.data != 'fedemnist':
        user_groups = utils.distribute_data(train_dataset, args)

    # Poison the validation dataset
    idxs = (val_dataset.targets == args.base_class).nonzero().flatten().tolist()
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # Initialize model, agents, and aggregator
    global_model = models.get_model(args.data).to(args.device)
    agents, agent_data_sizes = [], {}

    for _id in range(0, args.num_agents):
        agent = Agent(_id, args, train_dataset, user_groups[_id]) if args.data != 'fedemnist' else Agent(_id, args)
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

    n_model_params = len(parameters_to_vector(global_model.parameters()))
    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, writer)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # Training loop
    for rnd in tqdm(range(1, args.rounds + 1)):
        rnd_global_params = parameters_to_vector(global_model.parameters()).detach()
        agent_updates_dict = {}

        for agent_id in np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False):
            update = agents[agent_id].local_train(global_model, criterion)
            agent_updates_dict[agent_id] = update
            vector_to_parameters(copy.deepcopy(rnd_global_params), global_model.parameters())

        aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)

        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                
                # Append intermediate values to arrays
                val_acc_arr.append(val_acc)
                val_loss_arr.append(val_loss)

                # Save best model using early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(global_model.state_dict(), "best_model.pth")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at round {rnd}")
                        break

                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                
                # Append poison results to arrays
                poison_acc_arr.append(poison_acc)
                poison_loss_arr.append(poison_loss)

                cum_poison_acc_mean += poison_acc

                writer.add_scalar('Validation/Loss', val_loss, rnd)
                writer.add_scalar('Validation/Accuracy', val_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean / rnd, rnd)

    print("Training finished.")

    # Print arrays with intermediate values
    print("\nValidation Accuracies per Round:", val_acc_arr)
    print("Validation Losses per Round:", val_loss_arr)
    print("Poison Accuracies per Round:", poison_acc_arr)
    print("Poison Losses per Round:", poison_loss_arr)
