import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='cifar10', help="Dataset to train on")
    parser.add_argument('--num_agents', type=int, default=10, help="Number of agents")
    parser.add_argument('--agent_frac', type=float, default=1, help="Fraction of agents per round")
    parser.add_argument('--rounds', type=int, default=30, help="Number of communication rounds")
    parser.add_argument('--local_ep', type=int, default=2, help="Number of local epochs per agent")
    parser.add_argument('--bs', type=int, default=256, help="Local batch size")
    parser.add_argument('--client_lr', type=float, default=0.001, help="Client learning rate")
    parser.add_argument('--client_moment', type=float, default=0.9, help="Client momentum")
    parser.add_argument('--server_lr', type=float, default=1, help="Server learning rate for signSGD")
    parser.add_argument('--base_class', type=int, default=5, help="Base class for backdoor attack")
    parser.add_argument('--target_class', type=int, default=7, help="Target class for backdoor attack")
    parser.add_argument('--poison_frac', type=float, default=0.5, help="Fraction of dataset to poison")
    parser.add_argument('--pattern_type', type=str, default='plus', help="Poison pattern type")
    parser.add_argument('--robustLR_threshold', type=int, default=0, help="Threshold for robust learning rate")
    parser.add_argument('--clip', type=float, default=0, help="Weight clip value")
    parser.add_argument('--noise', type=float, default=0, help="Noise ratio for aggregation")
    parser.add_argument('--snap', type=int, default=1, help="Inference frequency")
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help="Device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")

    # Add this line to fix the error
    parser.add_argument('--num_corrupt', type=int, default=0, help="Number of corrupt agents")
    parser.add_argument('--aggr', type=str, default='avg', help="Aggregation method (e.g., 'avg', 'comed', 'sign')")

    return parser.parse_args()



