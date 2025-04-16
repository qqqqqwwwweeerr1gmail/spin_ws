

from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ppo-pyt-bench')
    # eg.add('env_name', 'CartPole-v0', '', True)
    eg.add('env_name', 'Acrobot-v1', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', 500)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)


# python -m spinup.run plot /home/ws/git/spin_ws/data/ppo-pyt-bench_acrobot-v1_hid64-64_relu/ppo-pyt-bench_acrobot-v1_hid64-64_relu_s90


# python -m spinup.run test_policy /home/ws/git/spin_ws/data/ppo-pyt-bench_acrobot-v1_hid64-64_relu/ppo-pyt-bench_acrobot-v1_hid64-64_relu_s90























