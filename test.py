import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange
import pickle

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers, 
                               task=args.task, 
                               mode='test')

    logs = {'tasks': []}
    train_returns, valid_returns = [[] for _ in range(args.gradient_steps)], []
    #train_info = {}
    for batch in trange(args.num_batches):
        print (args.fast_lr)
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        if args.meta_batch_size == 2:
            tasks = [{'direction': 1}, {'direction': -1}]
        print (tasks)
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=args.gradient_steps,
                                                        fast_lr=args.fast_lr,
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        logs['tasks'].extend(tasks)
        #for i in range(args.gradient_steps):
        #    train_returns[i].append(get_returns(train_episodes[i]))
        valid_returns.append(get_returns(valid_episodes))
        #for step in range(0, args.gradient_steps, 50):
        #    train_info[step] = [task._info_list[0] for task in train_episodes[step]]
        #with open('train_episodes.pkl', 'wb') as f:
        #    pickle.dump(train_episodes, f)

    #logs['train_returns'] = [np.concatenate(train_returns[i], axis=0) for i in range(args.gradient_steps)]
    # train_episodes is of shape [step_num, task_num, 10]
    logs['train_returns'] = train_episodes
    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)
    #logs['train_info'] = train_info
    #logs['valid_info'] = valid_episodes[0]._info_list[0]

    with open(args.output, 'wb') as f:
        np.savez(f, **logs)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
        help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')

    evaluation.add_argument('--gradient_steps', type=int, default=1,
        help='number of gradient update per task')
    evaluation.add_argument('--fast_lr', type=float, default=0.1,
        help='learning rate for inner update')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--task', type=str, default=None,
        help='the task name (only useful for cheetah-dir-uni)')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
