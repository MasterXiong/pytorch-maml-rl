import gym
import torch
import json
import os
import yaml
from tqdm import trange
import numpy as np
import pickle

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.fast_lr is not None:
        config['fast-lr'] = args.fast_lr
    print (config['fast-lr'])

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    print (policy.state_dict()['mu.bias'])
    if args.init_model is not None:
        policy.load_state_dict(torch.load(os.path.join(args.init_model, 'policy.th')))
    print (policy.state_dict()['mu.bias'])
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device, 
                           task=args.task)

    num_iterations = 0
    if args.num_batches is not None:
        meta_train_batches = args.num_batches
    else:
        meta_train_batches = config['num-batches']

    learning_curve = []
    for batch in trange(meta_train_batches):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))

        valid_returns = np.array(get_returns(valid_episodes))
        print (valid_returns.mean(), valid_returns.shape)
        learning_curve.append([batch, valid_returns.mean(), valid_returns.std()])
        with open(os.path.join(args.output_folder, 'learning_curve.pkl'), 'wb') as f:
            pickle.dump([learning_curve], f)

        # Save policy
        if args.output_folder is not None:
            if batch % 10 == 0:
                with open(os.path.join(args.output_folder, 'policy_{}.th'.format(batch)), 'wb') as f:
                    torch.save(policy.state_dict(), f)
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)

    if meta_train_batches == 0:
        with open(policy_filename, 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--num_batches', type=int, default=None,
        help='the number of batches for meta-training')
    misc.add_argument('--init_model', type=str, default=None,
        help='the model used for initialization')
    misc.add_argument('--fast_lr', type=float, default=None,
        help='learning rate for inner update')
    misc.add_argument('--task', type=str, default=None,
        help='the task name (only useful for cheetah-dir-uni)')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
