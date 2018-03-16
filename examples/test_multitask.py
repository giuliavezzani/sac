import argparse

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator

import doodad as dd
from sac.algos.sac_multitask import SACMultiTask
from sac.envs.gym_env import GymEnv
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp
from sac.policies.gmm import GMMPolicyMultiTask
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunctionMultiHead, NNVFunctionMultiHead
from sac.envs.multitask.ant import AntEnvRandGoalRing
from sac.envs.multitask.point_mass import PointEnv


SHARED_PARAMS = {
    "seed": [1],
    "lr": 3E-4,
    "discount": 0.99,
    "tau": 0.01,
    "K": 4,
    'layer_sizes': 128,
    'layer_sizes_qf': 128,
    'layer_sizes_extra_qf': 128,
    "batch_size": 640,
    "max_pool_size": 1E6,
    "n_train_repeat": 1,
    "epoch_length": 1000,
    "snapshot_mode": 'last',
    "snapshot_gap": 100,
    "sync_pkl": True,
}


ENV_PARAMS = {
    'swimmer': { # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'swimmer-rllab',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'scale_reward': 100,
    },
    'hopper': { # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'scale_reward': 1,
    },
    'half-cheetah': { # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'scale_reward': 1,
        'max_pool_size': 1E7,
    },
    'walker': { # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'scale_reward': 3,
    },
    'ant': {  # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'reward_scale': 3,
    },
    'humanoid': { # 21 DoF
        'prefix': 'humanoid',
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'scale_reward': 3,
    },
    'point-mass': {  # 8 DoF
        'prefix': 'point-mass',
        'env_name': 'PointMass',
        'max_path_length': 100,
        'n_epochs': 10000,
        'reward_scale': 1,
    },
}
DEFAULT_ENV = 'swimmer'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())

def parse_args():
    dd_log_dir = dd.get_args('log_dir', '../../data/sac/multitasl/tests/')
    dd_file_goal = dd.get_args('file_goals', '/home/giulia/NIPS/sac/sac/envs/goals/ant_10_goals.pkl')
    dd_file_env = dd.get_args('file_env', '/home/giulia/NIPS/sac/low_gear_ratio_ant.xml')
    dd_save_file = dd.get_args('save_file', '../../data/rewards/')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default='ant')
    parser.add_argument('--exp_name',type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=dd_log_dir)
    parser.add_argument('--file_goals', type=str, default=dd_file_goal)
    parser.add_argument('--file_env', type=str, default=dd_file_env)
    parser.add_argument('--save_file', type=str, default=dd_save_file)
    args = parser.parse_args()

    return args

def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = SHARED_PARAMS
    params.update(env_params)
    file_params = {'file_goals': args.file_goals}
    params.update(file_params)
    file_params = {'file_env': args.file_env}
    params.update(file_params)
    file_params = {'log_dir': args.log_dir}
    params.update(file_params)
    file_params = {'save_file': args.save_file}
    params.update(file_params)

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def run_experiment(variant):
    #if variant['env_name'] == 'humanoid-rllab':
    #    from rllab.envs.mujoco.humanoid_env import HumanoidEnv
    #    env = normalize(HumanoidEnv())
    #elif variant['env_name'] == 'swimmer-rllab':
    #    from rllab.envs.mujoco.swimmer_env import SwimmerEnv
    #    env = normalize(SwimmerEnv())
    #else:
    #    env = normalize(GymEnv(variant['env_name']))
    envs = []
    num_tasks = 10
    policies = []
    qfs = []
    kernel_fns = []
    pools = []

    print('LOG_DIR', variant['log_dir'])


    base_kwargs = dict(
        min_pool_size=variant['max_path_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
        num_tasks=num_tasks,
    )

    M = variant['layer_sizes_qf']
    MP = variant['layer_sizes']
    N = variant['layer_sizes_extra_qf']

    for task in range(num_tasks):
        if variant['env_name'] == 'Ant-v1':
            envs.append(AntEnvRandGoalRing(file_goals=variant['file_goals'], file_env=variant['file_env'], goal=task))
        elif variant['env_name'] == 'PointMass':
            envs.append(PointEnv(file_goals=variant['file_goals'], file_env=variant['file_env'], goal=task))

        #qfs.append(NNQFunction(env_spec=envs[task].spec, hidden_layer_sizes=[M, M], task=task))
        #kernel_fns.append(adaptive_isotropic_gaussian_kernel)

        pools.append(SimpleReplayBuffer(
            env_spec=envs[task].spec,
            max_replay_buffer_size=1e6,
    ))


    qf = NNQFunctionMultiHead(env_spec=envs[0].spec, hidden_layer_sizes= [M, M], hidden_layer_sizes_extra = [N, N, N, N], num_tasks=num_tasks)
    #qf = NNQFunctionMultiHead(env_spec=envs[0].spec,  hidden_layer_sizes_extra = [N, N], num_tasks=num_tasks)

    ## To adapt also V function
    vf = NNVFunctionMultiHead(env_spec=envs[0].spec, hidden_layer_sizes=[M, M], hidden_layer_sizes_extra = [N, N, N, N], num_tasks=num_tasks)
    #vf = NNVFunctionMultiHead(env_spec=envs[0].spec,  hidden_layer_sizes_extra = [N, N], num_tasks=num_tasks)

    for task in range(num_tasks):
        policies.append( GMMPolicyMultiTask(env_spec=envs[task].spec, K=variant['K'], hidden_layer_sizes=[M, M], qf=qf, reg=0.001,task=task,
        ))

    algorithm = SACMultiTask(
        base_kwargs=base_kwargs,
        envs=envs,
        policies=policies,
        pools=pools,
        qf=qf,
        vf=vf,
        lr=variant['lr'],
        scale_reward=variant['reward_scale'],
        discount=variant['discount'],
        tau=variant['tau'],
        num_tasks=num_tasks,
        save_full_state=False,
        batch_size=variant['batch_size'],
        save_file=variant['save_file'],
    )

    algorithm.train()


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=variant['prefix'] + '-' + args.exp_name + '-' + str(i).zfill(2),
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

if __name__ == '__main__':
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
