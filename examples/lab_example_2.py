
"""
Runs one instance of the deepmind lab environment and optimizes using DQN algorithm.
Can use a GPU for the agent (applies to both sample and train). No parallelism
employed, so everything happens in one python process; can be easier to debug.

The kwarg snapshot_mode="last" to logger context will save the latest model at
every log point (see inside the logger for other options).

In viskit, whatever (nested) key-value pairs appear in config will become plottable
keys for showing several experiments.  If you need to add more after an experiment,
use rlpyt.utils.logging.context.add_exp_param().

"""
from rlpyt.samplers.serial.sampler import SerialSampler

from rlpyt.envs.lab import DeepmindLabEnv, LabTrajInfo
from rlpyt.algos.dqn.r2d1 import R2D1
from rlpyt.agents.dqn.atari.atari_r2d1_agent import AtariR2d1Agent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

from rlpyt.experiments.configs.atari.dqn.atari_r2d1 import configs
from rlpyt.utils.launching.affinity import make_affinity

def build_and_train(level="nav_maze_random_goal_01", run_ID=0, cuda_idx=None):
    config = configs['r2d1']
    config['eval_env'] = dict(level=level)
    config['env'] = dict(level=level)

    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=4,  # Use 16 cores across all experiments.
        n_gpu=1,  # Use 8 gpus across all experiments.
        hyperthread_offset=6,  # If machine has 24 cores.
        n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        gpu_per_run=1,  # How many GPUs to parallelize one run across.
    )

    # sampler = GpuSampler(
    #     EnvCls=DeepmindLabEnv,
    #     env_kwargs=config['env'],
    #     eval_env_kwargs=config['eval_env'],
    #     CollectorCls=GpuWaitResetCollector,
    #     TrajInfoCls=LabTrajInfo,
    #     **config["sampler"]
    # )
    sampler = SerialSampler(
        EnvCls=DeepmindLabEnv,
        env_kwargs=config['env'],
        eval_env_kwargs=config['env'],
        batch_T=16,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = R2D1(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariR2d1Agent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "lab_dqn_" + level
    log_dir = "lab_example_2"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--level', help='Deepming level', default='nav_maze_random_goal_01')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=str, default=None)
    args = parser.parse_args()
    build_and_train(
        level=args.level,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
