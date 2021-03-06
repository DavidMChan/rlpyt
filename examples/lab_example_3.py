
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
from rlpyt.envs.lab import DeepmindLabEnv
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context


def build_and_train(level="nav_maze_random_goal_01", run_ID=0, cuda_idx=None):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(8)))
    sampler = SerialSampler(
        EnvCls=DeepmindLabEnv,
        env_kwargs=dict(level=level),
        eval_env_kwargs=dict(level=level),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=5,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = PPO()
    agent = AtariFfAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=affinity,
    )
    config = dict(level=level)
    name = "lab_ppo"
    log_dir = "lab_example_3"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--level', help='Deepmind level', default='nav_maze_random_goal_01')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        level=args.level,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
