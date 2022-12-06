import sys

import torch
# Composer high level imports
from dm_control import composer

from MimicArmTask import MimicArm
from MimicArmTask import PandaArm
from eval_policy import eval_policy
from network import FeedForwardNN
from ppo import PPO
from utils import *
import json


def train(env, hyperparameters=None, actor_model='', critic_model=''):
    print(f"Training", flush=True)

    # Create a model for PPO
    model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

    # If an existing model is available, the training can be continued on the model
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded", flush=True)

    elif actor_model != '' or critic_model != '':
        print(
            f"Error: Either specify both actor/critic models or none at all. "
            f"We don't want to accidentally override anything!")
        sys.exit(0)

    else:
        print(f"Training from scratch...")

    # Train the model for a certain number of timesteps
    model.learn(total_timesteps=30_000)


def test(env, actor_model):
    print(f"Testing {actor_model}", flush=True)

    if actor_model == '':
        print(f"Didnt specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Dimnension of Observation space and Action space
    obs_dim = list(env.observation_spec()["panda/joint_positions"].shape)[1] + \
              list(env.observation_spec()["panda2/joint_positions"].shape)[1]
    act_dim = list(env.action_spec().shape)[0]

    # Build the policy
    policy = FeedForwardNN(in_dim=obs_dim, out_dim=act_dim, act_min=env.action_spec().minimum[0],
                           act_max=env.action_spec().maximum[0])

    # Load the model of the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate policy and return the result in a dictionary
    log = eval_policy(policy=policy, env=env, render=False, num_of_eps=50)

    return log


def main():
    # Setting hyperparameters
    hyperparameters = {
        "timesteps_per_batch": 1000,
        "max_timesteps_per_episode": 50,
        "n_updates_per_iteration": 80,
        "clip": 0.2,
        "lr": 0.0003,
        "render": False,
        "render_every_i": 10,
    }

    # Create the environment
    path_to_arm = r"Env\mujoco_menagerie-franka_emika_panda_v0\franka_emika_panda\panda.xml"
    path_to_arm2 = r"Env\mujoco_menagerie-franka_emika_panda_v0\franka_emika_panda\panda2.xml"
    panda = PandaArm(path_to_arm)
    panda2 = PandaArm(path_to_arm2)

    task = MimicArm(panda, panda2, control_ts=0.5)
    env = composer.Environment(task, random_state=None)

    # Train the model
    train(env=env, hyperparameters=hyperparameters, actor_model='', critic_model='')


def main_test():
    # Create the environment
    path_to_arm = r"Env\mujoco_menagerie-franka_emika_panda_v0\franka_emika_panda\panda.xml"
    path_to_arm2 = r"Env\mujoco_menagerie-franka_emika_panda_v0\franka_emika_panda\panda2.xml"
    panda = PandaArm(path_to_arm)
    panda2 = PandaArm(path_to_arm2)

    task = MimicArm(panda, panda2, control_ts=0.5, reduce_action_space=False)
    env = composer.Environment(task, random_state=None, time_limit=10.)

    # Specify path to the model of the actor network
    actor_path = r"Models\22_12_03\16_36\ppo_actor_30.pth"

    # Create the path for the Log files, if it does not exist
    log_test_root = root_dir(folder="Test_log")

    # Write the results into a JSON file
    test_log = {}
    test_log_path = os.path.join(log_test_root, "test_log.json")
    for i in range(10):
        print(f"Iteration #{i + 1}")
        test_log_iter = test(env=env, actor_model=actor_path)
        test_log[f"it_{i + 1}"] = test_log_iter

    with open(test_log_path, "w") as json_out_file:
        json.dump(test_log, json_out_file, indent=4)


if __name__ == '__main__':
    # Training
    main()

    # Testing
    main_test()
