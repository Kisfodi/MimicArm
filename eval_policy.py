import numpy as np


def _log_summary(ep_len, ep_ret, ep_num):

    # Round values for more sophisticated results
    ep_len_str = str(round(ep_len, 5))
    ep_ret_str = str(round(ep_ret, 5))

    # Print results
    print(flush=True)
    print(f"---------- Episode #{ep_num + 1} ----------", flush=True)
    print(f"Episodic length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"---------------------------------------", flush=True)
    print(flush=True)

    return ep_ret, ep_len


def rollout(policy, env, render, num_of_eps=50):
    # Run for a certain number of episodes
    for _ in range(num_of_eps):

        # Observations and terminating condition
        env_state = env.reset()
        obs = np.concatenate((env_state.observation['panda/joint_positions'],
                              env_state.observation['panda2/joint_positions']), axis=1)
        done = env_state.last()

        # Number of timesteps so far
        t = 0

        # Log data
        ep_len = 0  # Length of episode
        ep_ret = 0  # Return of an episode

        while not done:
            t += 1

            # Render if specified (NOT Specified)
            if render:
                print("Rendering...")
                # env.physics.render()

            # Query for deterministic action from the policy
            action = policy(obs).detach().numpy()

            # print(f"Action: {action.ravel()}")

            # final_action = np.concatenate((action[0], env.task.arm1_step[1:], env.task.arm2_step))

            final_action = np.concatenate((action.ravel(), env.task.arm2_step))

            # Take a step in the environment
            env_state = env.step(final_action)
            obs = np.concatenate((env_state.observation['panda/joint_positions'],
                                  env_state.observation['panda2/joint_positions']), axis=1)
            rew = env_state.reward
            done = env_state.last()

        # Track length and return of episode
        ep_ret = rew
        ep_len = t

        yield ep_len, ep_ret


def eval_policy(policy, env, render=False, num_of_eps=50):
    log_test = {}
    returns = []
    lengths = []
    # Total number of episodes
    total = 0

    # Number of successful episodes
    success = 0

    # Rollout with the policy and the environment, log the data of each episode
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render, num_of_eps=num_of_eps)):
        reward, length = _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
        if reward >= .6:
            success += 1
        total += 1
        returns.append(reward)
        lengths.append(length)

    # Calculate success rate
    success_rate = round(success / total, 5)
    print(f"Success_rate: {success_rate}")

    log_test['ep_return'] = returns
    log_test['ep_lens'] = lengths
    log_test['success_rate'] = success_rate
    return log_test
