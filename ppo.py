import os.path
import torch
from torch.optim import Adam

import utils
from torch.distributions import MultivariateNormal

import collections
import dm_env
import time

import numpy as np
import json


class PPO:

    def __init__(self, policy_class, env, **hyperparameters):

        # Ensures compatibility with dm_control
        assert (type(env.observation_spec()) == collections.OrderedDict)
        assert (type(env.action_spec()) == dm_env.specs.BoundedArray)

        # Initialize hyperparameters for the training
        self._initialize_hyperparameters(hyperparameters)

        # Initialize base folders for model paths and log files
        self.model_root = utils.root_dir(folder="Models")
        self.log_root = utils.root_dir(folder="Log")

        # Environment information
        self.env = env
        self.obs_dim = list(self.env.observation_spec()["panda/joint_positions"].shape)[1] +\
                       list(self.env.observation_spec()["panda2/joint_positions"].shape)[1]

        # The action space is reduced or not
        if self.env.task.reduce_action_space:
            self.act_dim = 1
        else:
            self.act_dim = list(self.env.action_spec().shape)[0]

        # Upper and lower boundaries for the actions
        self.action_min = env.action_spec().minimum[:self.act_dim]
        self.action_max = env.action_spec().maximum[:self.act_dim]


        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim, act_min=self.action_min,
                                  act_max=self.action_max)
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        # self.actor_optim = Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-3)
        # self.critic_optim = Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-3)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize covariance matrix used to query the actor network
        self.var = 0.5
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.var)
        self.cov_mat = torch.diag(self.cov_var)

        # Threshold of the good performance
        self.threshold_high = 0.6
        self.threshold_low = 0.1

        # Provides writing logs to summarize each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0.0,
            'i_so_far': 0.0,
            'batch_lens': [],
            'batch_rews': [],
            'batch_rews_mod': [],
            'actor_losses': [],
            'good_ep_num': 0,
            'total_ep_num': 0,
            'success_rate': 0.0
        }

    def learn(self, total_timesteps):

        # Initial information
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        print(f"Models will be saved to folder {self.model_root}")

        t_so_far = 0  # Timesteps stimulated so far
        i_so_far = 0  # Iterations so far
        while t_so_far < total_timesteps:

            # The collected data
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Keeping track of the number of timesteps collected for this
            t_so_far += np.sum(batch_lens)

            # Keeping track of the number of iterations
            i_so_far += 1

            # Log the timesteps and iterations
            self.logger['t_so_far'] = int(t_so_far)
            self.logger['i_so_far'] = i_so_far

            # Calculate value of the critic network
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate advantage estimate of the k-th iteration
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Loop for updating the networks for n epochs
            for epoch in range(self.n_updates_per_iteration):
                # Calculate the value of the critic network and the current log probability of the action
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate the probability ratio
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses (unclipped and clipped)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                # Gradients and backward propagation for the actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Gradients and backward propagation for the critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach().tolist())

            # Summary of the log so far
            self._log_summary(tofile=True)

            # Save the model at every 5th iteration and at the last iteration
            if i_so_far % self.save_freq == 0 or t_so_far >= total_timesteps:
                # Define the file names based on the iteration number
                actor_name, critic_name = utils.model_names(it_num=i_so_far)

                torch.save(self.actor.state_dict(), os.path.join(self.model_root, actor_name))
                torch.save(self.critic.state_dict(), os.path.join(self.model_root, critic_name))

    def rollout(self):
        batch_obs = []  # Batch observations
        batch_acts = []  # Batch actions
        batch_log_probs = []  # Log probabilities for each action
        batch_rews = []  # Batch rewards
        batch_rews_mod = []  # Batch rewards modified after an episode
        batch_rtgs = []  # Batch rewards-to-go
        batch_lens = []  # Lengths of episodes in batch

        t = 0  # Timesteps run so far int the batch

        # Tracks the successful episodes per batch
        success_per_batch = 0
        # Tracks the total number of episodes per batch
        total_eps_per_batch = 0

        while t < self.timesteps_per_batch:

            # Rewards for an episode
            ep_rews = []
            ep_rews_mod = []

            # Prints out how the batch progresses
            print(f"Progress of batch: {t}/{self.timesteps_per_batch}")

            # Reset the environment
            env_state = self.env.reset()

            joints = np.concatenate((env_state.observation['panda/joint_positions'],
                                     env_state.observation['panda2/joint_positions']), axis=1)
            obs = joints.ravel()
            done = env_state.last()

            # Calculate the mean action, the action based on the mean, and its log probability
            action, log_prob, action_mean = self.get_action(obs)

            # If action space is reduced, then the calculated action will replace the previous action
            action_current = self.env.task.arm1_step
            np.clip(action, a_min=self.action_min[:self.act_dim], a_max=self.action_max[:self.act_dim], out=action)
            # print(f"Probability of action {action}: {np.exp(log_prob)}")

            action_current[:self.act_dim] = action.ravel()[:self.act_dim]

            # The step() function accepts control parameters of both arms, so they need to be concatenated
            action_final = np.concatenate((action_current.ravel(), self.env.task.arm2_step))

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):

                # Render the environment (NOT Specified)
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    # self.env.render()
                    pass

                t += 1  # Increment timesteps run so far in the batch

                # Keep track of the observations
                batch_obs.append(obs)

                # Print out some data in order to check the data of the episode
                # print(f"Arm 1: {obs[0]}")
                # print(f"Arm 2: {obs[9]}")

                # print(f"Mean of action: {action_mean}")
                print(f"Performed action: {action_final[0]}")
                print(f"Needed action: {action_final[8]}\n")

                # Make a step in the environment
                env_state = self.env.step(action_final)

                # Collect data after taking a step
                joints = np.concatenate((env_state.observation['panda/joint_positions'],
                                         env_state.observation['panda2/joint_positions']), axis=1)

                obs = joints.ravel()
                rew = env_state.reward
                done = env_state.last()

                # Difference of the current reward compared to the previous reward
                if len(ep_rews) == 0:
                    latest = rew
                else:
                    latest = ep_rews[-1]

                print(f"Rew: {rew}; latest: {latest}")
                rel_change = (rew / latest) - 1

                # Track recent reward, action, and log probability
                ep_rews.append(rew)
                ep_rews_mod.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # Print out results of current timestep
                print(f"Timestep of episode: {ep_t + 1}/{self.max_timesteps_per_episode}, "
                      f"reward: {round(rew, 6)}, relative change: {round(rel_change, 6)}\n")

                # If the reward does not change significantly,
                # either a new action is taken to increase the chances for finding an optimal action,
                # OR if the action si optimal, the episode is terminated

                if abs(rel_change) <= 0.001 and self.env.task.get_elapsed_time(self.env.physics) >= 2.5:
                    ep_rews_mod = np.array(ep_rews_mod)
                    if done or ep_t >= self.max_timesteps_per_episode - 1:
                        print("End of episode")
                        if ep_rews[-1] >= self.threshold_high:
                            ep_rews_mod += 1
                            success_per_batch += 1
                        total_eps_per_batch += 1
                        break
                    ep_rews_mod = list(ep_rews_mod)
                    print("Taking new action")

                    self.env.task.reset_time(self.env.physics)

                    done = env_state.last()

                    action, log_prob, action_mean = self.get_action(obs)

                    # If action space is reduced, then the calculated action will replace the previous action
                    np.clip(action, a_min=self.action_min[:self.act_dim], a_max=self.action_max[:self.act_dim],
                            out=action)
                    print(f"Probability of action {action}: {np.exp(log_prob)}")
                    action_current[:self.act_dim] = action.ravel()[:self.act_dim]

                    # The step() function accepts control parameters of both arms, so they need to be concatenated
                    action_final = np.concatenate((action_current.ravel(), self.env.task.arm2_step))

                    continue

                # If the episode is terminated because the maximum number of timesteps were reached or another condition
                # was fuliflled
                if done \
                        or abs(rel_change) <= 0.001 and self.env.task.get_elapsed_time(self.env.physics) >= 2.5 \
                        or ep_t >= self.max_timesteps_per_episode - 1:
                    print("End of episode")
                    ep_rews_mod = np.array(ep_rews_mod)
                    if ep_rews[-1] >= self.threshold_high:
                        success_per_batch += 1

                        ep_rews_mod += 1
                    elif ep_rews[-1] <= self.threshold_low:
                        ep_rews_mod[:] = 0

                    total_eps_per_batch += 1

                    break

            # Track length and rewards of an episode
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_rews_mod.append(list(ep_rews_mod))

        # Reshape data as tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # Rewards-to-go using original reward
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Rewards-to-go using modified reward
        # batch_rtgs = self.compute_rtgs(batch_rews_mod)

        # Log episodic returns and lengths, and other data for a batch
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_rews_mod'] = batch_rews_mod
        self.logger['batch_lens'] = batch_lens
        self.logger['good_ep_num'] = success_per_batch
        self.logger['total_ep_num'] = total_eps_per_batch
        self.logger['success_rate'] = round((success_per_batch / total_eps_per_batch), 5)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):

        # The rewards-to-go per episode for a batch
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # Discounted reward so far

            # Iterating through all rewards of an episode, starting from the back
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert to tensor
        batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs, imgs=None):

        # Query the actor network for a mean action
        mean = self.actor(obs)

        # Create a distribution using the mean action and the covariance matrix
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for the sampled action
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach(), mean.detach().numpy()

    def evaluate(self, batch_obs, batch_acts):

        # Query the critic network for a value V for each observation in the batches
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of the batch action using the actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def _initialize_hyperparameters(self, hyperparameters):
        self.gamma = 0.9  # Discount factor
        self.timesteps_per_batch = 1000  # Timesteps per batch
        self.max_timesteps_per_episode = 50  # Timesteps per episode
        self.n_updates_per_iteration = 80  # Number of times to update actor/critic for one batch
        self.clip = 0.2  # Clipping factor for the Surrogate
        self.lr = 0.0003  # Learning rate for actor and critic optimizer
        self.save_freq = 5  # How often the models are saved
        self.render = False  # If rendering is available
        self.render_every_i = 10  # Frequency of rendering
        self.seed = None  # Sets random seed, for reproducibility of the results

        # If other values are defined, change to those values
        if hyperparameters is not None:
            for param, val in hyperparameters.items():
                # print('self.' + param + '=' + str(val))
                exec('self.' + param + '=' + str(val))

        # Set the seed if specified
        if self.seed is not None:
            assert (type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self, tofile=False):

        # Calculate the time it took to complete an iteration
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        self.logger["duration"] = delta_t
        delta_t = str(round(delta_t, 3))

        # Timesteps taken so far
        t_so_far = self.logger['t_so_far']
        # Iterations taken so far
        i_so_far = self.logger['i_so_far']

        # Average length of episodes
        avg_ep_lens = np.mean(self.logger['batch_lens'])

        # Average return of episodes, NOTE: for each batch, the last return value is taken,
        # to better show the results of the training so far
        avg_ep_rews = np.mean([ep_rews[-1] for ep_rews in self.logger['batch_rews']])
        avg_ep_rews_mod = np.mean([ep_rews[-1] for ep_rews in self.logger['batch_rews_mod']])

        self.logger['avg_ep_rews'] = avg_ep_rews
        self.logger['avg_ep_rews_mod'] = avg_ep_rews_mod

        # Average of actor loss
        avg_actor_loss = np.mean([np.mean(np.array(losses)) for losses in self.logger['actor_losses']])

        # Convert values to strings
        avg_ep_lens = str(round(avg_ep_lens, 5))
        avg_ep_rews = str(round(avg_ep_rews, 5))
        avg_ep_rews_mod = str(round(avg_ep_rews_mod, 5))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        success_rate = str(self.logger['success_rate'])

        # Print logging statements
        print(flush=True)
        print(f"---------- Iteration #{i_so_far} ----------", flush=True)
        print(f"Average Episodic length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews} ; {avg_ep_rews_mod}", flush=True)
        print(f"Average Loss: {avg_actor_loss}")
        print(f"Total Episodes: {self.logger['total_ep_num']}", flush=True)
        print(f"Total Successful Episodes: {self.logger['good_ep_num']}", flush=True)
        print(f"Success rate: {success_rate}")
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} seconds", flush=True)
        print(f"-------------------------------------------", flush=True)
        print(flush=True)

        # If set to True, all data of an iteration is written to a JSON file as a dictionary
        if tofile:
            full_path = os.path.join(self.log_root, f"log_{i_so_far:02d}.json")
            with open(full_path, "w") as json_out_file:
                json.dump(self.logger, json_out_file, indent=4)

        # Reset batch-specific data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['batch_rews_mod'] = []
        self.logger['actor_losses'] = []
        self.logger['good_ep_num'] = 0
        self.logger['total_ep_num'] = 0
        self.logger['success_rate'] = 0.0
        self.logger["duration"] = 0.0
