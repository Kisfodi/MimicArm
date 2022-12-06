# Access to MuJoCo library functions.

# General
import numpy as np

from dm_control import mujoco
from dm_env import specs

# Composer high level imports
from dm_control import composer
# PyMJCF
from dm_control import mjcf
from dm_control.composer.observation import observable
# Imports for Composer
# Manipulation
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import observations

from dm_control.composer import variation

# Imports for Composer
from dm_control.locomotion.arenas import floors

# Offset pf the 2nd arm
ARM2_OFFSET = [-10, -10]

# Defining cameras for the robotic arms to use if rendering is needed

ARM1_TOP_DOWN = cameras.CameraSpec(
    name='arm1_top_down',
    pos=(0, 0, 3),
    xyaxes=(1, 0, 0, 0, 1, 0)
)

ARM1_FRONT_BACK = cameras.CameraSpec(
    name='arm1_front_back',
    pos=(3, 0, 0),
    xyaxes=(0, 1, 0, 0, 0, 1)
)

ARM1_LEFT_RIGHT = cameras.CameraSpec(
    name='arm1_left_right',
    pos=(0, 3, 0),
    xyaxes=(1, 0, 0, 0, 0, 1)
)
ARM2_TOP_DOWN = cameras.CameraSpec(
    name='arm2_top_down',
    pos=(ARM2_OFFSET[0], ARM2_OFFSET[1], 3),
    xyaxes=(1, 0, 0, 0, 1, 0)
)

ARM2_FRONT_BACK = cameras.CameraSpec(
    name='arm2_front_back',
    pos=(ARM2_OFFSET[0] + 3, ARM2_OFFSET[1], 0),
    xyaxes=(0, 1, 0, 0, 0, 1)
)

ARM2_LEFT_RIGHT = cameras.CameraSpec(
    name='arm2_left_right',
    pos=(ARM2_OFFSET[0], ARM2_OFFSET[1] - 3, 0),
    xyaxes=(1, 0, 0, 0, 0, 1)
)


class PandaArm(composer.Robot):

    # Build the model using the .xml file
    def _build(self, xml):
        try:
            self._model = mjcf.from_path(xml)
        except FileNotFoundError as e:
            print(e)

    # Define the observables of the arm
    def _build_observables(self):
        return PandaArmObservables(self)

    # The model of the arm
    @property
    def mjcf_model(self):
        return self._model

    # Actuators of the model
    @property
    def actuators(self):
        return tuple(self._model.find_all('actuator'))


class PandaArmObservables(composer.Observables):

    # Position of joints
    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        joint_pos = observable.MJCFFeature('qpos', all_joints)

        return joint_pos

    # Velocity of joints (Not used during thesis)
    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        joint_vel = observable.MJCFFeature('qvel', all_joints)
        return joint_vel


class MimicArm(composer.Task):
    def __init__(self, arm1, arm2=None, control_ts=None, reduce_action_space=True):

        # Define the robotic arms
        self._arm1 = arm1
        self._arm2 = arm2

        # Define an arena as the base of the environment
        self._arena = floors.Floor(size=(16, 16))

        # Add the 1st arm to the environment
        self._arena.attach(self._arm1)
        self._arm1.observables.joint_positions.enabled = True
        self._arm1.observables.joint_velocities.enabled = True

        # if 2 arms are defined
        if (arm2 is not None):
            # Add the 1st arm to the environment
            self._arena.attach(self._arm2)
            self._arm2.observables.joint_positions.enabled = True
            self._arm2.observables.joint_velocities.enabled = True

        # Set camera specs as task-specific observables if needed (Not used)
        self._task_observables = cameras.add_camera_observables(self._arena, observations.VISION,
                                                                ARM1_TOP_DOWN,
                                                                ARM1_FRONT_BACK,
                                                                ARM1_LEFT_RIGHT,
                                                                ARM2_TOP_DOWN,
                                                                ARM2_FRONT_BACK,
                                                                ARM2_LEFT_RIGHT)

        # Set the control timestep: how much time a step() in the environment takes
        if control_ts is not None:
            self.control_timestep = control_ts

        # Adds variation to the environment
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Upper and lower limits for joint positions
        self.max_qpos = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
        self.min_qpos = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0., 0.])

        # Absolute range of the joint positions
        self.range_qpos = self.max_qpos - self.min_qpos

        # Boolean if the action space is reduced
        self.reduce_action_space = reduce_action_space

    def action_spec(self, physics):

        # The original shape of action space is (16,)
        names = [physics.model.id2name(i, 'actuator') or str(i)
                 for i in range(physics.model.nu)]

        # Reduce it to a size of 8 (actuators of the 1st robotic arm)
        names = names[:8]
        action_spec = mujoco.action_spec(physics)
        return specs.BoundedArray(shape=action_spec.minimum[:8].shape,
                                  dtype=action_spec.dtype,
                                  minimum=action_spec.minimum[:8],
                                  maximum=action_spec.maximum[:8],
                                  name='\t'.join(names))

    # The root of the environment of the task
    @property
    def root_entity(self):
        return self._arena

    # The model of the 1st arm
    @property
    def arm1(self):
        return self._arm1

    # Model of the 2nd arm
    @property
    def arm2(self):
        return self._arm2

    # Return task-specific observables
    @property
    def task_observables(self):
        return self._task_observables

    # Initialize episode
    def initialize_episode_mjcf(self, random_state):

        # Apply variations to the model
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):

        print("Initializing episode...")

        # Apply variations to the physics of the model
        self._physics_variator.apply_variations(physics, random_state)

        # If the action space is reduced, then create a random state where only the first control is different
        if self.reduce_action_space:
            actuators = np.array([np.random.uniform(low=self.action_spec(physics).minimum[i],
                                                    high=self.action_spec(physics).maximum[i])
                                  for i in range(len(self.action_spec(physics).maximum) - 1)])

            # Set the initial control states of the arms
            self.arm1_step = np.concatenate(([np.random.uniform(low=self.action_spec(physics).minimum[0],
                                                                high=self.action_spec(physics).maximum[0])], actuators))
            self.arm2_step = np.concatenate(([np.random.uniform(low=self.action_spec(physics).minimum[0],
                                                                high=self.action_spec(physics).maximum[0])], actuators))
        # If action space is not reduced, then create random states where the controls are different
        else:
            self.arm1_step = np.array([np.random.uniform(low=self.action_spec(physics).minimum[i],
                                                         high=self.action_spec(physics).maximum[i])
                                       for i in range(len(self.action_spec(physics).maximum))])

            self.arm2_step = np.array([np.random.uniform(low=self.action_spec(physics).minimum[i],
                                                         high=self.action_spec(physics).maximum[i])
                                       for i in range(len(self.action_spec(physics).maximum))])

        # The step() function of the environment and its physics,
        # and the reset() function of the environment only accepts arrays of shape (16,)
        init_step = np.concatenate((self.arm1_step, self.arm2_step))

        # Simulate the physics so that the environment starts from the initial state defined above
        physics.set_control(init_step)
        while physics.time() < 2.:
            physics.step()

        # Sets starting time for an interva
        self.start_time = physics.time()

     # Reset starting time during an episode
    def reset_time(self, physics):
        self.start_time = physics.time()

    # Get the time elapsed since the starting point
    def get_elapsed_time(self, physics):

        end_time = physics.time()

        # Avoid inaccuray in the time (e.g. 0.4999999 instead of 0.5)
        return round(end_time - self.start_time, 4)


    def should_terminate_episode(self, physics):


        elapsed = self.get_elapsed_time(physics)

        # Terminate episode if the reward passes a certain threshold for good performance,
        # after a certain amount of time is passed
        if self.get_reward(physics) >= .6:
            return elapsed >= 2.5

        return False

    # Set discount factor (Not used, there is a factor ppo.py instead)
    def get_discount(self, physics):
        return 0.99

    # Get reward for a given step
    def get_reward(self, physics):

        # Joint positions of the 1st and 2nd arm
        arm1_qpos = physics.named.data.qpos[:9]
        arm2_qpos = physics.named.data.qpos[9:]

        # Absolute difference of joint positions
        pos_diff = (arm1_qpos - arm2_qpos)

        # Relative difference of joint positions
        pos_diff_rel = np.abs(pos_diff / self.range_qpos)

        # Apply weigh if the relative difference falls under a certain percentage
        pos_diff_rel = np.array([0 if pos <= 0.05 else pos for pos in pos_diff_rel])

        # Scalar included in the exponential. It must be a negative value.
        exponent_scalar = -1.0

        # If the scalar is initially set to positive, then set it to the negative of its absolute value
        if exponent_scalar > 0.0:
            exponent_scalar = -1.0 * abs(exponent_scalar)
        # If it is initially 0, then set it to a default value
        elif exponent_scalar == 0.0:
            exponent_scalar = -1.0

        # Calculate reward: 1. calculate the norm of the differences, 2. multiply with the scalar, 3. exponentiate
        # the result For reduced action space, use absolute differences
        if self.reduce_action_space:
            return np.exp(exponent_scalar * np.linalg.norm(pos_diff))

        # For original action space, use the weighted relative differences
        else:
            return np.exp(exponent_scalar * np.linalg.norm(pos_diff_rel))
