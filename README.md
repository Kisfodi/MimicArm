# MimicArm

## Introduction

**MimicArm** is a Reinforcement Learning environment where the goal is to make a robotic arm _imitate_ the orientation of another arm. The environment is simulated using the MuJoCo physics engine, and the implementation is achieved by using the [dm_control](https://github.com/deepmind/dm_control) library.

## The envrionment

The environment contains two **Franka Emika Panda** robotic arms, initialized with random keyframes. The goal of the agent is to control the actuators of the first arm (the agent) so that it is able to copy the state of the second arm, which is the _control_ or _goal_ state of the environment.

### Model

The model of the Panda arm is available in the [MuJoCo Menagerie](https://github.com/deepmind/mujoco_menagerie) library.

### Observation space

### Action space

## Training

## Testing
