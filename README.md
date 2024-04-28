# MORAL

We are sharing the core MORAL codes, encompassing utilities, models, and necessary dependencies for reproduction.

---

## Introduction
Model-based offline Reinforcement Learning (RL) constructs environment models from offline datasets to perform conservative policy optimization. Existing approaches focus on learning state transitions through ensemble models, applying conservative estimates to unobserved transitions to mitigate extrapolation errors. These methods, however, necessitate a meticulous design process involving the calibration of ensemble models and rollout horizons for varying tasks, leading to heightened algorithmic complexity and constrained universality. To address these challenges, we introduce a novel approach, named Model-based Offline Reinforcement learning with AdversariaL game-driven policy optimization (MORAL). In the framework, we formulate a Markov Zero-sum Game (MZG) process to execute alternating sampling based on constructed ensemble models, replacing fixed horizon rollout in existing state-of-the-arts. This adversarial process makes full use of ensemble models while avoiding the optimistic estimation of models. To further enhance the robustness, a differential factor is integrated into the adversarial game to regularize policy and prevent divergence, ensuring error minimization in extrapolations. This dynamic optimization adapts to diverse offline tasks without costly tuning, showing remarkable universality. Extensive experiments on D4RL benchmark demonstrate that MORAL outperforms other model-based offline RL methods and achieves robust performance.

---

## Installation
1. Install [MuJoCo 2.0.0](https://github.com/deepmind/mujoco/releases) to `~/.mujoco/mujoco200`.
2. Create a conda environment and install requirements with `MORAL_env.yml`.

---

## Usage
For example, to run the hopper-medium task in D4RL benchmark, use the following:

```
python main.py --task=hopper-medium-v2
```
Detailed configuration can be found in `config.py`.


#### Logging
By default, TensorBoard logs will be generated in the `log/` directory.
