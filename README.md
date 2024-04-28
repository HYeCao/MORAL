# MORAL

We are sharing the core MORAL codes, encompassing utilities, models, and necessary dependencies for reproduction.

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
