# Zero Shot Reinforcement Learning under Partial Observability

## Setup
### Dependencies
Assuming you have [MuJoCo](https://mujoco.org/) installed, setup a conda env with [Python 3.9.16](https://www.python.org/downloads/release/python-3916/) using `requirements.txt` as usual:
```
conda create --name zsrl python=3.9.16
```
then install the dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```

## Algorithms
We provide implementations of the following algorithms: 

| **Algorithm**         | **Authors**                                              | **Command Line Argument** |
|-----------------------|----------------------------------------------------------|-------|
 | FB                    | [Touati et. al (2023)](https://arxiv.org/abs/2209.14935) |   `fb` |
| HILP                  | [Park et. al (2024)](https://arxiv.org/abs/2402.15567)   | `rsf` |
| FB with Memory (FB-M) | [anon.]()                                                | `rfb` |

## Memory Models $f$

| **Memory Model**     | **Authors**                                               | **Command Line Argument**   |
|----------------------|-----------------------------------------------------------|-----------------------------|
 | GRU                  | [Cho et. al (2014)](https://arxiv.org/abs/1406.1078)      | `--memory_type=gru`         |
| Transformer          | [Vaswani et. al (2017)](https://arxiv.org/abs/1706.03762) | `--memory_type=transformer` |
| S4d                  | [Gu et. al (2022)](https://arxiv.org/abs/2206.1189)       | `--memory_type=s4d`         |
| MLP (frame-stacking) | n/a                                                       | `--memory_type=mlp`         |

You can modify their hyperparameters:

| **Hyperparameter**                  | **Description**                                                                           | Default                              | Command Line Arg                 |
|-------------------------------------|-------------------------------------------------------------------------------------------|--------------------------------------|----------------------------------|
 | $F/\pi$'s context length ($L_F$)    | Length of the trajectory passed to the forward model $F$ and policy $\pi$ during training | $32$                                 | `--history_length`               |
| $B$'s context length ($L_B$)        | Length of the trajectory passed to the backward model $B$ during training                 | $8$                                  | `--backward_history_length`      |
| Model Dimension                     | Hidden state dimension                                                                    | $512$ (GRU) & $32$ (Transformer/S4d) | `--model_dimension`              |
| Memory-based forward model / policy | Whether $F/\pi$ should contain a memory model                                             | `True`                               | `--recurrent_F/--no-recurrent_F` |
| Memory-based backward model         | Whether $B$ should contain a memory model                                                 | `True`                               | `--recurrent_B/--no-recurrent_B` |

## ExORL
In the paper we report results with agents trained on different partially observed variants of ExORL domains. The domains are:

| **Domain** | **Eval Tasks**                             | **Dimensionality** | **Type**   | **Reward** |
|-----------|--------------------------------------------|--------------------|------------|-----------|
| Walker    | `stand` `walk` `run` `flip`                | Low                | Locomotion | Dense     |
| Quadruped | `stand` `roll` `roll_fast` `jump` `escape` | High               | Locomotion | Dense     |
| Cheetah   | `run` `run_backward` `walk` `walk_backward` | Low                | Locomotion | Dense     |

## POMDPS
We implement a set of POMDPs that exhibit different types of partial observability:

| **POMDP Setting**       | **Description**                                                                                                              | **Default Hyperparameter(s)**                 | **Environment Command Line Arg** |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------|---------------------------------|
 | Flickering states       | States are dropped (zeroed) with probability $p_{\text{flickering}}$                                                         | `flickering_prob=0.2`                         | `{env_name}_flickering`         |
 | Noisy states            | Isotropic 0-mean Gaussian noise is added to states with variace $\sigma_{\text{noise}}$                                      | `noise_std=0.2`                               | `{env_name}_noise`              |
 | Dropped state variables | Subsets of states variables (sensors) are dropped (zeroed) with probability $p_{\text{sensors}}$                             | `missing_sensor_prob=0.2`                     | `{env_name}_sensors`            |
| Removed velocities      | Velocities are removed from the state                                                                                        | n/a                                           | `{env_name}_occluded`           |
| Changed dynamics        | Mass and damping coefficients in the underlying MuJoCu simulator are scaled to different values between training and testing | `train_multiplies=1.0` `eval_multipliers=1.0` | `{env_name}`                    |

For each domain, you'll need to download the RND dataset manually from the [ExORL benchmark](https://github.com/denisyarats/exorl/tree/main) then reformatted. 
To download the `rnd` dataset on the `walker` domain, seperate their command line args with an `_` and run:  

```bash
python exorl_reformatter.py walker_rnd
```

this will create a single `dataset.npz` file in the `dataset/walker/rnd/buffer` directory.

To train a standard FB-M model, with GRU memory model on `rnd` to solve all tasks in the `walker_flickering` domain, run:
```bash
python main_exorl.py rfb walker_flickering rnd --memory_type=gru --eval_task stand run walk flip
```

## License 
This work licensed under a standard MIT License, see `LICENSE.md` for further details.