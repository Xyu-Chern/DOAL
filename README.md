# 

[![PyPI version](https://img.shields.io/pypi/v/flowrl.svg)](https://pypi.org/project/flowrl) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green.svg)](https://www.python.org/) [![Python 3.8+](https://static.pepy.tech/badge/flowrl)](https://pepy.tech/projects/flowrl)

Flow RL is a high-performance reinforcement learning library, combining modern deep RL algorithms with flow and diffusion models for advanced policy parameterization, planning ability or dynamics modeling. It features:
- **State-of-the-Art Algorithms and Efficiency**: We provide JAX implementations of SOTA algorithms, such FQL, BDPO, DAC and etc;
- **Flexible Flow Architectures**: We provide built-in support various types of flow and diffusion models, such as CNFs and DDPM;
- **Comprehensive Evaluations**: We test the algorithms on commonly adopted benchmark and provide the results.

## 🚀 Installation & Usage
Currently FlowRL is hosted on PyPI and therefore can be installed via `pip install flowrl`. However, we recommend to clone and install the library using the following commands:
```bash
git clone https://github.com/typoverflow/flow-rl.git
cd flow-rl
pip install -e .
```



## 📊 Supported Algorithms
Offline RL:
|Algorithm|Location|WandB Report|
|:---:|:---:|:---:|
|IQL|`DOAL/agent/iql.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl?nw=urvdu9rz7b&panelDisplayName=eval%2Fmean&panelSectionName=eval) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=urvdu9rz7b)|
|DIQL|`DOAL/agent/ivr.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl/panel/nz7r4sj4n?nw=oslzekjlr1q) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=oslzekjlr1q)|
|FQL|`DOAL/agent/fql/fql.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl?nw=u9y84ki7rdi&panelDisplayName=eval%2Fmean&panelSectionName=eval) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=u9y84ki7rdi)|


