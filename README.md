# Comparative Study of Reasoning, Planning, and Execution with Monte Carlo Tree Search in LLM-Based Web Agents

CS294 Large Language Model Agents Hackathon (Fundamental Track) Work Repository

Team Positronic Web Pilot

## 📝 About
In this project, we propose to integrate Monte Carlo Tree Search (MCTS) techniques to enhance reasoning, planning, and execution in LLM web agents.

We design two prompts, one of which enforce stricter decision-making instruction, while the other one encourage more exploration and harness the advantageous exploitation-exploration trade-off of MCTS. Experiments on the WebShop benchmark demonstrate that combining flexible prompts with MCTS significantly improves agent performance among all the configurations tested.

## 🔗 Demo
[![Check out the demo of our proposed algorithm on Youtube](https://img.youtube.com/vi/a0-t8fPYWIQ/default.jpg)](https://youtu.be/a0-t8fPYWIQ)

## 🧰 Experiments
### 1. Prerequisites Installation
We use the GPT-4o model as LLM and integrate MCTS to the [SeeAct](https://osu-nlp-group.github.io/SeeAct/) framework. We run experiments on the [WebShop](https://webshop-pnlp.github.io) dataset.

#### 1.1 SeeAct Installation
```bash
# create conda environment
conda create -n seeactmcts python=3.11
conda activate seeactmcts

# install dependencies
pip install seeact
pip uninstall openai # we will use older version commits hence need a compatible openai 
pip install openai==0.28.0

# set up PlayWright and install the browser kernels
playwright install chromium
```

#### 1.2 WebShop Installation
Follow the step 1 to 6 in the official [README](https://github.com/princeton-nlp/WebShop/blob/master/README.md) or [README-MAC](https://github.com/princeton-nlp/WebShop/blob/master/README_INSTALL_ARM-MAC.md) if you are using Apple Mac device.

**Note1** You would need to create another new conda environment for WebShop to start the environment
```bash
conda create -n webshop python=3.8.13
conda activate webshop
```
**Note2** In step 5, you would need to run the command below to load the full version data and follow step 6, to run the experiments in our demo.
```bash
./setup.sh all
```

### 2. Experiment Replication 
#### 2.1 Launch WebShop
```bash
cd ./ experiments/WebShop-master
./run_dev.sh
```

#### 2.2 Run Web Agent
First navigate to ./experiments/SeeAct-main/src and create a .env file. 
```bash
cd ./experiments/SeeAct-main/src
vim .env # then add your openai API key: OPENAI_API_KEY="YOUR_API_KEY"
```
Then you could run the scripts for experiment with web agents.
```bash
# single rollout using strict prompt
for n in {0..20}; do python seeact.py -c config/webshop_mode.toml -n $n; done;

# multiple rollouts with MCTS using strict prompt
for n in {0..20}; do python seeact_3.py -c config/webshop_mode.toml -n $n; done;

# single rollout using flexible prompt
for n in {0..20}; do python seeact.py -c config/webshop_mode2.toml -n $n; done;

# multiple rollouts with MCTS using strict prompt
for n in {0..20}; do python seeact_3.py -c config/webshop_mode2.toml -n $n; done;
```

#### 2.3 Calculate scores
```bash
# You shoulf modify the folder names in agg_score.py before running
python agg_score.py
```

##
👉 Primary contact of this project: Shiying He (sy.he0303@gmail.com)

👉 MOOC course site: [http://llmagents-learning.org/f24](http://llmagents-learning.org/f24).

👉 Hackathon website: [https://rdi.berkeley.edu/llm-agents-hackathon/](https://rdi.berkeley.edu/llm-agents-hackathon/).

