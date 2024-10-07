# Setup for Running Web Navigation Using Language Agent Tree Search(LATS)

This is to document the first-time setup to run [LAST Decision-making experiment](https://github.com/lapisrocks/LanguageAgentTreeSearch/tree/main?tab=readme-ov-file#decision-making-webshop) in this [paper](https://arxiv.org/abs/2310.04406)

### 1.Install WebShop 

The official instruction is [here](https://github.com/princeton-nlp/WebShop). However there are some nuance regarding whether your device is Macbook ARM or Intel.

1.1 Install Python 3.8.13 and Java 11 ``brew install openjdk@11``

1.2 Create a virtual environment using [Anaconda](https://anaconda.org/anaconda/python) and navigate to webshop/
```sh
conda create -n webshop python=3.8.13
conda activate webshop
cd webshop
```

1.3 Install requirements and download 1,000 products (run one of the commands based on your device)
```sh
# if Intel machine
./setup.sh -d small 

# else
./setup_arm.sh -d small
```

1.4 Start the instance (need to check if it is necessary, but the LAST is working with it)
```sh
./run_dev.sh
```

### 2.Run LAST

2.1 Install LAST Dependencies
```sh
cd LanguageAgentTreeSearch
pip install -r requirements.txt
```

2.2 Set up OpenAi
```sh
export OPENAI_API_KEY=<your key>
```

2.3 Run experiment
```sh
sh lats_test.sh
```

The options can be modified
* --backend: choose from 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'llama2', 'text-davinci-002'
* --iterations: maximum number of trajectories to sample
* --prompt_sample choose from 'standard' or 'cot' 

