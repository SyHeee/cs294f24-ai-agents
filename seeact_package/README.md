# Quick Setup for Running SeeAct

## Create a conda environment
```bash
conda create -n seeactagent python=3.11
conda activate seeactagent
```

## Install Dependencies 
```bash
pip install -r requirement.txt
playwright install chromium
```

## Create a .env file under simple_version/ and set your API keys
```bash
OPENAI_API_KEY=...
```


## Follow the example_config.toml to create a task, modify the config path and run the main code
```bash
python example.py
```
