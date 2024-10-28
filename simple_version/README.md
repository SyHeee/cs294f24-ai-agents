# Quick Setup for Running Current (yet not working well) Version

## Create a conda environment
```bash
conda create -n webagent python=3.11
```

## Install Dependencies 
```bash
pip install -r requirement.txt
playwright install
```

## Create a .env file under simple_version/ and set your API keys
```bash
OPENAI_API_KEY=...
```


## Run the main code
```bash
python agent.py
```
