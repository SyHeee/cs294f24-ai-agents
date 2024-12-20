# Web Agent with Monte Carlo Tree Search

The code is dependent on openai==0.28.0

## How to run the base code
```python
python seeact.py -c config/webshop_mode.toml
```

## How to run the experiment
```bash
for n in {0..20}; do python seeact.py -c config/webshop_mode.toml -n $n; done;
for n in {0..20}; do python seeact_3.py -c config/webshop_mode.toml -n $n; done;
for n in {0..20}; do python seeact.py -c config/webshop_mode2.toml -n $n; done;
for n in {0..20}; do python seeact_3.py -c config/webshop_mode2.toml -n $n; done;
```
