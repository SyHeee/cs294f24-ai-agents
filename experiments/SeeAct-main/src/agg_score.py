import json
import sys
import os
from os.path import join as oj
import pandas as pd


#dir_paths = ["../2024_12_7_first_results", "../2024_12_7_new_prompt", "../2024_12_8_first_click", "../2024_12_8_seeact_3"]
dir_paths = ["../2024_12_7_results"]
for dir_path in dir_paths:
    out = []
    i = 0
    for path in os.listdir(dir_path):
        files = os.listdir(oj(dir_path, path))
        if len([x for x in files if x.endswith("json")]) == 0:
            out.append([i, -2])
            i += 1
        for z in files:
            if not z.endswith("json"):
                continue
            with open(oj(dir_path, path, z), "rb") as f:
                result = json.load(f)
                score = result["success_or_not"] 
                out.append([i, float(score) if score != "" else -1])
            i += 1
    out = pd.DataFrame(out, columns=["index", "score"])
    #print(out)
    print(out.score.clip(0, 1).mean(), (out.score < 0).mean(), out[out.score >=0].score.mean())
