import time
import gymnasium as gym
import requests
from bs4 import BeautifulSoup
import json
import os
import numpy as np
import re
import string
from collections import Counter

# import wikipedia

DATA_DIR = "data"
HOTPOTQA_SPLIT_FILE = {
  "train": "hotpot_train_v1.1_simplified.json",
  "dev": "hotpot_dev_v1_simplified.json",
  "test": "hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
  "train": "train.jsonl",
  "dev": "paper_dev.jsonl",
}


def clean_str(p):
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except UnicodeDecodeError:
        return p


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class WikiEnv(gym.Env):

  def __init__(self):
    """
      Initialize the environment.
    """
    super().__init__()
    self.page = None  # current Wikipedia page
    self.obs = None  # current observation
    self.lookup_keyword = None  # current lookup keyword
    self.lookup_list = None  # list of paragraphs containing current lookup keyword
    self.lookup_cnt = None  # current lookup index
    self.steps = 0  # current number of steps
    self.answer = None  # current answer from the agent
    self.observation_space = self.action_space = textSpace()
    self.search_time = 0
    self.num_searches = 0
    
  def _get_obs(self):
    return self.obs

  def _get_info(self):
    return {"steps": self.steps, "answer": self.answer}

  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset(seed=seed)
    self.obs = ("Interact with Wikipedia using search[], lookup[], and "
                "finish[].\n")
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None
    self.steps = 0
    self.answer = None
    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  def construct_lookup_list(self, keyword):
    # find all paragraphs
    if self.page is None:
      return []
    paragraphs = self.page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]

    parts = sentences
    parts = [p for p in parts if keyword.lower() in p.lower()]
    return parts

  @staticmethod
  def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
      sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret

  def search_step(self, entity):
    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    old_time = time.time()
    response_text = requests.get(search_url).text
    self.search_time += time.time() - old_time
    self.num_searches += 1
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
      self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
    else:
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page):
        self.search_step("[" + entity + "]")
      else:
        self.page = ""
        for p in page:
          if len(p.split(" ")) > 2:
            self.page += clean_str(p)
            if not p.endswith("\n"):
              self.page += "\n"
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
  
  def step(self, action):
    reward = 0
    done = False
    action = action.strip()
    # if self.answer is not None:  # already finished
    #   done = True
    #   return self.obs, reward, done, self._get_info()
    
    if action.startswith("search[") and action.endswith("]"):
      entity = action[len("search["):-1]
      # entity_ = entity.replace(" ", "_")
      # search_url = f"https://en.wikipedia.org/wiki/{entity_}"
      self.search_step(entity)
    elif action.startswith("lookup[") and action.endswith("]"):
      keyword = action[len("lookup["):-1]
      if self.lookup_keyword != keyword:  # reset lookup
        self.lookup_keyword = keyword
        self.lookup_list = self.construct_lookup_list(keyword)
        self.lookup_cnt = 0
      if self.lookup_cnt >= len(self.lookup_list):
        self.obs = "No more results.\n"
      else:
        self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
        self.lookup_cnt += 1
    elif action.startswith("finish[") and action.endswith("]"):
      answer = action[len("finish["):-1]
      self.answer = answer
      done = True
      self.obs = f"Episode finished, reward = {reward}\n"
    elif action.startswith("think[") and action.endswith("]"):
      self.obs = "Nice thought."
    else:
      self.obs = "Invalid action: {}".format(action)

    self.steps += 1

    return self.obs, reward, done, self._get_info()
  
  def get_time_info(self):
    speed = self.search_time / self.num_searches if self.num_searches else 0
    return {
        "call_speed": speed,
        "call_time": self.search_time,
        "num_calls": self.num_searches,
    }



class HistoryWrapper(gym.ObservationWrapper):
  def __init__(self, env, obs_format, prompt=None):
    super().__init__(env)
    assert obs_format in ["obs", "history"]
    if obs_format == "history":
      assert hasattr(self.env, "traj")
    self.obs_format = obs_format
    self.prompt = prompt if prompt is not None else ""

  def observation(self, obs):
    if self.obs_format == "obs":
      return obs
    elif self.obs_format == "history":
      observation = self.env.traj["observations"][0] + "\n"
      for i, (o, a) in enumerate(zip(self.env.traj["observations"][1:], self.env.traj["actions"]), 1):
        observation += f"Action {i}: {a}\nObservation {i}: {o}\n\n"
      return self.prompt + observation
    

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  
  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall
  
class HotPotQAWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    data_file = f"{DATA_DIR}/{HOTPOTQA_SPLIT_FILE[split]}"
    self.data = json.load(open(data_file))
    self.data = [(d['question'], d['answer']) for d in self.data]
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Question: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "hotpot_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      score = (pred == gt)
      return int(score)
    return 0
  
  def get_metrics(self, info):
    if info['answer'] is not None:
      pred = normalize_answer(self.data[self.data_idx][1])
      gt = normalize_answer(info['answer'])
      em = (pred == gt)
      f1 = f1_score(pred, gt)[0]
      return {'reward': em, 'em': em, 'f1': f1}
    return {'reward': 0, 'em': 0, 'f1': 0}

  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update(self.get_metrics(info))
    return obs, reward, done, info
  
  def __len__(self):
    return len(self.data)

class FeverWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    
    data_path = f"./data/{FEVER_SPLIT_FILE[split]}"
    with open(data_path, "r") as json_file:
      json_list = list(json_file)

    data = []
    for json_str in json_list:
      json_str = json.loads(json_str)
      label = json_str["label"]
      claim = json_str["claim"]
      data.append((claim, label))

    self.data = data
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Claim: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "fever_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      label = normalize_answer(self.data[self.data_idx][1])
      pred = normalize_answer(info['answer'])
      if label == pred:
        return 1
    return 0

  def step(self, action):
    # TODO: first step obs does not have question. 
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update({'em': reward, 'reward': reward, 'f1': reward})
    return obs, reward, done, info
    
  def __len__(self):
    return len(self.data)
  
  
class LoggingWrapper(gym.Wrapper):
  def __init__(self, env, folder="trajs", file_id=None):
    super().__init__(env)
    self.trajs = []
    self.traj = {"observations": [], "actions": []}
    self.folder = folder
    self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
    self.file_path = f"{self.folder}/{self.file_id}.json"
    os.makedirs("trajs", exist_ok=True)

  def __len__(self):
    return len(self.env.data)
  

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=idx)
    observation = output[0] if return_info else output
    self.traj = {"observations": [observation], "actions": []}
    return output

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.traj["observations"].append(obs)
    self.traj["actions"].append(action)
    if done:
      self.traj.update(info)
    return obs, reward, done, info

  def update_record(self):
    if len(self.traj) > 0:
      self.trajs.append(self.traj)
      self.traj = {"observations": [], "actions": []}
  
  def write(self):
    self.update_record()
    with open(self.file_path, "w") as f:
      json.dump(self.trajs, f)
      print(f"Saved trajs to trajs/{self.file_id}.json")
    
  def close(self):
    self.write()
