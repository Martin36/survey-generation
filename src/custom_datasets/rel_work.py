import os
import torch
from torch.utils.data import Dataset
from utils_package.util_funcs import load_json

# LOCAL
ROOT_DIR = "/home/funkquist/survey-generation/"
# SLURM
# ROOT_DIR = "/storage/ukp/work/funkquist/"

BASE_PATH = ROOT_DIR+"data/related_work/benchmark/"


def load_data(split: str):
  data_folders = [
    "aburaed_et_at/", 
    # "chen_et_al/delve/",
    # "chen_et_al/s2orc/", 
    # "lu_et_al/", 
    # "xing_et_al/explicit/",
    # "xing_et_al/hp/",
    # "xing_et_al/hr/",
  ]
  data = []
  for folder in data_folders:
    data_file = BASE_PATH+folder+split+".json"
    cur_data = load_json(data_file)
    for d in cur_data:
      d["dataset"] = folder.replace("/", "")
    data.extend(cur_data)
  return data


class RelatedWorksDataset(Dataset):

  def __init__(self, split: str) -> None:
    self.data = load_data(split)

  def __len__(self) -> int:
    return len(self.train) + len(self.val) + len(self.test)

  def __getitem__(self, idx: int) -> dict:
    if idx < len(self.train):
      
      return self.data[idx]
