import random
from utils_package.util_funcs import load_jsonl, store_jsonl
from utils_package.logger import get_logger

logger = get_logger()

def create_mini_dataset(data):
  keys_to_keep = ["target", "input"]
  result = []
  for d in data:
    d = {k: d[k] for k in keys_to_keep}
    result.append(d)
  return result

BASE_PATH = "data/related_work/benchmark/"
DATA_FOLDERS = [
  "aburaed_et_at/", 
  "chen_et_al/delve/",
  "chen_et_al/s2orc/", 
  "lu_et_al/", 
  "xing_et_al/explicit/",
  "xing_et_al/hp/",
  "xing_et_al/hr/",
]
OUTPUT_FILE = BASE_PATH+"mini_dataset.jsonl"

result = []
for folder in DATA_FOLDERS:
  splits = ["train", "val", "test"]
  for split in splits:
    file = BASE_PATH+folder+split+".jsonl"
    data = load_jsonl(file)
    if len(data) > 5:
      data_sample = random.sample(data, 5)
      result.extend(data_sample)

store_jsonl(result, OUTPUT_FILE)
logger.info(f"Stored mini test dataset in '{OUTPUT_FILE}'")
