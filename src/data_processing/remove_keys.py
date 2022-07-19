from utils_package.util_funcs import load_jsonl, store_jsonl
from utils_package.logger import get_logger

logger = get_logger()

def remove_keys(data):
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

for folder in DATA_FOLDERS:
  splits = ["train", "val", "test"]
  for split in splits:
    file = BASE_PATH+folder+split+".jsonl"
    data = load_jsonl(file)
    data = remove_keys(data)
    store_jsonl(data, file)
    logger.info(f"Removed keys from '{file}'")
