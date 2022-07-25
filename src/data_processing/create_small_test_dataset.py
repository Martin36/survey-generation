import random
from utils_package.util_funcs import load_jsonl, store_jsonl
from utils_package.logger import get_logger

logger = get_logger()

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
OUTPUT_TRAIN_FILE = BASE_PATH+"mini_dataset_train.jsonl"
OUTPUT_VAL_FILE = BASE_PATH+"mini_dataset_val.jsonl"

PER_DS_SAMPLE_SIZE = 50

result = []
for folder in DATA_FOLDERS:
  splits = ["train", "val", "test"]
  for split in splits:
    file = BASE_PATH+folder+split+".jsonl"
    data = load_jsonl(file)
    if len(data) > PER_DS_SAMPLE_SIZE:
      data_sample = random.sample(data, PER_DS_SAMPLE_SIZE)
      result.extend(data_sample)

store_jsonl(result, OUTPUT_TRAIN_FILE)
logger.info(f"Stored mini test train dataset in '{OUTPUT_TRAIN_FILE}'")

# Create eval
val_result = random.sample(result, len(result)//10)
store_jsonl(val_result, OUTPUT_VAL_FILE)
logger.info(f"Stored mini test val dataset in '{OUTPUT_VAL_FILE}'")

