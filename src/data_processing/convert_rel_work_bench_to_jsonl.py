from utils_package.util_funcs import load_json, store_jsonl
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

for folder in DATA_FOLDERS:
  splits = ["train", "val", "test"]
  for split in splits:
    input_file = BASE_PATH+folder+split+".json"
    output_file = BASE_PATH+folder+split+".jsonl"
    data = load_json(input_file)
    store_jsonl(data, output_file)
    logger.info(f"Stored '{input_file}' in '{output_file}'")

