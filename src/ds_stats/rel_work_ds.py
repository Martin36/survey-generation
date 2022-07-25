from collections import defaultdict
from tqdm import tqdm
from utils_package.util_funcs import load_jsonl, store_json
from utils_package.logger import get_logger
from transformers import AutoTokenizer

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
OUTPUT_FILE = "stats/rel_work_benchmark_stats.json"

stats = defaultdict(int)
input_len_dist = defaultdict(int)
output_len_dist = defaultdict(int)
total_inputs_longer_than_16384 = 0  # 16384 is the max len of LED model
total_inputs_longer_than_4096 = 0
total_outputs_longer_than_1024 = 0
total_data_points = 0

tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
tokenizer.model_max_length = int(1e30) # Just to make sure that no input texts will be truncated

for folder in DATA_FOLDERS:
  ds_name = folder.replace("/", "_")
  ds_name = ds_name[:-1]
  current_input_len_dist = defaultdict(int)
  current_output_len_dist = defaultdict(int)
  current_total_inputs_longer_than_16384 = 0
  current_inputs_longer_than_4096 = 0
  current_outputs_longer_than_1024 = 0
  current_total_data_points = 0

  splits = ["train", "val", "test"]
  for split in splits:
    input_file = BASE_PATH+folder+split+".jsonl"
    data = load_jsonl(input_file)

    for d in tqdm(data):
      total_data_points += 1
      current_total_data_points += 1

      input_tokens = tokenizer.tokenize(d["input"])
      output_tokens = tokenizer.tokenize(d["target"])
      input_len = len(input_tokens)
      output_len = len(output_tokens)

      current_input_len_dist[input_len] += 1
      current_output_len_dist[output_len] += 1

      input_len_dist[input_len] += 1
      output_len_dist[output_len] += 1

      if input_len > 16384:
        total_inputs_longer_than_16384 += 1
        current_total_inputs_longer_than_16384 += 1

      if input_len > 4096:
        current_inputs_longer_than_4096 += 1
        total_inputs_longer_than_4096 += 1
      
      if output_len > 1024:
        current_outputs_longer_than_1024 += 1
        total_outputs_longer_than_1024 += 1

  stats[ds_name] = {
    "input_len_dist": current_input_len_dist,
    "output_len_dist": current_output_len_dist,
    "inputs_longer_than_16384": current_total_inputs_longer_than_16384,
    "inputs_longer_than_4096": current_inputs_longer_than_4096,
    "outputs_longer_than_1024": current_outputs_longer_than_1024,
    "total_data_points": current_total_data_points,
    "percentage_inputs_longer_than_16384": current_total_inputs_longer_than_16384/current_total_data_points,
    "percentage_inputs_longer_than_4096": current_inputs_longer_than_4096/current_total_data_points,
    "percentage_outputs_longer_than_1024": current_outputs_longer_than_1024/current_total_data_points,
  }

stats["total"] = {
  "input_len_dist": input_len_dist,
  "output_len_dist": output_len_dist,
  "inputs_longer_than_16384": total_inputs_longer_than_16384,
  "inputs_longer_than_4096": total_inputs_longer_than_4096,
  "outputs_longer_than_1024": total_outputs_longer_than_1024,
    "total_data_points": total_data_points,
    "percentage_inputs_longer_than_16384": total_inputs_longer_than_16384/total_data_points,
    "percentage_inputs_longer_than_4096": total_inputs_longer_than_4096/total_data_points,
    "percentage_outputs_longer_than_1024": total_outputs_longer_than_1024/total_data_points,
}

store_json(stats, OUTPUT_FILE, sort_keys=True)
logger.info(f"Stored 'stats' in '{OUTPUT_FILE}'")


