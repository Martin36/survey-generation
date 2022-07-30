from typing import TypedDict
from nltk import tokenize
from tqdm import tqdm

from utils_package.util_funcs import load_json, store_jsonl
from utils_package.logger import get_logger

logger = get_logger()

BASE_PATH = "data/"
LU_ET_AL_PATH = BASE_PATH+"related_work/lu_et_al/"
OUTPUT_PATH = BASE_PATH+"related_work/benchmark/easy/"

class Datum(TypedDict):
  target: str             # The target related work section/citation sentence
  input: str              # The unified format of the input


def get_data_for_split(split):
  result = list()
  data = load_json(LU_ET_AL_PATH+split+".json")
  for d in tqdm(data):
    input_docs=[r["abstract"] for r in d["ref_abstract"].values()]
    merged_input_docs = " ".join(input_docs)
    target = " ".join(tokenize.sent_tokenize(merged_input_docs)[:3])
    res_obj = Datum(
      target=target.strip(),
      input=input_docs,
    )
    result.append(res_obj)
  return result

def create_first_sent_ds():
  train = get_data_for_split("train")
  val = get_data_for_split("val")
  test = get_data_for_split("test")

  store_dataset(train, val, test, "lu_et_al/")


def store_dataset(train: list, val: list, test: list, folder: str):
  train_file = OUTPUT_PATH+folder+"train.jsonl"
  store_jsonl(train, train_file)
  logger.info(f"Stored 'train' in '{train_file}'")

  val_file = OUTPUT_PATH+folder+"val.jsonl"
  store_jsonl(val, val_file)
  logger.info(f"Stored 'val' in '{val_file}'")

  test_file = OUTPUT_PATH+folder+"test.jsonl"
  store_jsonl(test, test_file)
  logger.info(f"Stored 'test' in '{test_file}'")


if __name__ == "__main__":
  create_first_sent_ds()
