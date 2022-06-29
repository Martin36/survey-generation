from collections import defaultdict
import re, pickle, random
from typing import List, TypedDict

from utils_package.util_funcs import load_json, load_jsonl, store_json, unique
from utils_package.logger import get_logger

logger = get_logger()

### CLOUD PATH ###
BASE_PATH = "/ukp-storage-1/funkquist/"
### LOCAL PATH ###
#BASE_PATH = "data/"

ABURAED_ET_AL_2_PATH = BASE_PATH+"related_work/aburaed_et_al_2/"
CHEN_ET_AL_PATH = BASE_PATH+"related_work/chen_et_al/"
LU_ET_AL_PATH = BASE_PATH+"related_work/lu_et_al/"
SHAH_ET_AL_PATH = BASE_PATH+"related_work/shah_et_al/"
XING_ET_AL_PATH = BASE_PATH+"related_work/xing_et_al/"

OUTPUT_PATH = BASE_PATH+"related_work/benchmark/"

### TARGET DATA STRUCTURE ###

class Datum(TypedDict):
  target: str             # The target related work section/citation sentence
  input_docs: List[str]   # List of text from cited papers
  src_dataset: str        # Name of the dataset where the data sample comes from
  task: str               # Name of original task e.g. related work generation or citation generation


def get_data_aburaed_et_al_2(split):
  src_ending = ".txt.src"
  tgt_ending = ".txt.tgt.tagged"

  src = open(ABURAED_ET_AL_2_PATH+split+src_ending, "rb").read().decode("utf8").splitlines()
  tgt = open(ABURAED_ET_AL_2_PATH+split+tgt_ending, "rb").read().decode("utf8").splitlines()

  results = list()

  for sline, tline in zip(src,tgt):
    tline = re.sub("(\([^()]*\d{2,4}[^()]*\))|([A-Z]\w+\s*.{1,8}\([^()]*\d{4}[^()]*\))|([A-Z]\w+\s*\([^()]*\d{4}[^()]*\))|([A-Z]\w+\s*and\s*[A-Z]\w+\s*\([^()]*\d{4}\))|(\[[^\[\]]*\d{1,4}[^\[\]]*\])", "<CITE>", tline)
    tline = re.sub("\d{1}", "#", tline)
    sline = re.sub("\d{1}", "#", sline)

    res_obj = Datum(
      target=tline,
      input_docs=[sline],
      src_dataset="aburaed et al 2",
      task="citation sentence generation"
    )

    results.append(res_obj)

  return results

def process_aburaed_et_al_2():
  splits = ["test", "train", "val"]

  results = list()
  for split in splits:
    results += get_data_aburaed_et_al_2(split)

  return results
    
def get_data_chen_et_al(data_path):
  result = list()
  delve_data = load_jsonl(CHEN_ET_AL_PATH+data_path+".jsonl")
  for d in delve_data:
    res_obj = Datum(
      target=d["abs"],
      input_docs=d["multi_doc"],
      src_dataset="chen et al",
      task="related work generation"
    )
    result.append(res_obj)
  return result

def process_chen_et_al():
  delve_path = "delve/"
  s2orc_path = "s2orc/"

  splits = ["test", "train", "valid"]

  result = list()

  for split in splits:
    result += get_data_chen_et_al(delve_path+split)
    result += get_data_chen_et_al(s2orc_path+split)

  return result

def process_lu_et_al():
  # This dataset also contains abstracts of the target papers
  # Is this something that could be added?

  splits = ["test", "train", "val"]

  result = list()

  for split in splits:
    data = load_json(LU_ET_AL_PATH+split+".json")
    for d in data:
      res_obj = Datum(
        target=d["related_work"],
        input_docs=[r["abstract"] for r in d["ref_abstract"].values()],
        src_dataset="lu et al",
        task="related work generation"
      )

      result.append(res_obj)

  return result

def load_pickle_data(path):
  data = list()

  with open(path, "rb") as f:
    while True:
      try:
        data.append(pickle.load(f))
      except EOFError:
        break
  
  data = data[0]

  return data

def filter_non_empty_texts(data):
  return {k:v for k,v in data.items() if len(v.strip())}

# TODO: Figure out how this dataset is strucutred
def process_shah_et_al():
  rel_works_path = "aan/papers_text/papers_trimmed_related_works.p"
  abs_path = "aan/papers_text/papers_abstracts.p"
  file_abs_path = "acl_anthology_pdfs/file_abstract.p"

  rel_works_data = load_pickle_data(SHAH_ET_AL_PATH+rel_works_path)
  rel_works_data = filter_non_empty_texts(rel_works_data)

  abs_data = load_pickle_data(SHAH_ET_AL_PATH+abs_path)
  abs_data = filter_non_empty_texts(abs_data)

  file_abs_data = load_pickle_data(SHAH_ET_AL_PATH+file_abs_path)

  merged_data = list()
  stats = defaultdict(int)

  for file_name in rel_works_data:
    res_obj = {
      "abstract": abs_data[file_name] if file_name in abs_data.keys() else None,
      "related_work": rel_works_data[file_name],
      "file_name": file_name
    }
    merged_data.append(res_obj)

    if not res_obj["abstract"]:
      stats["# missing abstract"] += 1

  for file_name in abs_data:
    res_obj = {
      "abstract": abs_data[file_name],
      "related_work": rel_works_data[file_name] if file_name in rel_works_data.keys() else None,
      "file_name": file_name
    }
    merged_data.append(res_obj)

    if not res_obj["related_work"]:
      stats["# missing related work"] += 1

  # TODO: Remove duplicates

  has_both_data = [d for d in merged_data if d["abstract"] and d["related_work"]]
  file_names = [d["file_name"] for d in merged_data]
  unique_file_names = unique(file_names)

  logger.info(f"# missing abstract: {stats['# missing abstract']}")
  logger.info(f"# missing related work: {stats['# missing related work']}")
  logger.info(f"# containing both: {len(has_both_data)}")
  logger.info(f"# files: {len(file_names)}")
  logger.info(f"# unique files: {len(unique_file_names)}")

  print(abs_data)


def process_xing_et_al():
  # This dataset also contains context before and after
  # the citation and distinguishes between explicit citations
  # e.g. citations where the citation is in the sentence
  # and implicit citations where the citation might not be in the sentence
  # It also has the abstract of the target paper
  data_path = "citation.jsonl"

  data = load_jsonl(XING_ET_AL_PATH+data_path)

  result = list()

  for d in data:
    res_obj = Datum(
      target=d["explicit_citation"],
      input_docs=[d["src_abstract"]],
      src_dataset="xing et al",
      task="citation sentence generation"
    )

    result.append(res_obj)

  return result


def create_dataset_splits(data):
  random.shuffle(data)

  train_size = 0.8

  train_samples = int(len(data)*train_size)
  val_samples = int((len(data)-train_samples)/2)

  train_data = data[:train_samples]
  val_data = data[train_samples:train_samples+val_samples]
  test_data = data[train_samples+val_samples:]

  logger.info(f"# initial: {len(data)}") 
  logger.info(f"# sum splits: {len(train_data+val_data+test_data)}") 
  logger.info(f"# train: {len(train_data)}") 
  logger.info(f"# val: {len(val_data)}") 
  logger.info(f"# test: {len(test_data)}") 

  return train_data, val_data, test_data


if __name__ == "__main__":
  result = process_aburaed_et_al_2()
  result += process_chen_et_al()
  result += process_lu_et_al()
  result += process_xing_et_al()

  stats = {
    "# of samples": len(result),
    "# of related work samples": len([d for d in result if d["task"] == "related work generation"]),
    "# of citation sentence samples": len([d for d in result if d["task"] == "citation sentence generation"])
  }

  train, val, test = create_dataset_splits(result)

  store_json(result, OUTPUT_PATH+"data.json")
  logger.info(f"Stored 'result' in '{OUTPUT_PATH}'")

  store_json(stats, OUTPUT_PATH+"stats.json")
  logger.info(f"Stored 'stats' in '{OUTPUT_PATH}'")

  store_json(train, OUTPUT_PATH+"train.json")
  logger.info(f"Stored 'train' in '{OUTPUT_PATH}'")

  store_json(val, OUTPUT_PATH+"val.json")
  logger.info(f"Stored 'val' in '{OUTPUT_PATH}'")

  store_json(test, OUTPUT_PATH+"test.json")
  logger.info(f"Stored 'test' in '{OUTPUT_PATH}'")
