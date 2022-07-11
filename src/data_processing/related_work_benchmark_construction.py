from collections import defaultdict
import re, pickle, random
from typing import List, TypedDict

from utils_package.util_funcs import load_json, load_jsonl, store_json, unique
from utils_package.logger import get_logger

logger = get_logger()

### CLOUD PATH ###
# BASE_PATH = "/ukp-storage-1/funkquist/"
### LOCAL PATH ###
BASE_PATH = "data/"

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


def get_data_aburaed_et_al(split):
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
    )

    results.append(res_obj)

  return results


def process_aburaed_et_al():
  train = list()
  train = get_data_aburaed_et_al("train")

  val = list()
  val = get_data_aburaed_et_al("val")

  test = list()
  test = get_data_aburaed_et_al("test")

  store_dataset(train, val, test, "aburaed_et_at/")


def get_data_chen_et_al(data_path):
  result = list()
  data = load_jsonl(CHEN_ET_AL_PATH+data_path+".jsonl")
  for d in data:
    res_obj = Datum(
      target=d["abs"],
      input_docs=d["multi_doc"],
    )
    result.append(res_obj)
  return result

def process_chen_et_al():
  for path in ["delve/", "s2orc/"]: 
    train = get_data_chen_et_al(path+"train")
    val = get_data_chen_et_al(path+"valid")
    test = get_data_chen_et_al(path+"test")

    store_dataset(train, val, test, "chen_et_al/"+path)    


def get_lu_et_al_data(split):
  result = list()
  data = load_json(LU_ET_AL_PATH+split+".json")
  for d in data:
    res_obj = Datum(
      target=d["related_work"],
      abstract=d["abstract"],
      input_docs=[r["abstract"] for r in d["ref_abstract"].values()],
    )
    result.append(res_obj)
  return result

def process_lu_et_al():
  train = get_lu_et_al_data("train")
  val = get_lu_et_al_data("val")
  test = get_lu_et_al_data("test")

  store_dataset(train, val, test, "lu_et_al/")


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


def get_xing_et_al_data(data, dataset):
  result = list()
  for d in data:
    if dataset == "explicit":
      res_obj = Datum(
        target=d["explicit_citation"],
        abstract=d["tgt_abstract"],
        input_docs=[d["src_abstract"]],
        context_before=d["text_before_explicit_citation"],
        context_after=d["text_after_explicit_citation"],
      )

    if dataset == "hr":
      res_obj = Datum(
        target=d["implicit_citation_0.1"],
        abstract=d["tgt_abstract"],
        input_docs=[d["src_abstract"]],
        context_before=d["text_before_implicit_citation_0.1"],
        context_after=d["text_after_implicit_citation_0.1"],
      )

    if dataset == "hp":
      res_obj = Datum(
        target=d["implicit_citation_0.9"],
        abstract=d["tgt_abstract"],
        input_docs=[d["src_abstract"]],
        context_before=d["text_before_implicit_citation_0.9"],
        context_after=d["text_after_implicit_citation_0.9"],
      )

    result.append(res_obj)

  return result

def process_xing_et_al():
  # This dataset also contains context before and after
  # the citation and distinguishes between explicit citations
  # e.g. citations where the citation is in the sentence
  # and implicit citations where the citation might not be in the sentence
  # It also has the abstract of the target paper

  data = load_jsonl(XING_ET_AL_PATH+"citation.jsonl")

  train_input = [d for d in data if d["train_or_test"] == "train"]
  test_input = [d for d in data if d["train_or_test"] == "test"]

  for ds in ["explicit", "hp", "hr"]:
    train = get_xing_et_al_data(train_input, ds)
    val = []
    test = get_xing_et_al_data(test_input, ds)

    store_dataset(train, val, test, "xing_et_al/"+ds+"/")


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


def store_dataset(train: list, val: list, test: list, folder: str):
  train_file = OUTPUT_PATH+folder+"train.json"
  store_json(train, train_file)
  logger.info(f"Stored 'train' in '{train_file}'")

  val_file = OUTPUT_PATH+folder+"val.json"
  store_json(val, val_file)
  logger.info(f"Stored 'val' in '{val_file}'")

  test_file = OUTPUT_PATH+folder+"test.json"
  store_json(test, test_file)
  logger.info(f"Stored 'test' in '{test_file}'")


if __name__ == "__main__":
  process_aburaed_et_al()
  process_chen_et_al()
  process_lu_et_al()
  process_xing_et_al()

  # stats = {
  #   "# of samples": len(result),
  #   "# of related work samples": len([d for d in result if d["task"] == "related work generation"]),
  #   "# of citation sentence samples": len([d for d in result if d["task"] == "citation sentence generation"])
  # }

  # train, val, test = create_dataset_splits(result)

  # store_json(result, OUTPUT_PATH+"data.json")
  # logger.info(f"Stored 'result' in '{OUTPUT_PATH}'")

  # store_json(stats, OUTPUT_PATH+"stats.json")
  # logger.info(f"Stored 'stats' in '{OUTPUT_PATH}'")

  # store_json(train, OUTPUT_PATH+"train.json")
  # logger.info(f"Stored 'train' in '{OUTPUT_PATH}'")

  # store_json(val, OUTPUT_PATH+"val.json")
  # logger.info(f"Stored 'val' in '{OUTPUT_PATH}'")

  # store_json(test, OUTPUT_PATH+"test.json")
  # logger.info(f"Stored 'test' in '{OUTPUT_PATH}'")
