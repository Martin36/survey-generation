import re
from typing import List, TypedDict

from utils_package.util_funcs import load_json, load_jsonl, store_jsonl
from utils_package.logger import get_logger

logger = get_logger()

BASE_PATH = "data/"

ABURAED_ET_AL_2_PATH = BASE_PATH+"related_work/aburaed_et_al_2/"
CHEN_ET_AL_PATH = BASE_PATH+"related_work/chen_et_al/"
LU_ET_AL_PATH = BASE_PATH+"related_work/lu_et_al/"
XING_ET_AL_PATH = BASE_PATH+"related_work/xing_et_al/"

OUTPUT_PATH = BASE_PATH+"related_work/benchmark/"

class Datum(TypedDict):
  target: str             # The target related work section/citation sentence
  input: str              # The unified format of the input


def convert_input_docs_to_unified_format(input_docs: List[str]):
  result = ""
  for doc in input_docs:
    result += doc[0] + "DOC_" + doc[1:] + " "
  return result.strip()

def remove_leading_brackets(input_docs: List[str]):
  return [doc[4:] for doc in input_docs]


def get_data_aburaed_et_al(split):
  src_ending = ".txt.src"
  tgt_ending = ".txt.tgt.tagged"

  src = open(ABURAED_ET_AL_2_PATH+split+src_ending, "rb").read().decode("utf8").splitlines()
  tgt = open(ABURAED_ET_AL_2_PATH+split+tgt_ending, "rb").read().decode("utf8").splitlines()

  results = list()

  for sline, tline in zip(src,tgt):
    tline = re.sub("(\([^()]*\d{2,4}[^()]*\))|([A-Z]\w+\s*.{1,8}\([^()]*\d{4}[^()]*\))|([A-Z]\w+\s*\([^()]*\d{4}[^()]*\))|([A-Z]\w+\s*and\s*[A-Z]\w+\s*\([^()]*\d{4}\))|(\[[^\[\]]*\d{1,4}[^\[\]]*\])", "<CITE>", tline)
    tline = re.sub("\d{1}", "#", tline)
    tline = re.sub("<t>", "", tline)
    tline = re.sub("<\/t>", "", tline)
    tline = re.sub("<cite>", "[0]", tline)
    tline = tline.strip()
    
    sline = re.sub("\d{1}", "#", sline)
    # sline = "[0] " + sline
    sline = sline.strip()

    res_obj = Datum(
      target=tline,
      #input=convert_input_docs_to_unified_format([sline]),
      input=[sline],
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
      #input=convert_input_docs_to_unified_format(d["multi_doc"]),
      input=remove_leading_brackets(d["multi_doc"]),
    )
    result.append(res_obj)
  return result

def process_chen_et_al():
  for path in ["delve/", "s2orc/"]: 
    train = get_data_chen_et_al(path+"train")
    val = get_data_chen_et_al(path+"valid")
    test = get_data_chen_et_al(path+"test")

    store_dataset(train, val, test, "chen_et_al/"+path)  

    del train, val, test  


def replace_cite_with_nr(text: str, cite_to_nr: dict):
  for k, v in cite_to_nr.items():
    text = text.replace(k, v)
  return text

def get_lu_et_al_data(split):
  result = list()
  data = load_json(LU_ET_AL_PATH+split+".json")
  for d in data:
    cite_to_nr = {k: f"[{i}]" for i, k in enumerate(d["ref_abstract"].keys())}
    target = replace_cite_with_nr(d["related_work"], cite_to_nr)
    input_docs=[f'{cite_to_nr[k]} {r["abstract"]}' for k,r in d["ref_abstract"].items()]
    res_obj = Datum(
      target=target,
      #input=convert_input_docs_to_unified_format(input_docs),
      input=remove_leading_brackets(input_docs),
    )
    result.append(res_obj)
  return result

def process_lu_et_al():
  train = get_lu_et_al_data("train")
  val = get_lu_et_al_data("val")
  test = get_lu_et_al_data("test")

  store_dataset(train, val, test, "lu_et_al/")


def replace_xing_et_al_references(text: str):
  text = re.sub("#REFR", "[0]", text)
  return text

def get_xing_et_al_data(data, dataset):
  result = list()
  for d in data:
    if dataset == "explicit":
      target = replace_xing_et_al_references(d["explicit_citation"])
    if dataset == "hr":
      target = replace_xing_et_al_references(d["implicit_citation_0.1"])
    if dataset == "hp":
      target = replace_xing_et_al_references(d["implicit_citation_0.9"])

    # TODO: Remove #OTHERREF from input data?
    # input_docs = ["[0] " + d["src_abstract"]]
    input_docs = [d["src_abstract"].strip()]
    res_obj = Datum(
      target=target,
      # input=convert_input_docs_to_unified_format(input_docs),
      input=input_docs,
    )
    result.append(res_obj)

  return result

def process_xing_et_al():
  data = load_jsonl(XING_ET_AL_PATH+"citation.jsonl")

  train_input = [d for d in data if d["train_or_test"] == "train"]
  test_input = [d for d in data if d["train_or_test"] == "test"]

  for ds in ["explicit", "hp", "hr"]:
    train = get_xing_et_al_data(train_input, ds)
    val = []
    test = get_xing_et_al_data(test_input, ds)

    store_dataset(train, val, test, "xing_et_al/"+ds+"/")


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
  process_aburaed_et_al()
  process_chen_et_al()
  process_lu_et_al()
  process_xing_et_al()