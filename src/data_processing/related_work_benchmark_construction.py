import re
from typing import List, TypedDict

from utils_package.util_funcs import load_json, load_jsonl


ABURAED_ET_AL_2_PATH = "data/related_work/aburaed_et_al_2/"
CHEN_ET_AL_PATH = "data/related_work/chen_et_al/"

### TARGET DATA STRUCTURE ###

class Datum(TypedDict):
  target: str             # The target related work section/citation sentence
  input_docs: List[str]   # List of text from cited papers
  src_dataset: str        # Name of the dataset where the data sample comes from
  task: str               # Name of original task e.g. related work generation or citation generation


def process_aburaed_et_al_2():
  test_src_file = "test.txt.src"
  test_tgt_file = "test.txt.tgt.tagged"

  src = open(ABURAED_ET_AL_2_PATH+test_src_file, "rb").read().decode("utf8").splitlines()
  tgt = open(ABURAED_ET_AL_2_PATH+test_tgt_file, "rb").read().decode("utf8").splitlines()

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

def process_chen_et_al():
  delve_path = "delve/"
  s2orc_path = "s2orc/"

  s2orc_test = load_jsonl(CHEN_ET_AL_PATH+s2orc_path+"valid.jsonl")

  result = list()

  for d in s2orc_test:
    print(d)

    res_obj = Datum(
      target=d["abs"],
      input_docs=d["multi_doc"],
      src_dataset="chen et al",
      task="related work generation"
    )

    result.append(res_obj)

  return result


if __name__ == "__main__":
  # process_aburaed_et_al_2()
  process_chen_et_al()
