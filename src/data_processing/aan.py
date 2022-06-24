from collections import defaultdict
from glob import glob
from utils_package.util_funcs import load_json, store_json
from utils_package.logger import get_logger

logger = get_logger()


ANN_PAPERS_PATH = "data/aan/papers_text/*"
ACL_METADATA_PATH = "data/aan/release/2014/acl-metadata.txt"
OUTPUT_PATH = "stats/overlapping_titles.json"


def parse_line(line):
  line_split = line.split("=")
  key = line_split[0].strip()
  value = line_split[1].strip()
  return (key, value)

if __name__ == "__main__":
  files = glob(ANN_PAPERS_PATH)
  txt_files = [f for f in files if ".txt" in f]

  with open(ACL_METADATA_PATH, encoding = "ISO-8859-1") as f:
    lines = f.readlines()
    counter = 0
    papers = list()
    paper = dict()
    
    for line in lines:
      if line == "\n":
        # New paper
        papers.append(paper)
        paper = dict()

      key, value = parse_line(line)
      paper[key] = value

    # TODO: Look for survey in the titles

