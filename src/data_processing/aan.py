from collections import defaultdict
from glob import glob
from utils_package.util_funcs import load_json, store_json
from utils_package.logger import get_logger

from utils import title_includes_search_strings

logger = get_logger()


ANN_PAPERS_PATH = "data/aan/papers_text/*"
ACL_METADATA_PATH = "data/aan/release/2014/acl-metadata.txt"
OUTPUT_PATH = "stats/aan_corpus.json"


def parse_line(line):
  line_split = line.split("=")
  key = line_split[0].strip()
  value = line_split[1].strip()
  value = value.replace("{", "")
  value = value.replace("}", "")
  return (key, value)


if __name__ == "__main__":
  files = glob(ANN_PAPERS_PATH)
  txt_files = [f for f in files if ".txt" in f]

  with open(ACL_METADATA_PATH, encoding = "ISO-8859-1") as f:
    lines = f.readlines()
    counter = 0
    papers = list()
    survey_papers = list()
    paper = dict()
    stats = defaultdict(int)
    
    for line in lines:
      if line == "\n":
        # Finished parsing one paper
        stats["# papers"] += 1
        
        if title_includes_search_strings(paper["title"]):
          stats["# survey papers"] += 1
          survey_papers.append(paper)
        
        papers.append(paper)
        paper = dict()
      else:
        key, value = parse_line(line)
        paper[key] = value

  stats["survey papers"] = survey_papers

  print(f'# papers: {stats["# papers"]}')
  print(f'# survey papers: {stats["# survey papers"]}')

  store_json(stats, OUTPUT_PATH)
  logger.info(f"Stored 'stats' in '{OUTPUT_PATH}'")


