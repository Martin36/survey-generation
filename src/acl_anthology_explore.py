from collections import defaultdict
from pybtex.database.input import bibtex
from tqdm import tqdm
from utils_package.util_funcs import store_json
from utils_package.logger import get_logger

logger = get_logger()

BIB_PATH = "data/anthology+abstracts.bib"
# BIB_PATH = "data/anthology.bib"
OUTPUT_PATH = "stats/anthology.json"
YEAR_DIST_PATH = "stats/anthology_year_dist.json"
PAPERS_PATH = "stats/anthology_matched_papers.json"


def match_search_terms(text: str):
  search_terms = ["survey", "review", "overview", "shared task"]
  for term in search_terms:
    if term in text:
      return True
  return False      


if __name__ == "__main__":

  parser = bibtex.Parser()
  bib_data = parser.parse_file(BIB_PATH)

  year_dist = defaultdict(int)
  stats = defaultdict(int)
  paper_objs = list()

  for entry in tqdm(bib_data.entries.values()):
    title = entry.fields["title"]
    year = entry.fields["year"]
    abstract = entry.fields.get("abstract", None)

    year_dist[year] += 1

    stats["# of papers"] += 1

    if match_search_terms(title.lower()):
      stats["# of survey papers"] += 1
      stats["# matched with title"] += 1
      paper_objs.append({
        "title": title,
        "year": year,
        "abstract": abstract,
        "matched_by": "title",
      })
      continue

    if not abstract:
      stats["# of papers without abstract"] += 1
    else:
      if match_search_terms(abstract.lower()):
        stats["# of survey papers"] += 1
        stats["# matched with abstract"] += 1
        paper_objs.append({
          "title": title,
          "year": year,
          "abstract": abstract,
          "matched_by": "abstract",
        })


  store_json(stats, OUTPUT_PATH)
  logger.info(f"Stored 'stats' in '{OUTPUT_PATH}'")

  store_json(year_dist, YEAR_DIST_PATH)
  logger.info(f"Stored 'year_dist' in '{YEAR_DIST_PATH}'")

  store_json(paper_objs, PAPERS_PATH)
  logger.info(f"Stored 'paper_objs' in '{PAPERS_PATH}'")
