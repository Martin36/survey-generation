from collections import defaultdict
from utils_package.util_funcs import load_json, store_json
from utils_package.logger import get_logger

logger = get_logger()


ANTHOLOGY_TITLES_PATH = "stats/anthology_survey_titles.json"
S2ORC_TITLES_PATH = "stats/acl_papers/titles.json"
OUTPUT_PATH = "stats/overlapping_titles.json"

if __name__ == "__main__":
  anthology_data = load_json(ANTHOLOGY_TITLES_PATH)
  s2orc_data = load_json(S2ORC_TITLES_PATH)

  stats = defaultdict(int)
  titles = set()
  titles_not_in_anthology = list()

  for d in anthology_data:
    if d in s2orc_data:
      stats["# anthology titles in S2ORC"] += 1
      titles.add(d)
  
  for d in s2orc_data:
    if d in anthology_data:
      stats["# S2ORC titles in anthology"] += 1
      titles.add(d)
    else:
      titles_not_in_anthology.append(d)

  stats["titles"] = list(titles)
  stats["titles NOT in anthology"] = titles_not_in_anthology

  store_json(stats, OUTPUT_PATH)
  logger.info(f"Stored 'stats' in '{OUTPUT_PATH}'")

