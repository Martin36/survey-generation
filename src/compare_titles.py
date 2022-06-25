from collections import defaultdict
from utils_package.util_funcs import load_json, store_json
from utils_package.logger import get_logger

logger = get_logger()


ANTHOLOGY_TITLES_PATH = "stats/anthology_survey_titles.json"
S2ORC_TITLES_PATH = "stats/acl_papers/titles.json"
AAN_TITLES_PATH = "stats/aan_corpus.json"
OUTPUT_PATH = "stats/overlapping_titles.json"

if __name__ == "__main__":
  anthology_data = load_json(ANTHOLOGY_TITLES_PATH)
  s2orc_data = load_json(S2ORC_TITLES_PATH)
  aan_data = load_json(AAN_TITLES_PATH)
  aan_titles = [d["title"] for d in aan_data["survey papers"]]

  stats = defaultdict(int)
  titles = set()
  titles_not_in_anthology = list()
  all_titles = anthology_data+s2orc_data+aan_titles
  all_titles_unique = list(set(all_titles))

  stats["# titles in all"] = len(all_titles)
  stats["# unique titles"] = len(all_titles_unique)

  for d in anthology_data:
    if d in s2orc_data:
      stats["# anthology titles in S2ORC"] += 1
      titles.add(d)
    
    if d in aan_titles:
      stats["# anthology titles in AAN"] += 1
  
  for d in s2orc_data:
    if d in anthology_data:
      stats["# S2ORC titles in anthology"] += 1
      titles.add(d)
    else:
      titles_not_in_anthology.append(d)

    if d in aan_titles:
      stats["# S2ORC titles in AAN"] += 1

  for d in aan_titles:
    if d not in s2orc_data and d not in anthology_data:
      stats["# aan titles in neither"] += 1

  stats["titles"] = list(titles)
  stats["titles NOT in anthology"] = titles_not_in_anthology

  store_json(stats, OUTPUT_PATH)
  logger.info(f"Stored 'stats' in '{OUTPUT_PATH}'")

