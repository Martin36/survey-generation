from collections import defaultdict
from pybtex.database.input import bibtex
from utils_package.util_funcs import store_json
from utils_package.logger import get_logger

logger = get_logger()

BIB_PATH = "data/anthology.bib"
OUTPUT_PATH = "stats/anthology.json"
YEAR_DIST_PATH = "stats/anthology_year_dist.json"
SURVEY_TITLES_PATH = "stats/anthology_survey_titles.json"

if __name__ == "__main__":

  parser = bibtex.Parser()
  bib_data = parser.parse_file(BIB_PATH)

  year_dist = defaultdict(int)
  stats = defaultdict(int)
  survey_titles = list()

  for entry in bib_data.entries.values():
    title = entry.fields["title"]
    year = entry.fields["year"]

    year_dist[year] += 1

    stats["# of papers"] += 1

    if "survey" in title.lower():
      stats["# of survey papers"] += 1
      survey_titles.append(title)

  store_json(stats, OUTPUT_PATH)
  logger.info(f"Stored 'stats' in '{OUTPUT_PATH}'")

  store_json(year_dist, YEAR_DIST_PATH)
  logger.info(f"Stored 'year_dist' in '{YEAR_DIST_PATH}'")

  store_json(survey_titles, SURVEY_TITLES_PATH)
  logger.info(f"Stored 'survey_titles' in '{SURVEY_TITLES_PATH}'")
