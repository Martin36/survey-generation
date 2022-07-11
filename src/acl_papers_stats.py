
from collections import defaultdict
import pandas as pd
from utils_package.util_funcs import load_json, store_json
from utils_package.logger import get_logger

logger = get_logger()

DATA_FILE = "data/acl_metadata_survey_papers.json"
OUTPUT_FILE = "stats/acl_papers/section_title_stats.json"
TITLES_FILE = "stats/acl_papers/titles.json"

def get_stats(data):
	stats = defaultdict(int)
	for d in data:
		stats["# instances"] += 1

		if d["abstract"]:
			stats["# instances with abstract"] += 1
			
		if d["has_pdf_body_text"]:
			stats["# instances with body text"] += 1

	return stats

def get_titles(data):
	titles = list()
	for d in data:
		title = d["title"]
		titles.append(title)
	return titles

def store_stats(stats, file):
	# Convert json to table
	stats_transformed = {
		"section_title": list(stats.keys()),
		"count": list(stats.values())
	}

	df = pd.DataFrame.from_dict(stats_transformed)
	df.to_csv(file)

	logger.info(f"Stored stats in '{file}'")


if __name__ == "__main__":
	data = load_json(DATA_FILE)

	stats = get_stats(data)
	titles = get_titles(data)

	store_json(stats, OUTPUT_FILE)
	logger.info(f"Stored 'stats' in '{OUTPUT_FILE}'")

	store_json(titles, TITLES_FILE)
	logger.info(f"Stored 'titles' in '{TITLES_FILE}'")
	
