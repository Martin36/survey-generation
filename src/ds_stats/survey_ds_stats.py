
from collections import defaultdict
import pandas as pd
from utils_package.util_funcs import load_json
from utils_package.logger import get_logger

logger = get_logger()

ACL_SAMPLE_DATA_FILE = "acl_pdf_parse_survey_papers.json"
ACL_JSON_OUTPUT_FILE = "stats/acl_papers/section_title_stats.json"
ACL_TABLE_OUTPUT_FILE = "stats/acl_papers/section_title_stats.csv"

CS_SAMPLE_DATA_FILE = "cs_pdf_parse_survey_papers.json"
CS_JSON_OUTPUT_FILE = "stats/cs_papers/section_title_stats.json"
CS_TABLE_OUTPUT_FILE = "stats/cs_papers/section_title_stats.csv"


def get_stats(sample_data):
	stats = defaultdict(int)
	for d in sample_data:
		section_names = [body_part["section"] for body_part in d["body_text"]]

		for name in section_names:
			stats[name] += 1

	# Sort the stats by value in descending order
	stats = {k: v for k, v in sorted(
		stats.items(), key=lambda item: item[1], reverse=True)}
	return stats


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
	acl_sample_data = load_json(ACL_SAMPLE_DATA_FILE)
	cs_sample_data = load_json(CS_SAMPLE_DATA_FILE)

	acl_stats = get_stats(acl_sample_data)
	cs_stats = get_stats(cs_sample_data)

	store_stats(acl_stats, ACL_TABLE_OUTPUT_FILE)
	store_stats(cs_stats, CS_TABLE_OUTPUT_FILE)
