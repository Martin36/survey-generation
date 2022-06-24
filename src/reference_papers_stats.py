
from collections import defaultdict
import pandas as pd
from utils_package.util_funcs import load_json, store_json
from utils_package.logger import get_logger

logger = get_logger()

METADATA_FILE = "data/reference_papers_metadata.json"
PDF_PARSES_FILE = "data/reference_papers_pdf_parses.json"
OUTPUT_FILE = "data/reference_papers_stats_from_extracted.json"

def get_stats(metadata):
  stats = defaultdict(int)

  for d in metadata:
    stats["# instances"] += 1

    if d["has_pdf_parse"]:
      stats["# instances with pdf"] += 1
          
    if d["abstract"]:
      stats["# instances with abstract"] += 1

  return stats


if __name__ == "__main__":
	metadata = load_json(METADATA_FILE)
	# pdf_parse_data = load_json(PDF_PARSES_FILE)

	metadata_stats = get_stats(metadata)

	store_json(metadata_stats, OUTPUT_FILE)
