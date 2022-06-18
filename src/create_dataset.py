from collections import defaultdict
from utils_package.util_funcs import load_json, store_json
from utils_package.logger import get_logger
from iteration_utilities import unique_everseen

logger = get_logger()


PFD_PARSES_FILE = "acl_pdf_parse_survey_papers.json"
METADATA_FILE = "acl_metadata_survey_papers.json"
# OUTPUT FILES
STATS_FILE = "acl_papers/stats.json"
DATASET_FILE = "acl_papers/data.json"
REFERENCE_PAPERS_FILE = "acl_papers/reference_papers.json"

SECTION_HEADINGS_TO_INCLUDE = [""]  # TODO:

if __name__ == "__main__":
	pdf_parses = load_json(PFD_PARSES_FILE)
	metadata = load_json(METADATA_FILE)

	results = list()
	reference_papers = list()
	stats = defaultdict(int)
	nr_of_citations_dist = defaultdict(int)
	paragraph_length_dist = defaultdict(int)
	citations_total = 0         # Will contain duplicates
	citations_not_in_s2orc = 0  # Will contain duplicates

	for pdf_parse, metadatum in zip(pdf_parses, metadata):
		title = metadatum["title"]
		body_texts = pdf_parse["body_text"]
		bib_entries = pdf_parse["bib_entries"]

		for body_text in body_texts:
			section = body_text["section"]
			text = body_text["text"]
			cite_spans = body_text["cite_spans"]
			nr_of_citations = len(cite_spans)
			has_cite_spans = nr_of_citations > 0

			has_some_missing_cite_span = False
			for cite_span in cite_spans:
				bib_entry = bib_entries[cite_span["ref_id"]]
				cite_span["title"] = bib_entry["title"]
				cite_span["link"] = bib_entry["link"]

				if not cite_span["link"]:
					citations_not_in_s2orc += 1
					has_some_missing_cite_span = True

				citations_total += 1

				reference_papers.append({
					"title": cite_span["title"],
					"link": cite_span["link"]
				})

			text_len = len(text)

			nr_of_citations_dist[nr_of_citations] += 1
			paragraph_length_dist[text_len] += 1

			res_obj = {
					"title": title,
					"section": section,
					"text": text,
					"cite_spans": cite_spans,
					"has_cite_spans": has_cite_spans,
					"has_some_missing_cite_span": has_some_missing_cite_span
					# TODO: Include the cited papers in each datum?
			}

			results.append(res_obj)

	unique_reference_papers = list(unique_everseen(reference_papers))

	stats["# citations distribution"] = nr_of_citations_dist
	stats["paragraph length distribution"] = paragraph_length_dist
	stats["# data points"] = len(results)
	stats["# data points with citations"] = len(
			[d for d in results if d["has_cite_spans"]])
	stats["# data points with missing citations"] = len(
			[d for d in results if d["has_some_missing_cite_span"]])
	stats["# unique reference papers"] = len(unique_reference_papers)
	stats["citations not in s2orc"] = citations_not_in_s2orc
	stats["citations total"] = citations_total
	store_json(stats, STATS_FILE)
	logger.info(f"Stored stats in '{STATS_FILE}'")

	store_json(results, DATASET_FILE)
	logger.info(f"Stored results in '{DATASET_FILE}'")

	store_json(unique_reference_papers, REFERENCE_PAPERS_FILE)
	logger.info(f"Stored reference papers in '{REFERENCE_PAPERS_FILE}'")

