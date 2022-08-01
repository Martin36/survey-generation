import collections, glob, gzip, io, json, logging, multiprocessing, os, time, tqdm
from typing import Any, Dict
from utils_package.util_funcs import store_json

from utils import match_search_terms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

S2ORC_PATH = "/storage/ukp/shared/S2ORC/20200705v1/full"
RESULTS_PATH = "stats/s2orc/result_inc_abs.json"
ACL_SURVEY_PAPERS_PATH = "stats/s2orc/acl_survey_papers_inc_abs.json"
COMPUTER_SCIENCE_SURVEY_PAPERS_PATH = "stats/s2orc/computer_science_survey_papers.json"
SURVEY_CITATIONS_DIST_PATH = "stats/s2orc/survey_citation_dist.json"
OTHER_CITATION_DIST_PATH = "stats/s2orc/other_citation_dist.json"

NUM_PROCESSES = 8

DEBUGGING = False


def filter_acl_survey_papers(metadata_json: Dict[str, Any]):

    if metadata_json["acl_id"] is None:
        return False
    if not match_search_terms(metadata_json["title"]) and \
        not match_search_terms(metadata_json["abstract"]):
        return False
    if not metadata_json["has_pdf_parse"]:
        return False
    if not metadata_json["has_pdf_parsed_abstract"]:
        return False
    if not metadata_json["has_pdf_parsed_body_text"]:
        return False
    if not metadata_json["has_pdf_parsed_bib_entries"]:
        return False
    if not metadata_json["has_pdf_parsed_ref_entries"]:
        return False

    return True


def filter_computer_science_survey_papers(metadata_json: Dict[str, Any]):

    if not metadata_json["mag_field_of_study"]:
        return False
    if "Computer Science" not in metadata_json["mag_field_of_study"]:
        return False
    if not match_search_terms(metadata_json["title"]):
        return False
    if not metadata_json["has_pdf_parse"]:
        return False
    if not metadata_json["has_pdf_parsed_abstract"]:
        return False
    if not metadata_json["has_pdf_parsed_body_text"]:
        return False
    if not metadata_json["has_pdf_parsed_bib_entries"]:
        return False
    if not metadata_json["has_pdf_parsed_ref_entries"]:
        return False

    return True


def explore_metadata(file_path: str):
    """
    Explore the metadata of the S2ORC documents in a given metadata JSONL file.

    :param file_path: file path to metadata JSONL file
    :return: (counter with statistics, counter with timings)
    """
    stat_counter = collections.Counter()
    survey_citation_dist = collections.defaultdict(int)
    other_citation_dist = collections.defaultdict(int)
    time_counter = collections.Counter()
    acl_surveys_metadata = list()
    computer_science_metadata = list()

    tick = time.time()

    def process_file(file):
        reader = io.BufferedReader(file)

        for line in reader.readlines():
            metadata = json.loads(line)

            stat_counter["# total instances"] += 1

            if metadata["acl_id"] is not None:
                stat_counter["# ACL instances"] += 1
                if match_search_terms(metadata["title"]):
                    stat_counter["# ACL survey instances matched with title"] += 1
                    survey_citation_dist[len(metadata["outbound_citations"])] += 1
                elif match_search_terms(metadata["abstract"]):
                    stat_counter["# ACL survey instances matched with abstract"] += 1
                    survey_citation_dist[len(metadata["outbound_citations"])] += 1
                else:
                    other_citation_dist[len(metadata["outbound_citations"])] += 1

                if not metadata["has_pdf_parse"]:
                    stat_counter["# ACL instances without full text"] += 1


            if filter_acl_survey_papers(metadata):
                stat_counter["# ACL survey full text instances"] += 1
                acl_surveys_metadata.append(metadata)

            if filter_computer_science_survey_papers(metadata):
                stat_counter["# Computer Science survey full text instances"] += 1
                computer_science_metadata.append(metadata)

    if file_path.split(".")[-1] == "gz":
        with gzip.open(file_path, "rb") as file:
            process_file(file)
    else:
        with open(file_path, "rb") as file:
            process_file(file)
    
    # stat_counter["citation distribution for title matched papers"] = survey_citation_dist
    stat_counter["average number of citations for title matched papers"] = calc_citations_distribution_average(survey_citation_dist)
    # stat_counter["citation distribution for other ACL papers"] = other_citation_dist
    stat_counter["average number of citations for other ACL papers"] = calc_citations_distribution_average(other_citation_dist)

    tack = time.time()
    time_counter["time"] += tack - tick

    result = {
        "stat_counter": stat_counter,
        "time_counter": time_counter,
        "acl_surveys_metadata": acl_surveys_metadata,
        "computer_science_metadata": computer_science_metadata,
        "survey_citation_dist": survey_citation_dist,
        "other_citation_dist": other_citation_dist
    }

    return result


def merge_dicts(x: dict, y: dict):
    """
        Merges two dictionaries. If the same key exists in both, the values are added together.
        NOTE: The values should only be numbers.
    """
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}

def calc_citations_distribution_average(dist: dict):
    total_citations = 0
    total_papers = 0

    for citations, nr_of_papers in dist.items():
        total_citations += int(citations)*nr_of_papers
        total_papers += nr_of_papers

    if total_papers == 0:
        return 0
    
    avg_citations = int(total_citations/total_papers)

    return avg_citations


if __name__ == "__main__":
    logger.info("Explore S2ORC.")
    logger.info(f"Running on {multiprocessing.cpu_count()} cores.")
    logger.info(f"S2ORC path: '{S2ORC_PATH}'")
    logger.info(f"Results path: '{RESULTS_PATH}'")
    logger.info(f"Number of processes: {NUM_PROCESSES}")

    if DEBUGGING:
        metadata_file_paths = ["data/s2orc/metadata/sample.jsonl"]
    else: 
        metadata_file_paths = glob.glob(os.path.join(S2ORC_PATH, "metadata", "*.jsonl.gz"))
        metadata_file_paths.sort()

    logger.info(f"Found {len(metadata_file_paths)} metadata files")

    metadata_stat_counter = collections.Counter()
    metadata_time_counter = collections.Counter()
    acl_surveys_metadata = list()
    computer_science_surveys_metadata = list()
    survey_citation_dist = dict()
    other_citation_dist = dict()

    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        for result in tqdm.tqdm(
            p.imap(explore_metadata, metadata_file_paths), total=len(metadata_file_paths), 
            desc="Gather Metadata Statistics"):
            
            metadata_stat_counter += result["stat_counter"]
            metadata_time_counter += result["time_counter"]
            acl_surveys_metadata += result["acl_surveys_metadata"]
            computer_science_surveys_metadata += result["computer_science_metadata"]
            survey_citation_dist = merge_dicts(survey_citation_dist, result["survey_citation_dist"]) 
            other_citation_dist = merge_dicts(other_citation_dist, result["other_citation_dist"]) 

    # Divide by the number of files since this average gets added for each metadata file
    metadata_stat_counter["average number of citations for other ACL papers"] = metadata_stat_counter["average number of citations for other ACL papers"]/len(metadata_file_paths)
    metadata_stat_counter["average number of citations for title matched papers"] = metadata_stat_counter["average number of citations for title matched papers"]/len(metadata_file_paths)

    logger.info("Store results.")
    results = {
        "metadata": {key: metadata_stat_counter[key] for key in sorted(metadata_stat_counter.keys())},  # sort by key
        "metadata_timings": dict(metadata_time_counter.items()),
    }

    store_json(results, RESULTS_PATH)
    logger.info(f"Stored 'results' in '{RESULTS_PATH}'")

    store_json(acl_surveys_metadata, ACL_SURVEY_PAPERS_PATH)
    logger.info(f"Stored 'acl_surveys_metadata' in '{ACL_SURVEY_PAPERS_PATH}'")

    store_json(computer_science_surveys_metadata, COMPUTER_SCIENCE_SURVEY_PAPERS_PATH)
    logger.info(f"Stored 'computer_science_surveys_metadata' in '{COMPUTER_SCIENCE_SURVEY_PAPERS_PATH}'")

    store_json(survey_citation_dist, SURVEY_CITATIONS_DIST_PATH)
    logger.info(f"Stored 'survey_citation_dist' in '{SURVEY_CITATIONS_DIST_PATH}'")

    store_json(other_citation_dist, OTHER_CITATION_DIST_PATH)
    logger.info(f"Stored 'other_citation_dist' in '{OTHER_CITATION_DIST_PATH}'")

    logger.info("All done!")
