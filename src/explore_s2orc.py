import collections
import glob
import gzip
import io
import json
import logging
import multiprocessing
import os
import time
from typing import Tuple, Any, Dict

import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

S2ORC_PATH = "/ukp-storage-1/shared/S2ORC/20200705v1/full"
RESULTS_PATH = "/ukp-storage-1/funkquist/result.json"
ACL_SURVEY_PAPERS_PATH = "/ukp-storage-1/funkquist/acl_survey_papers.json"
COMPUTER_SCIENCE_SURVEY_PAPERS_PATH = "/ukp-storage-1/funkquist/computer_science_survey_papers.json"

NUM_PROCESSES = 8


def title_includes_search_strings(title: str):
    search_strings = ["survey"]
    match = False
    title = title.lower()

    for s in search_strings:
        if s in title:
            match = True

    return match


def filter_acl_survey_papers(metadata_json: Dict[str, Any]):

    if metadata_json["acl_id"] is None:
        return False
    if not title_includes_search_strings(metadata_json["title"]):
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
    if not title_includes_search_strings(metadata_json["title"]):
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


def explore_metadata(file_path: str) -> Tuple[collections.Counter, collections.Counter]:
    """
    Explore the metadata of the S2ORC documents in a given metadata JSONL file.

    :param file_path: file path to metadata JSONL file
    :return: (counter with statistics, counter with timings)
    """
    stat_counter = collections.Counter()
    time_counter = collections.Counter()
    acl_surveys_metadata = list()
    computer_science_metadata = list()

    tick = time.time()
    with gzip.open(file_path, "rb") as file:
        reader = io.BufferedReader(file)

        for line in reader.readlines():
            metadata = json.loads(line)

            stat_counter["# total instances"] += 1

            if metadata["acl_id"] is not None:
                stat_counter["# ACL instances"] += 1
                if title_includes_search_strings(metadata["title"]):
                    stat_counter["# ACL survey instances"] += 1

                if not metadata["has_pdf_parse"]:
                    stat_counter["# ACL instances without full text"] += 1


            if filter_acl_survey_papers(metadata):
                stat_counter["# ACL survey full text instances"] += 1
                acl_surveys_metadata.append(metadata)

            if filter_computer_science_survey_papers(metadata):
                stat_counter["# Computer Science survey full text instances"] += 1
                computer_science_metadata.append(metadata)


    tack = time.time()
    time_counter["time"] += tack - tick

    return stat_counter, time_counter, acl_surveys_metadata, computer_science_metadata


def explore_pdf_parses(file_path: str) -> Tuple[collections.Counter, collections.Counter]:
    """
    Explore the PDF parses of the S2ORC documents in a given PDF parse JSONL file.

    :param file_path: file path to PDF parse JOSNL file
    :return: (counter with statistics, counter with timings)
    """
    stat_counter = collections.Counter()
    time_counter = collections.Counter()

    tick = time.time()
    with gzip.open(file_path, "rb") as file:
        reader = io.BufferedReader(file)
        for line in reader.readlines():
            pdf_parse = json.loads(line)

            # collect statistics
            stat_counter["# instances"] += 1

            if "abstract" in pdf_parse.keys():
                stat_counter["# has abstract"] += 1

            if "body_text" in pdf_parse.keys():
                stat_counter["# has body_text"] += 1

            if "bib_entries" in pdf_parse.keys():
                stat_counter["# has bib_entries"] += 1

            if "ref_entries" in pdf_parse.keys():
                stat_counter["# has ref_entries"] += 1

    tack = time.time()
    time_counter["time"] += tack - tick

    return stat_counter, time_counter


if __name__ == "__main__":
    logger.info("Explore S2ORC.")
    logger.info(f"Running on {multiprocessing.cpu_count()} cores.")
    logger.info(f"S2ORC path: '{S2ORC_PATH}'")
    logger.info(f"Results path: '{RESULTS_PATH}'")
    logger.info(f"Number of processes: {NUM_PROCESSES}")

    # find the dataset files
    metadata_file_paths = glob.glob(os.path.join(S2ORC_PATH, "metadata", "*.jsonl.gz"))
    metadata_file_paths.sort()

    pdf_parse_file_paths = glob.glob(os.path.join(S2ORC_PATH, "pdf_parses", "*.jsonl.gz"))
    pdf_parse_file_paths.sort()

    logger.info(f"Found {len(metadata_file_paths)} metadata files and {len(pdf_parse_file_paths)} PDF parse files.")

    # gather statistics in parallel
    logger.info("Gather statistics.")

    metadata_stat_counter = collections.Counter()
    metadata_time_counter = collections.Counter()
    acl_surveys_metadata_complete = list()
    computer_science_surveys_metadata_complete = list()

    pdf_parse_stat_counter = collections.Counter()
    pdf_parse_time_counter = collections.Counter()

    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        for stat_count, time_count, acl_surveys_metadata, computer_science_metadata in tqdm.tqdm(
            p.imap(explore_metadata, metadata_file_paths), total=len(metadata_file_paths), 
            desc="Gather Metadata Statistics"):
            
            metadata_stat_counter += stat_count
            metadata_time_counter += time_count
            acl_surveys_metadata_complete += acl_surveys_metadata
            computer_science_surveys_metadata_complete += computer_science_metadata

        # for stat_count, time_count in tqdm.tqdm(p.imap(explore_pdf_parses, pdf_parse_file_paths),
        #                                         total=len(pdf_parse_file_paths), desc="Gather PDF Parse Statistics"):
        #     pdf_parse_stat_counter += stat_count
        #     pdf_parse_time_counter += time_count

    # store the results
    logger.info("Store results.")
    results = {
        "metadata": {key: metadata_stat_counter[key] for key in sorted(metadata_stat_counter.keys())},  # sort by key
        "pdf_parse": {key: pdf_parse_stat_counter[key] for key in sorted(pdf_parse_stat_counter.keys())},  # sort by key
        "metadata_timings": dict(metadata_time_counter.items()),
        "pdf_parse_timings": dict(pdf_parse_time_counter.items())
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with open(ACL_SURVEY_PAPERS_PATH, "w", encoding="utf-8") as file:
        json.dump(acl_surveys_metadata_complete, file, indent=2)

    with open(COMPUTER_SCIENCE_SURVEY_PAPERS_PATH, "w", encoding="utf-8") as file:
        json.dump(computer_science_surveys_metadata_complete, file, indent=2)

    logger.info("All done!")