import collections, gzip, time, io, json, logging, multiprocessing, glob, os, tqdm

from utils_package.util_funcs import load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

S2ORC_PATH = "/ukp-storage-1/shared/S2ORC/20200705v1/full"
REFERENCE_PAPERS_FILE = "/ukp-storage-1/funkquist/reference_papers.json"
METADATA_RESULTS_PATH = "/ukp-storage-1/funkquist/reference_papers_metadata.json"
PDF_PARSE_RESULTS_PATH = "/ukp-storage-1/funkquist/reference_papers_pdf_parses.json"
STATS_RESULTS_PATH = "/ukp-storage-1/funkquist/reference_papers_stats.json"

NUM_PROCESSES = 8

reference_papers = load_json(REFERENCE_PAPERS_FILE)
reference_paper_ids = [p["link"] for p in reference_papers]


def explore_metadata(file_path: str):

  stat_counter = collections.Counter()
  time_counter = collections.Counter()
  reference_papers_metadata = list()

  tick = time.time()
  with gzip.open(file_path, "rb") as file:
    reader = io.BufferedReader(file)

    for line in reader.readlines():
      metadata = json.loads(line)

      stat_counter["# instances"] += 1

      if metadata["paper_id"] in reference_paper_ids:
        reference_papers_metadata.append(metadata)

        if metadata["has_pdf_parse"]:
          stat_counter["# instances with pdf"] += 1
          
        if metadata["abstract"]:
          stat_counter["# instances with abstract"] += 1
      

  tack = time.time()
  time_counter["time"] += tack - tick

  return stat_counter, time_counter, reference_papers_metadata


def explore_pdf_parses(file_path: str):

  stat_counter = collections.Counter()
  time_counter = collections.Counter()
  reference_papers_pdf_parses = list()

  tick = time.time()
  with gzip.open(file_path, "rb") as file:
    reader = io.BufferedReader(file)
    for line in reader.readlines():
      pdf_parse = json.loads(line)

      if pdf_parse["paper_id"] in reference_paper_ids:
        reference_papers_pdf_parses.append(pdf_parse)

        # collect statistics
        stat_counter["# instances"] += 1

        if len(pdf_parse["abstract"]):
          stat_counter["# has abstract"] += 1

        if len(pdf_parse["body_text"]):
          stat_counter["# has body text"] += 1

  tack = time.time()
  time_counter["time"] += tack - tick

  return stat_counter, time_counter, reference_papers_pdf_parses


if __name__ == "__main__":
  logger.info("Explore S2ORC.")
  logger.info(f"Running on {multiprocessing.cpu_count()} cores.")
  logger.info(f"S2ORC path: '{S2ORC_PATH}'")
  logger.info(f"Metadata results path: '{METADATA_RESULTS_PATH}'")
  logger.info(f"PDF parses results path: '{PDF_PARSE_RESULTS_PATH}'")
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
  reference_papers_metadata_complete = list()
  reference_papers_pdf_parses_complete = list()

  pdf_parse_stat_counter = collections.Counter()
  pdf_parse_time_counter = collections.Counter()

  with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
    for stat_count, time_count, metadata in tqdm.tqdm(
      p.imap(explore_metadata, metadata_file_paths), total=len(metadata_file_paths), 
      desc="Gather Metadata Statistics"):
      
      metadata_stat_counter += stat_count
      metadata_time_counter += time_count
      reference_papers_metadata_complete += metadata

    for stat_count, time_count, papers in tqdm.tqdm(p.imap(explore_pdf_parses, pdf_parse_file_paths),
                                          total=len(pdf_parse_file_paths), desc="Gather PDF Parse Statistics"):
      pdf_parse_stat_counter += stat_count
      pdf_parse_time_counter += time_count
      reference_papers_pdf_parses_complete += papers


  logger.info("Store results.")
  results = {
      "metadata": {key: metadata_stat_counter[key] for key in sorted(metadata_stat_counter.keys())},  # sort by key
      "pdf_parse": {key: pdf_parse_stat_counter[key] for key in sorted(pdf_parse_stat_counter.keys())},  # sort by key
      "metadata_timings": dict(metadata_time_counter.items()),
      "pdf_parse_timings": dict(pdf_parse_time_counter.items())
  }

  with open(STATS_RESULTS_PATH, "w", encoding="utf-8") as file:
      json.dump(results, file, indent=2)

  with open(METADATA_RESULTS_PATH, "w", encoding="utf-8") as file:
      json.dump(reference_papers_metadata_complete, file, indent=2)

  with open(PDF_PARSE_RESULTS_PATH, "w", encoding="utf-8") as file:
      json.dump(reference_papers_pdf_parses_complete, file, indent=2)

  logger.info("All done!")

