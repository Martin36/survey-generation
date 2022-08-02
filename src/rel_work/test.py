from dataclasses import dataclass, field
import torch, evaluate, logging, os, random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser
from datasets import load_dataset
from utils_package.logger import get_logger
from utils_package.util_funcs import store_json

from utils import concat_input

@dataclass
class Args():
  model: str = field(
    default=None,
    metadata={"help": "Path to pretrained model or shortcut name"}
  )
  decoding_strategy: str = field(
    default="beam_search",
    metadata={"help": "Strategy to use for decoding, e.g. beam_search, top_k or top_p"}
  )
  top_k: int = field(
    default=50,
    metadata={"help": "K for top-k decoding"}
  )
  top_p: float = field(
    default=1.0,
    metadata={"help": "P for top-p decoding"}
  )
  use_sep_token: bool = field(
    default=False,
    metadata={"help": "If True the input documents will be separated by the sep token in the input. Defaults to False"}
  )
  datasets: str = field(
    default="all",
    metadata={"help": "Comma separated list of datasets to use e.g. 'lu_et_al,xing_et_al'. Default uses all datasets"}
  )
  batch_size: int = field(
    default=8,
    metadata={"help": "Batch size for running predictions"}
  )


def get_args():
  parser = HfArgumentParser([Args])
  args = parser.parse_args()

  if not args.model:
    raise ValueError("Model path is required.")
  if args.decoding_strategy == "top_p" and args.top_p == 1.0:
    raise ValueError("With decoding_strategy=top_p you must provide a value for top_p.")

  return args

def map_rouge_output_to_json(rouge_output):
  result = dict()
  for key, value in rouge_output.items():
    result[key] = dict()
    result[key]["low"] = {
      "precision": value.low.precision,
      "recall": value.low.recall,
      "fmeasure": value.low.fmeasure
    }
    result[key]["mid"] = {
      "precision": value.mid.precision,
      "recall": value.mid.recall,
      "fmeasure": value.mid.fmeasure
    }
    result[key]["high"] = {
      "precision": value.high.precision,
      "recall": value.high.recall,
      "fmeasure": value.high.fmeasure
    }
  return result

BASE_DATA_PATH = "data/related_work/benchmark/"
DEBUGGING = False

def get_data_files(datasets: str):
  if DEBUGGING:
    data_files = [BASE_DATA_PATH+"mini_dataset.jsonl"]
    ds_names = ["mini_dataset"]
    return data_files, ds_names

  data_files = list()
  ds_names = list()
  valid_datasets = ["aburaed_et_al", "lu_et_al", "xing_et_al", "chen_et_al", "all"]
  for dataset in datasets.split(","):
    if dataset == "aburaed_et_at" or dataset == "all":
      data_files.append(BASE_DATA_PATH+"aburaed_et_at/test.jsonl")
      ds_names.append("aburaed_et_at")
    if dataset == "chen_et_al" or dataset == "all":
      data_files.append(BASE_DATA_PATH+"chen_et_al/delve/test.jsonl")
      data_files.append(BASE_DATA_PATH+"chen_et_al/s2orc/test.jsonl")
      ds_names.append("chen_et_al_delve")
      ds_names.append("chen_et_al_s2orc")
    if dataset == "lu_et_al" or dataset == "all":
      data_files.append(BASE_DATA_PATH+"lu_et_al/test.jsonl")
      ds_names.append("lu_et_al")
    if dataset == "xing_et_al" or dataset == "all":
      data_files.append(BASE_DATA_PATH+"xing_et_al/explicit/test.jsonl")
      data_files.append(BASE_DATA_PATH+"xing_et_al/hp/test.jsonl")
      data_files.append(BASE_DATA_PATH+"xing_et_al/hr/test.jsonl")
      ds_names.append("xing_et_al_explicit")
      ds_names.append("xing_et_al_hp")
      ds_names.append("xing_et_al_hr")
    if dataset not in valid_datasets:
      raise ValueError(f"Unknown dataset: {dataset}")

  return data_files, ds_names

if __name__ == "__main__":
  logger = get_logger()
  
  torch.manual_seed(15)

  logger.info(f"Cuda available: {torch.cuda.is_available()}")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger.info(f"Using device: {device}")

  args = get_args()

  model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  rouge = evaluate.load('rouge')

  if args.decoding_strategy == "top_p":
    do_sample = True
  elif args.decoding_strategy == "top_k":
    do_sample = True
  else:
    do_sample = False

  # map data correctly
  def generate_summary(batch):
    if hasattr(model.config, "max_encoder_position_embeddings"):
      max_length = model.config.max_encoder_position_embeddings
    if hasattr(model.config, "max_position_embeddings"):
      max_length = model.config.max_position_embeddings
    else:
      # TODO: what to set max_length to if none of the above attributes exist in the config?
      max_length = 512

    # TODO: Add option to use sep token here?
    batch["input"] = concat_input(batch["input"])
    
    inputs = tokenizer(
      batch["input"], 
      padding="max_length", 
      truncation=True, 
      max_length=max_length, 
      return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    outputs = model.generate(
      input_ids, 
      attention_mask=attention_mask,
      do_sample=do_sample,
      top_k=args.top_k,
      top_p=args.top_p,
    )

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if DEBUGGING:
      output_sample = random.sample(output_str, 3 if len(output_str) > 3 else len(output_str))
      for idx, output in enumerate(output_sample):
        logger.info(f"Random output sample {idx+1}: {output}")

    batch["pred"] = output_str

    return batch

  # Taken from: https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=kPnNi_tWaklV
  model.config.num_beams = 2
  # model.config.max_length = DECODER_LENGTH
  # model.config.min_length = 100
  model.config.length_penalty = 2.0
  model.config.early_stopping = True
  model.config.no_repeat_ngram_size = 3
  
  # TODO: How to set seed for predictions?

  data_files, ds_names = get_data_files(args.datasets)
  
  for file, ds_name in zip(data_files, ds_names):
    model_name = model.name_or_path
    model_name = model_name.replace("/", "_") # TODO: Find a better way to create the names of the model

    output_file = BASE_DATA_PATH+f"predictions/{model_name}/{ds_name}.json"
    if os.path.exists(output_file):
      logger.info(f"'{output_file}' already exists. Skipping dataset: '{ds_name}'")
      continue

    logger.info(f"Running model: {model_name}")
    logger.info(f"On dataset: {ds_name}")

    dataset = load_dataset("json", data_files=file)["train"]

    results = dataset.map(
      generate_summary, 
      batched=True, 
      batch_size=args.batch_size, 
    )

    pred_str = results["pred"]
    label_str = results["target"]

    predictions = [{
      "input": d["input"], 
      "target": d["target"], 
      "prediction": d["pred"],
    } for d in results]
    
    rouge_output = rouge.compute(
      predictions=pred_str, 
      references=label_str, 
    )

    output = {
      "dataset_name": ds_name,
      "model_name": model_name,
      "predictions": predictions,
      "hyperparameters": {
        "encoder_length": model.config.max_encoder_position_embeddings,
        "decoder_length": model.config.max_decoder_position_embeddings,
        "batch_size": args.batch_size,
        "num_beams": model.config.num_beams,
        "early_stopping": model.config.early_stopping,
        "length_penalty": model.config.length_penalty,
        "no_repeat_ngram_size": model.config.no_repeat_ngram_size,
        "decoding_strategy": args.decoding_strategy,
        "top_k": args.top_k,
        "top_p": args.top_p,
      },
      "rouge": map_rouge_output_to_json(rouge_output)
    }
    store_json(output, output_file)
    logger.info(f"Output stored in: {output_file}")

    print()
