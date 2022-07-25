from dataclasses import dataclass, field
import torch, evaluate, logging, os, random

from transformers import LEDTokenizerFast, LEDForConditionalGeneration, HfArgumentParser
from datasets import load_dataset
from utils_package.logger import get_logger
from utils_package.util_funcs import store_json

@dataclass
class Args():
  model: str = field(
    default=None,
    metadata={"help": "Path to pretrained model or shortcut name"}
  )

def get_args():
  parser = HfArgumentParser([Args])
  args = parser.parse_args()

  if not args.model:
    raise ValueError("Model path is required.")

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

if __name__ == "__main__":
  logger_2 = get_logger()

  logging.basicConfig(level=logging.INFO)
  
  torch.manual_seed(15)

  logger_2.info(f"Cuda available: {torch.cuda.is_available()}")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logger_2.info(f"Using device: {device}")

  args = get_args()

  model = LEDForConditionalGeneration.from_pretrained(args.model).to(device)
  tokenizer = LEDTokenizerFast.from_pretrained(args.model)
  rouge = evaluate.load('rouge')

  # HYPERPARAMETERS

  ENCODER_LENGTH = int(4096 / 1)
  DECODER_LENGTH = int(4096 / 4)
  BATCH_SIZE = 8
  DEBUGGING = True

  # map data correctly
  def generate_summary(batch):
    inputs = tokenizer(
      batch["input"], 
      padding="max_length", 
      truncation=True, 
      max_length=ENCODER_LENGTH, 
      return_tensors="pt"
    )
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(
      input_ids, 
      attention_mask=attention_mask,
      do_sample=True,
      top_k=50,
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if DEBUGGING:
      output_sample = random.sample(output_str, 3 if len(output_str) > 3 else len(output_str))
      for idx, output in enumerate(output_sample):
        logger_2.info(f"Random output sample {idx+1}: {output}")

    batch["pred"] = output_str

    return batch

  # Taken from: https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=kPnNi_tWaklV
  model.config.num_beams = 2
  model.config.max_length = DECODER_LENGTH
  # model.config.min_length = 100
  model.config.length_penalty = 2.0
  model.config.early_stopping = True
  model.config.no_repeat_ngram_size = 3
  
  # TODO: How to set seed for predictions?

  BASE_DATA_PATH = "data/related_work/benchmark/"
  data_files = [
      BASE_DATA_PATH+"aburaed_et_at/test.jsonl",
      BASE_DATA_PATH+"chen_et_al/delve/test.jsonl",
      BASE_DATA_PATH+"chen_et_al/s2orc/test.jsonl",
      BASE_DATA_PATH+"lu_et_al/test.jsonl",
      BASE_DATA_PATH+"xing_et_al/explicit/test.jsonl",
      BASE_DATA_PATH+"xing_et_al/hp/test.jsonl",
      BASE_DATA_PATH+"xing_et_al/hr/test.jsonl"
    ]

  ds_names = [
    "aburaed_et_at",
    "chen_et_al_delve",
    "chen_et_al_s2orc",
    "lu_et_al",
    "xing_et_al_explicit",
    "xing_et_al_hp",
    "xing_et_al_hr"
  ]

  if DEBUGGING:
    data_files = [BASE_DATA_PATH+"mini_dataset.jsonl"]
    ds_names = ["mini_dataset"]
  
  for file, ds_name in zip(data_files, ds_names):
    model_name = model.name_or_path
    model_name = model_name.replace("/", "_") # TODO: Find a better way to create the names of the model

    output_file = BASE_DATA_PATH+f"predictions/{model_name}/{ds_name}.json"
    if os.path.exists(output_file):
      logger_2.info(f"'{output_file}' already exists. Skipping dataset: '{ds_name}'")
      continue

    logger_2.info(f"Running model: {model_name}")
    logger_2.info(f"On dataset: {ds_name}")

    dataset = load_dataset("json", data_files=file)["train"]

    results = dataset.map(
      generate_summary, 
      batched=True, 
      batch_size=BATCH_SIZE, 
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
        "encoder_length": ENCODER_LENGTH,
        "decoder_length": DECODER_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_beams": model.config.num_beams,
        "early_stopping": model.config.early_stopping,
        "length_penalty": model.config.length_penalty,
        "no_repeat_ngram_size": model.config.no_repeat_ngram_size,
      },
      "rouge": map_rouge_output_to_json(rouge_output)
    }
    store_json(output, output_file)
    logger_2.info(f"Output stored in: {output_file}")

    print()
