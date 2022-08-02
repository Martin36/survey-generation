import random, logging, os
from typing import List
import evaluate
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser
from datasets import load_dataset, load_from_disk, load_metric
from utils_package.logger import get_logger
from pynvml import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from utils import concat_input

logger = get_logger()

# logging.basicConfig(level=logging.INFO)

torch.manual_seed(15)
random.seed(15)

logger.info(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

@dataclass
class Args():
  model: str = field(
    default=None,
    metadata={"help": "Path to pretrained model or shortcut name. Needs to be a model compatible with Huggingface 'AutoModelForSeq2SeqLM'"}
  )
  enc_len: int = field(
    default=4096,
    metadata={"help": "Length of the encoder input"}
  )
  dec_len: int = field(
    default=1024,
    metadata={"help": "Length of the decoder output"}
  )
  output_dir: str = field(
    default=None, # e.g. "models/rel_work/led-base-16384-multi-x"
    metadata={"help": "Output directory for checkpoints and models"}
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
  base_data_path: str = field(
    default="data/related_work/benchmark/",
    metadata={"help": "Path to the base data folder. This is the folder where the indiviual dataset folder e.g. 'lu_et_al' is located. Default 'data/related_work/benchmark/'"}
  )
  debugging: bool = field(
    default=False,
    metadata={"help": "Set this to true if you want to debug the training. It sets the eval steps to a lower number and uses a mini version of the full dataset"}
  )
  save_steps: int = field(
    default=300,
    metadata={"help": "Save model checkpoint every save_steps"}
  )
  eval_steps: int = field(
    default=300,
    metadata={"help": "Perform evaluation every eval_steps"}
  )
  use_cpu: bool = field(
    default=False,
    metadata={"help": "Use CPU instead of GPU, even if GPU is available"}
  )
  use_sep_token: bool = field(
    default=False,
    metadata={"help": "If True the input documents will be separated by the sep token in the input. Defaults to False"}
  )
  datasets: str = field(
    default="all",
    metadata={"help": "Comma separated list of datasets to use e.g. 'lu_et_al,xing_et_al'. Default uses all datasets"}
  )

def get_args():
  parser = HfArgumentParser([Args])
  args = parser.parse_args()

  if not args.model:
    raise ValueError("Model path is required.")
  if not args.output_dir:
    raise ValueError("Output directory is required.")
  if args.decoding_strategy == "top_p" and args.top_p == 1.0:
    raise ValueError("With decoding_strategy=top_p you must provide a value for top_p.")

  return args

args = get_args()

# Models to try:
# - allenai/led-base-16384
# - facebook/bart-base
# - allenai/led-large-16384-arxiv   # This one might be too large/take too long time to train

model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
rouge = evaluate.load('rouge')
# rouge = load_metric('rouge')

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

BATCH_SIZE = 8
EPOCHS = 3

USE_CAHCED_DATASET = False
PUBMED = False

if args.debugging: logger.warning("Debugging mode enabled")
if PUBMED: logger.info("Training on PubMed dataset")

# Slightly modified from: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
def compute_metrics(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions

  # all unnecessary tokens are removed
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.eos_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  merged_list = list(zip(label_str, pred_str)) # TODO: What to call this?
  rand_sample = random.sample(merged_list, 2 if len(merged_list) > 2 else len(merged_list))

  for idx, (label, pred) in enumerate(rand_sample):
    logger.info(f"Random evaluation sample nr {idx+1}")
    logger.info(f"Label: {label}")
    logger.info(f"Pred: {pred}")
    logger.info("-" * 80)

  rouge_output = rouge.compute(predictions=pred_str, 
    references=label_str, rouge_types=["rouge2"]
  )["rouge2"].mid

  return {
      "rouge2_precision": round(rouge_output.precision, 4),
      "rouge2_recall": round(rouge_output.recall, 4),
      "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
  }


# Slightly modified from: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
  if not PUBMED:
    if args.use_sep_token:
      batch["input"] = concat_input(batch["input"], tokenizer.sep_token)
    else:
      batch["input"] = concat_input(batch["input"])

  inputs = tokenizer(
    batch["input"] if not PUBMED else batch["article"], 
    padding="max_length", 
    truncation=True, 
    max_length=args.enc_len
  )
  outputs = tokenizer(
    batch["target"] if not PUBMED else batch["abstract"], 
    padding="max_length", 
    truncation=True, 
    max_length=args.dec_len
  )

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # We have to make sure that the PAD token is ignored
  batch["labels"] = [
      [-100 if token == tokenizer.pad_token_id else token for token in labels]
      for labels in batch["labels"]
  ]

  # TODO: Global attention mask?

  assert all([len(x) == args.enc_len for x in inputs.input_ids])
  assert all([len(x) == args.dec_len for x in outputs.input_ids])

  return batch


def get_data_files():
  data_files = {"train": [], "validation": []}
  valid_datasets = ["aburaed_et_al", "lu_et_al", "xing_et_al", "chen_et_al", "all"]
  for dataset in args.datasets.split(","):
    if dataset == "aburaed_et_at" or dataset == "all":
      data_files["train"].append(args.base_data_path+"aburaed_et_at/train.jsonl")
      data_files["validation"].append(args.base_data_path+"aburaed_et_at/val.jsonl")
    if dataset == "chen_et_al" or dataset == "all":
      data_files["train"].append(args.base_data_path+"chen_et_al/delve/train.jsonl")
      data_files["validation"].append(args.base_data_path+"chen_et_al/delve/val.jsonl")  
      data_files["train"].append(args.base_data_path+"chen_et_al/s2orc/train.jsonl")
      data_files["validation"].append(args.base_data_path+"chen_et_al/s2orc/val.jsonl")
    if dataset == "lu_et_al" or dataset == "all":
      data_files["train"].append(args.base_data_path+"lu_et_al/train.jsonl")
      data_files["validation"].append(args.base_data_path+"lu_et_al/val.jsonl")
    if dataset == "xing_et_al" or dataset == "all":
      data_files["train"].append(args.base_data_path+"xing_et_al/explicit/train.jsonl")
      data_files["validation"].append(args.base_data_path+"xing_et_al/explicit/val.jsonl")
      data_files["train"].append(args.base_data_path+"xing_et_al/hp/train.jsonl")
      data_files["validation"].append(args.base_data_path+"xing_et_al/hp/val.jsonl")
      data_files["train"].append(args.base_data_path+"xing_et_al/hr/train.jsonl")
      data_files["validation"].append(args.base_data_path+"xing_et_al/hr/val.jsonl")
    if dataset not in valid_datasets:
      raise ValueError(f"Unknown dataset: {dataset}")

  return data_files


if PUBMED:
  train_dataset = load_dataset("scientific_papers", "pubmed", split="train")
  #val_dataset = load_dataset("scientific_papers", "pubmed", split="validation")
  val_dataset = load_dataset("scientific_papers", "pubmed", split="validation[:250]")

  train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["article", "abstract", "section_names"]
  )
  val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["article", "abstract", "section_names"],
    load_from_cache_file=False
  )

  train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
  )
  val_dataset.set_format(
      type="torch",
      columns=["input_ids", "attention_mask", "labels"],
  )

  dataset = {
    "train": train_dataset,
    "validation": val_dataset
  }

elif args.debugging:
  data_files = {
    "train": [args.base_data_path+"mini_dataset_train.jsonl"],
    "validation": [args.base_data_path+"mini_dataset_val.jsonl"]
  }
  dataset = load_dataset("json", data_files=data_files)
  dataset = dataset.map(
    map_to_encoder_decoder_inputs, 
    batched=True, 
    batch_size=BATCH_SIZE, 
    remove_columns=["target", "input"],
  )
  dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
  )

else:
  TOKENIZED_DATASET_PATH = "data/related_work/benchmark/dataset_tokenized"
  if os.path.isdir(TOKENIZED_DATASET_PATH) and USE_CAHCED_DATASET:
    logger.info("Tokenized dataset already exists. Loading from disk...")
    dataset = load_from_disk(TOKENIZED_DATASET_PATH)
  else:
    data_files = get_data_files()
    dataset = load_dataset("json", data_files=data_files)

    logger.info("Mapping data to correct format...")
    dataset = dataset.map(
      map_to_encoder_decoder_inputs, 
      batched=True, 
      batch_size=BATCH_SIZE, 
      remove_columns=["target", "input"], 
      load_from_cache_file=False
    )
    dataset.set_format(
      type="torch", columns=["input_ids", "attention_mask", "labels"],
    )

    dataset.save_to_disk(TOKENIZED_DATASET_PATH)

if args.decoding_strategy == "top_p":
  do_sample = True
elif args.decoding_strategy == "top_k":
  do_sample = True
else:
  do_sample = False


# Taken from: https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=kPnNi_tWaklV
model.config.num_beams = 2
model.config.max_length = args.dec_len
# model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.do_sample = do_sample
model.config.max_encoder_position_embeddings = args.enc_len
model.config.max_decoder_position_embeddings = args.dec_len

use_fp16 = True if device == "cuda" and not args.use_cpu else False

if args.debugging:
  training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE//2,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=use_fp16,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=200,
    save_total_limit=5,
    num_train_epochs=EPOCHS,
    seed=15,  # For reproducability
    no_cuda=True if args.use_cpu else False,
  )
else:
  training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE//2,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=use_fp16,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    save_total_limit=5,
    num_train_epochs=EPOCHS,
    seed=15,  # For reproducability
    no_cuda=True if args.use_cpu else False,
  )

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=dataset["train"],
  eval_dataset=dataset["validation"],
  compute_metrics=compute_metrics,
  tokenizer=tokenizer,
)

logger.info("Starting training...")
result = trainer.train()
print_summary(result)
