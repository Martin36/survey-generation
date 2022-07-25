import random, torch, evaluate, logging, os

from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, HfArgumentParser
from datasets import load_dataset, load_from_disk
from utils_package.logger import get_logger
from pynvml import *

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
    metadata={"help": "Path to pretrained model or shortcut name"}
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
# - BART TODO

# model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
# tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-large-16384-arxiv")
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)
rouge = evaluate.load('rouge')

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()




# HYPERPARAMETERS

ENCODER_LENGTH = args.enc_len
DECODER_LENGTH = args.dec_len
BATCH_SIZE = 8
EPOCHS = 3
SAVE_EVAL_STEPS = 300
OUTPUT_DIR = args.output_dir

DEBUGGING = True
USE_CAHCED_DATASET = False

# Slightly modified from: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
def compute_metrics(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions
  inputs = pred.inputs

  # all unnecessary tokens are removed
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.eos_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
  if inputs:
    input_str = tokenizer.batch_decode(inputs, skip_special_tokens=True)
  else:
    input_str = [''] * len(pred_str)

  merged_list = list(zip(input_str, label_str, pred_str)) # TODO: What to call this?
  rand_sample = random.sample(merged_list, 2 if len(merged_list) > 2 else len(merged_list))

  if DEBUGGING:
    logger.info(f"Lenght input_str: '{len(input_str)}'")
    logger.info(f"Lenght label_str: '{len(label_str)}'")
    logger.info(f"Lenght pred_str: '{len(pred_str)}'")
    logger.info(f"Lenght merged list: '{len(merged_list)}'")

  for idx, (input, label, pred) in enumerate(rand_sample):
    logger.info(f"Random evaluation sample nr {idx+1}")
    logger.info(f"Input: {input}")
    logger.info(f"Label: {label}")
    logger.info(f"Pred: {pred}")
    logger.info("-" * 80)

  rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

  return {
      "rouge2_precision": round(rouge_output.precision, 4),
      "rouge2_recall": round(rouge_output.recall, 4),
      "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
  }


# Slightly modified from: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
  inputs = tokenizer(
    batch["input"], 
    padding="max_length", 
    truncation=True, 
    max_length=ENCODER_LENGTH
  )
  outputs = tokenizer(
    batch["target"], 
    padding="max_length", 
    truncation=True, 
    max_length=DECODER_LENGTH
  )

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["labels"] = outputs.input_ids.copy()
  batch["decoder_attention_mask"] = outputs.attention_mask

  # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
  batch["labels"] = [
      [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
  ]

  assert all([len(x) == ENCODER_LENGTH for x in inputs.input_ids])
  assert all([len(x) == DECODER_LENGTH for x in outputs.input_ids])

  return batch


BASE_DATA_PATH = "data/related_work/benchmark/"
data_files = {
  "train": [
    # BASE_DATA_PATH+"aburaed_et_at/train.jsonl",
    # BASE_DATA_PATH+"chen_et_al/delve/train.jsonl",
    # BASE_DATA_PATH+"chen_et_al/s2orc/train.jsonl",
    BASE_DATA_PATH+"lu_et_al/train.jsonl",
    # BASE_DATA_PATH+"xing_et_al/explicit/train.jsonl",
    # BASE_DATA_PATH+"xing_et_al/hp/train.jsonl",
    # BASE_DATA_PATH+"xing_et_al/hr/train.jsonl"
  ],
  "validation": [
    # BASE_DATA_PATH+"aburaed_et_at/val.jsonl",
    # BASE_DATA_PATH+"chen_et_al/delve/val.jsonl",
    # BASE_DATA_PATH+"chen_et_al/s2orc/val.jsonl",
    BASE_DATA_PATH+"lu_et_al/val.jsonl",
    # BASE_DATA_PATH+"xing_et_al/explicit/val.jsonl",
    # BASE_DATA_PATH+"xing_et_al/hp/val.jsonl",
    # BASE_DATA_PATH+"xing_et_al/hr/val.jsonl"
  ]
}

if DEBUGGING:
  data_files = {
    "train": [BASE_DATA_PATH+"mini_dataset_train.jsonl"],
    "validation": [BASE_DATA_PATH+"mini_dataset_val.jsonl"]
  }
  dataset = load_dataset("json", data_files=data_files)
  dataset = dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=BATCH_SIZE, remove_columns=["target"],#, "input"],
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
    dataset = load_dataset("json", data_files=data_files)

    logger.info("Mapping data to correct format...")
    dataset = dataset.map(
      map_to_encoder_decoder_inputs, batched=True, batch_size=BATCH_SIZE, remove_columns=["target", "input"],
    )
    dataset.set_format(
      type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
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
model.config.max_length = DECODER_LENGTH
# model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.do_sample = do_sample

# TODO: Is this necessary?
# tokenizer.additional_special_tokens = ["<BOS>", "<EOS>"]

if DEBUGGING:
  training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE//2,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=True,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=1,
    save_steps=5,
    save_total_limit=5,
    num_train_epochs=EPOCHS,
    seed=15,  # For reproducability
  )
else:
  training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE//2,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=True,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=SAVE_EVAL_STEPS,
    save_steps=SAVE_EVAL_STEPS,
    save_total_limit=5,
    num_train_epochs=EPOCHS,
    seed=15,  # For reproducability
  )

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=dataset["train"],
  eval_dataset=dataset["validation"],
  compute_metrics=compute_metrics,  # TODO: Try this to see if reduced evaluation time https://discuss.huggingface.co/t/evaluation-became-slower-and-slower-during-trainer-train/8682
  tokenizer=tokenizer,
)

logger.info("Starting training...")
result = trainer.train()
print_summary(result)
