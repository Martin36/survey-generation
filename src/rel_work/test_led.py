import torch
import evaluate
import logging

from transformers import LEDTokenizerFast, LEDForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from utils_package.logger import get_logger
from pynvml import *

logger = get_logger()

logging.basicConfig(level=logging.INFO)

logger.info(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
# tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-large-16384-arxiv")
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384").to(device)
tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")

# HYPERPARAMETERS

ENCODER_LENGTH = int(4096 / 1)
DECODER_LENGTH = int(4096 / 4)
BATCH_SIZE = 8


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

  outputs = model.generate(input_ids, attention_mask=attention_mask)

  # all special tokens including will be removed
  output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

  batch["pred"] = output_str

  return batch



# Taken from: https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=kPnNi_tWaklV
model.config.num_beams = 2
model.config.max_length = DECODER_LENGTH
# model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

BASE_DATA_PATH = "data/related_work/benchmark/"
data_files = {
  "test": [
    BASE_DATA_PATH+"aburaed_et_at/test.jsonl",
    BASE_DATA_PATH+"chen_et_al/delve/test.jsonl",
    BASE_DATA_PATH+"chen_et_al/s2orc/test.jsonl",
    BASE_DATA_PATH+"lu_et_al/test.jsonl",
    BASE_DATA_PATH+"xing_et_al/explicit/test.jsonl",
    BASE_DATA_PATH+"xing_et_al/hp/test.jsonl",
    BASE_DATA_PATH+"xing_et_al/hr/test.jsonl"
  ]
}

# JUST FOR TESTING
# data_files = {
#   "test": [BASE_DATA_PATH+"mini_dataset.jsonl"]
# } 

dataset = load_dataset("json", data_files=data_files)

results = dataset.map(
  generate_summary, 
  batched=True, 
  batch_size=BATCH_SIZE, 
  remove_columns=["input"]
)

pred_str = results["test"]["pred"]
label_str = results["test"]["target"]

rouge = evaluate.load('rouge')
rouge_output = rouge.compute(
  predictions=pred_str, 
  references=label_str, 
  rouge_types=["rouge2"]
)["rouge2"].mid

print(rouge_output)