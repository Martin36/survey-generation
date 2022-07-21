import torch
import evaluate
import logging
import os

from transformers import LEDTokenizerFast, LEDForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, load_from_disk
from utils_package.logger import get_logger
from pynvml import *

logger = get_logger()

logging.basicConfig(level=logging.INFO)

logger.info(f"Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
# tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-large-16384-arxiv")
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
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



# Slightly modified from: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
def compute_metrics(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions

  # logger.info(f"Label ids shape: {labels_ids.shape}")
  # logger.info(f"Pred ids shape: {pred_ids.shape}")

  # all unnecessary tokens are removed
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.eos_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  # logger.info(f"Pred str: {pred_str}")
  # logger.info(f"Label str: {label_str}")

  rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

  return {
      "rouge2_precision": round(rouge_output.precision, 4),
      "rouge2_recall": round(rouge_output.recall, 4),
      "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
  }

# HYPERPARAMETERS

ENCODER_LENGTH = int(4096 / 1)
DECODER_LENGTH = int(4096 / 4)
BATCH_SIZE = 8
EPOCHS = 1
SAVE_EVAL_STEPS = 300
TESTING = False

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
    BASE_DATA_PATH+"aburaed_et_at/train.jsonl",
    BASE_DATA_PATH+"chen_et_al/delve/train.jsonl",
    BASE_DATA_PATH+"chen_et_al/s2orc/train.jsonl",
    BASE_DATA_PATH+"lu_et_al/train.jsonl",
    BASE_DATA_PATH+"xing_et_al/explicit/train.jsonl",
    BASE_DATA_PATH+"xing_et_al/hp/train.jsonl",
    BASE_DATA_PATH+"xing_et_al/hr/train.jsonl"
  ],
  "validation": [
    BASE_DATA_PATH+"aburaed_et_at/val.jsonl",
    BASE_DATA_PATH+"chen_et_al/delve/val.jsonl",
    BASE_DATA_PATH+"chen_et_al/s2orc/val.jsonl",
    BASE_DATA_PATH+"lu_et_al/val.jsonl",
    BASE_DATA_PATH+"xing_et_al/explicit/val.jsonl",
    BASE_DATA_PATH+"xing_et_al/hp/val.jsonl",
    BASE_DATA_PATH+"xing_et_al/hr/val.jsonl"
  ]
}

# JUST FOR TESTING THE PIPELINE
if TESTING:
  data_files = {
    "train": [BASE_DATA_PATH+"mini_dataset.jsonl"],
    "validation": [BASE_DATA_PATH+"mini_dataset.jsonl"]
  }
  dataset = load_dataset("json", data_files=data_files)
  dataset = dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=BATCH_SIZE, remove_columns=["target", "input"],
  )
  dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
  )

else:
  TOKENIZED_DATASET_PATH = "data/related_work/benchmark/dataset_tokenized"
  if os.path.isdir(TOKENIZED_DATASET_PATH):
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

# # same for validation dataset
# val_dataset = val_dataset.map(
#   map_to_encoder_decoder_inputs, batched=True, batch_size=BATCH_SIZE, remove_columns=["target", "input_docs", "input"],
# )
# val_dataset.set_format(
#   type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# Taken from: https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing#scrollTo=kPnNi_tWaklV
model.config.num_beams = 2
model.config.max_length = DECODER_LENGTH
# model.config.min_length = 100
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3

if TESTING:
  training_args = Seq2SeqTrainingArguments(
    output_dir="models/rel_work/led-base-16384",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,  # TODO: Try to double the eval batch size to see if speed improves. However the speed is 5-6 times slower so it would still be a lot more
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=True,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=5,
    save_steps=5,
    save_total_limit=5,
    num_train_epochs=EPOCHS,
  )
else:
  training_args = Seq2SeqTrainingArguments(
    output_dir="models/rel_work/led-base-16384",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
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
  )

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=dataset["train"],
  eval_dataset=dataset["validation"],
  # compute_metrics=compute_metrics,  # TODO: Try this to see if reduced evaluation time https://discuss.huggingface.co/t/evaluation-became-slower-and-slower-during-trainer-train/8682
  tokenizer=tokenizer,
)

logger.info("Starting training...")
result = trainer.train()
print_summary(result)
