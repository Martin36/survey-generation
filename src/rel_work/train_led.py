import torch
import evaluate
import logging

from transformers import LEDTokenizer, LEDForConditionalGeneration, Seq2SeqTrainer
from transformers import TrainingArguments
from datasets import load_dataset

from custom_datasets.rel_work import RelatedWorksDataset

logging.basicConfig(level=logging.INFO)


# ['absl', 'rouge_score']
model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
rouge = evaluate.load('rouge')

# Slightly modified from: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
def compute_metrics(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions

  # all unnecessary tokens are removed
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.eos_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

  return {
      "rouge2_precision": round(rouge_output.precision, 4),
      "rouge2_recall": round(rouge_output.recall, 4),
      "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
  }

ENCODER_LENGTH = 4096
DECODER_LENGTH = 4096 # TODO: What should this be?
BATCH_SIZE = 16


# Slightly modified from: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
  inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=ENCODER_LENGTH)
  outputs = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=DECODER_LENGTH)

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


# train_dataset = RelatedWorksDataset("train")
# val_dataset = RelatedWorksDataset("val")
train_dataset = load_dataset("json", data_files="data/related_work/benchmark/aburaed_et_at/train.jsonl")
val_dataset = load_dataset("json", data_files="data/related_work/benchmark/aburaed_et_at/val.jsonl")


# make train dataset ready
train_dataset = train_dataset.map(
  map_to_encoder_decoder_inputs, batched=True, batch_size=BATCH_SIZE, remove_columns=["target", "input_docs", "input"],
)
train_dataset.set_format(
  type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# # same for validation dataset
val_dataset = val_dataset.map(
  map_to_encoder_decoder_inputs, batched=True, batch_size=BATCH_SIZE, remove_columns=["target", "input_docs", "input"],
)
val_dataset.set_format(
  type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

training_args = TrainingArguments(output_dir="models/rel_work")

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset["train"],
  eval_dataset=val_dataset["train"],
  tokenizer=tokenizer,
)

trainer.train()
