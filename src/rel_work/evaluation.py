import evaluate

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from utils_package.logger import get_logger
from utils_package.util_funcs import load_json

logger = get_logger()

# Metrics to use:
# - ROUGE
# - BERT-Score

@dataclass
class Args():
  results_file: str = field(
    default=None,
    metadata={"help": "Path to the results file. This file should consist of a list with objects with keys 'prediction' and 'label'"}
  )

def get_args():
  parser = HfArgumentParser([Args])
  args = parser.parse_args()

  if not args.results_file:
    raise ValueError("Please provide a results file")

  return args


def compute_rouge(pred_and_labels):
  """Calculated rouge scores for the predictions. 
    Uses the "mid" and "fmeasure" scores.

  Args:
      pred_and_labels (list): A list containing objects with keys 'prediction' and 'label'

  Returns:
      dict: A dictionary with keys 'rouge1', 'rouge2', 'rougeL' and values of the corresponding rouge scores
  """
  rouge = evaluate.load('rouge')
  
  pred_strs = [d["prediction"] for d in pred_and_labels]
  label_strs = [d["label"] for d in pred_and_labels]

  rouge_output = rouge.compute(
    predictions=pred_strs, 
    references=label_strs)

  result = {
    "rouge1": rouge_output["rouge1"].mid.fmeasure,
    "rouge2": rouge_output["rouge2"].mid.fmeasure,
    "rougeL": rouge_output["rougeL"].mid.fmeasure,
  }

  return result

def compute_bertscore(pred_and_labels):
  """Calculated BERT scores for the predictions. 
    Uses the "distilbert-base-uncased" for calculating the scores.

  Args:
      pred_and_labels (list): A list containing objects with keys 'prediction' and 'label'

  Returns:
      dict: A dictionary with keys 'precision', 'recall', 'f1' and values of the corresponding BERT scores
  """
  bertscore = evaluate.load('bertscore')
  
  pred_strs = [d["prediction"] for d in pred_and_labels]
  label_strs = [d["label"] for d in pred_and_labels]

  bertscore_output = bertscore.compute(
    predictions=pred_strs, 
    references=label_strs,
    model_type="distilbert-base-uncased")

  return bertscore_output


# TODO: Extend this to take a model as input?

if __name__ == "__main__":
  args = get_args()

  pred_and_labels = load_json(args.results_file)

  rouge_score = compute_rouge(pred_and_labels)
  bertscore = compute_bertscore(pred_and_labels)

  # TODO: What to do with these scores?