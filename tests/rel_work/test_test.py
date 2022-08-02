import unittest, evaluate

from rel_work.test import map_rouge_output_to_json

class TestTest(unittest.TestCase):

  def test_map_rouge_output_to_json(self):
    rouge = evaluate.load('rouge')
    pred_strs = ["test test test", "test test test"]
    label_strs = ["test test test", "test test test"]
    rouge_output = rouge.compute(
      predictions=pred_strs, 
      references=label_strs, 
    )
    output = map_rouge_output_to_json(rouge_output)
    target = {
      "rouge1": {
        "low": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "mid": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "high": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
      },
      "rouge2": {
        "low": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "mid": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "high": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
      },
      "rougeL": {
        "low": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "mid": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "high": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
      },
      "rougeLsum": {
        "low": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "mid": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
        "high": {
          "precision": 1.0,
          "recall": 1.0,
          "fmeasure": 1.0,
        },
      }
    }
    self.assertEqual(output, target, "Result should be correct")
    

if __name__ == '__main__':
  unittest.main()