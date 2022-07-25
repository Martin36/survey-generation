
from glob import glob
from utils_package.util_funcs import load_json
import pandas as pd

INPUT_FOLDER = "outputs/predictions/allenai_led-large-16384-arxiv"
OUTPUT_FILE = "outputs/predictions/allenai_led-large-16384-arxiv.csv"


if __name__ == "__main__":
  input_files = glob(INPUT_FOLDER + "/*.json")
  rows_list = []

  for file in input_files:
    input_data = load_json(file)

    dataset_name = input_data["dataset_name"]
    model_name = input_data["model_name"]
    encoder_length = input_data["hyperparameters"]["encoder_length"]
    decoder_length = input_data["hyperparameters"]["decoder_length"]
    batch_size = input_data["hyperparameters"]["batch_size"]
    num_beams = input_data["hyperparameters"]["num_beams"]
    early_stopping = input_data["hyperparameters"]["early_stopping"]
    length_penalty = input_data["hyperparameters"]["length_penalty"]
    no_repeat_ngram_size = input_data["hyperparameters"]["no_repeat_ngram_size"]

    for key in input_data["rouge"]:
      scores = input_data["rouge"][key][1]  # second element in the array corresponds to the "mid" rouge score
      precision = scores[0]
      recall = scores[1]
      f1 = scores[2]

      rows_list.append({
        "Model": model_name,
        "Dataset": dataset_name,
        "ROUGE type": key,
        "l,m or h": "mid",
        "P (R)": precision,
        "R (R)": recall,
        "F1 (R)": f1,
        "Enc len": encoder_length,
        "Dec len": decoder_length,
        "Dec type": "beam",
        "Beams/top k": num_beams,
        "Len pen": length_penalty,
        "Early stopping": early_stopping,
        "NR ng": no_repeat_ngram_size
      })

  output_df = pd.DataFrame(rows_list)
  output_df.to_csv(OUTPUT_FILE, index=False)
  print("Saved to:", OUTPUT_FILE)