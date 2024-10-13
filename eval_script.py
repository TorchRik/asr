import os
import warnings

import hydra

from src.metrics.utils import calc_cer, calc_wer

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="src_script")
def main(config):
    total_cer = 0
    total_wer = 0
    total_count = 0
    for file_name in os.listdir(config.predictions_path):
        with open(os.path.join(config.predictions_path, file_name), "r") as f:
            prediction_text = f.read()
        with open(os.path.join(config.ground_truth_path, file_name), "r") as f:
            ground_truth_text = f.read()
        total_cer += calc_cer(
            target_text=ground_truth_text, predicted_text=prediction_text
        )
        total_wer += calc_wer(
            target_text=ground_truth_text, predicted_text=prediction_text
        )
        total_count += 1
    cer = total_cer / total_count
    wer = total_wer / total_count
    print(f"CER: {cer}")
    print(f"CER: {wer}")


if __name__ == "__main__":
    main()
