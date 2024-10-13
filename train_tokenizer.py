import warnings

import hydra
import tokenizers
from hydra.utils import instantiate

from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="tokenizer_train")
def main(config):
    dataset_iterator = instantiate(config.datasets)

    tokenizer = instantiate(config.tokenizer)

    save_dir = ROOT_PATH / config.save_path
    save_dir.mkdir(exist_ok=True, parents=True)

    tokenizer.train_from_iterator(
        iterator=dataset_iterator, vocab_size=config.vocab_size, special_tokens=[""]
    )
    tokenizer.save_model(str(save_dir))


if __name__ == "__main__":
    main()
