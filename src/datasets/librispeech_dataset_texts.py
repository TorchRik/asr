import re

from src.datasets.librispeech_dataset import LibrispeechDataset


class LibrispeechTextDatasetIterator(LibrispeechDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_index = 0

    def __next__(self):
        if self._current_index < len(self):
            text = self._index[self._current_index]["text"]
            text = text.lower()
            text = re.sub(r"[^a-z ]", "", text)
            self._current_index += 1
            return text
        else:
            raise StopIteration

    def __iter__(self):
        return self
