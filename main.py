import argparse
from typing import List, Tuple

from flair.data import Sentence
from flair.models import SequenceTagger


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sentence", type=str, default=None)

    args = parser.parse_args()

    return args


def init():
    tagger = SequenceTagger.load("ner")

    return tagger


def inference(sentence: str, tagger: SequenceTagger) -> List[Tuple[str, float]]:
    tagger.predict(sentence)

    entities = []

    for entity in sentence.get_spans("ner"):
        entities.append((entity.text, entity.score))

    return entities


if __name__ == "__main__":
    args = get_args()

    tagger = init()

    sentence = Sentence(args.sentence)

    entities = inference(sentence=sentence, tagger=tagger)
