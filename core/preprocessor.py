import pymorphy3
import re
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

class TextPreprocessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        # Natasha для более точного POS
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.tagger = NewsMorphTagger(self.emb)

    def process(self, text: str):
        # Возвращает список предложений с токенами:
        # [{"token": "замену", "lemma": "замена", "pos": "NOUN", "idx": 0}, ...]
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.tagger)
        result = []
        for sent in doc.sents:
            tokens = []
            for token in sent.tokens:
                token.lemmatize(self.morph_vocab)
                tokens.append({
                    "token": token.text,
                    "lemma": token.lemma,
                    "pos": token.pos,
                    "start": token.start,
                    "stop": token.stop
                })
            result.append(tokens)
        return result