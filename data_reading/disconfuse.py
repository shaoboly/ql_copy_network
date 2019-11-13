import json
import os

from data_reading.summarization_read import SummarizeProcessor


class InputExample(object):
    def __init__(self, guid, article, summary):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = summary


class DisConfuProcessor(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'test_balanced.tsv'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'test_balanced.tsv'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'train_balanced.tsv'))
    @staticmethod
    def abstract2sents(abstract: str):
        """Splits abstract text from datafile into list of sentences.

        Args:
          abstract: string containing <s> and </s> tags for starts and ends of sentences

        Returns:
          sents: List of sentence strings (no tags)"""
        return [abstract.strip()]

    @staticmethod
    def _create_examples(file_path):
        print(file_path)
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            line = line.strip().split('\t')
            coherent_first_sentence,coherent_second_sentence,incoherent_first_sentence,incoherent_second_sentence = line[:4]

            src = " ".join([incoherent_first_sentence,incoherent_second_sentence])
            tgt = " ".join([coherent_first_sentence,coherent_second_sentence])

            examples.append(InputExample(guid="0", article=src, summary=tgt))

        return examples

