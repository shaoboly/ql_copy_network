import json
import os

from data_reading.s2l_read import S2LProcessor


class InputExample(object):
    def __init__(self, guid, article, summary):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = summary


class MSProcessor(S2LProcessor):

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'MSParS.Test.seq2seq.json'))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'MSParS.Dev.seq2seq.json'))

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'MSParS.Train.seq2seq.json'))

    @staticmethod
    def abstract2sents(abstract: str):
        """Splits abstract text from datafile into list of sentences.

        Args:
          abstract: string containing <s> and </s> tags for starts and ends of sentences

        Returns:
          sents: List of sentence strings (no tags)"""
        delim = ';'
        return [abstract.strip()]

    @staticmethod
    def _create_examples(file_path):
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            data = json.loads(line.strip())
            examples.append(InputExample(guid=data['guid'], article=data['article'], summary=data['abstract']))
        return examples
