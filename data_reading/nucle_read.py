import json
import os

from data_reading.summarization_read import SummarizeProcessor


class InputExample(object):
    def __init__(self, guid, article, summary):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = summary


class NULEProcessor(SummarizeProcessor):

    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'test.txt'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'test.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'train.txt'))

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
            line = line.lower().strip().split('\t')
            if len(line)!=2:
                print(line)
                continue
            src,tgt = line[0],line[1]
            if len(src.split())==0 or len(tgt.split())==0:
                print(line)
                continue

            examples.append(InputExample(guid="0", article=src, summary=tgt))

        return examples

