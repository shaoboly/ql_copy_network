import json
import os

from data_reading.summarization_read import SummarizeProcessor


class CNNDailyProcessor(SummarizeProcessor):
    def get_test_examples(self, data_dir,fname=None):
        if fname != None:
            print(fname)
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'test.txt'))

    def get_dev_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_train_examples(self, data_dir,fname=None):
        if fname != None:
            return self._create_examples(os.path.join(data_dir, fname))
        return self._create_examples(os.path.join(data_dir, 'train.txt'))

    @staticmethod
    def abstract2sents(abstract: str):
        """Splits abstract text from datafile into list of sentences.

        Args:
          abstract: string containing <s> and </s> tags for starts and ends of sentences

        Returns:
          sents: List of sentence strings (no tags)"""
        SENTENCE_START = '<s>'
        SENTENCE_END = '</s>'
        cur = 0
        sents = []
        while True:
            try:
                start_p = abstract.index(SENTENCE_START, cur)
                end_p = abstract.index(SENTENCE_END, start_p + 1)
                cur = end_p + len(SENTENCE_END)
                sents.append(abstract[start_p + len(SENTENCE_START):end_p])
            except ValueError:  # no more sntences
                return sents

    @staticmethod
    def _create_examples(file_path):
        examples = []
        for line in open(file_path, 'r', encoding='utf-8', errors='ignore'):
            data = json.loads(line.strip())
            examples.append(InputExample(guid=data['guid'], article=data['article'], summary=data['abstract']))
        return examples


class InputExample(object):
    def __init__(self, guid, article, summary):
        self.guid = guid
        self.article = article
        self.summary = summary
        self.true_summary = ' '.join([sent.strip() for sent in CNNDailyProcessor.abstract2sents(summary)])
