"""
Download CNN/DailyMail corpus and get url list here(https://github.com/abisee/cnn-dailymail).
Put url_lists directory in `../data/` directory.
Then run this file before train on CNN/DailyMail using following command, it takes about 20 min.
python preprocess_cnn_dm.py <cnn_stories_dir> <dailymail_stories_dir> <output_dir>
"""
import hashlib
import json
import os
import struct
import sys
import time

from tensorflow.core.example import example_pb2
from tqdm import tqdm

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', ''', '`', ''', dm_single_close_quote, dm_double_close_quote,
              ')']  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = '../data/url_lists/all_train.txt'
all_val_urls = '../data/url_lists/all_val.txt'
all_test_urls = '../data/url_lists/all_test.txt'

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506


class Sample(object):
    def __init__(self, s, data):
        article, abstract = data
        self.guid = s.split('.')[0]
        self.article = article
        self.abstract = abstract

    def __str__(self):
        return json.dumps({
            'guid': self.guid,
            'article': self.article,
            'abstract': self.abstract
        })


def load_stories(stories_dir, total_num):
    """load stories"""
    samples = []
    print('Preparing to load from %s ...' % stories_dir)
    stories = os.listdir(stories_dir)
    for i, s in enumerate(tqdm(stories, total=total_num)):
        samples.append(Sample(s, get_art_abs(os.path.join(stories_dir, s))))
    return samples


def read_text_file(text_file):
    lines = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if '@highlight' in line:
        return line
    if line == '':
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + ' .'


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == '':
            continue  # empty line
        elif line.startswith('@highlight'):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(['%s %s %s' % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract


def write_to_file(url_file, out_file, out_bin_file):
    print('Making bin file for URLs listed in %s...' % url_file)
    url_list = read_text_file(url_file)
    url_hashes = get_url_hashes(url_list)
    num_stories = len(url_hashes)
    sample_keys = sample_dict.keys()

    with open(out_file, 'w', encoding='utf-8') as writer, open(out_bin_file, 'wb') as bin_writer:
        for idx, s in enumerate(url_hashes):
            if idx % 1000 == 0:
                print('Writing story %i of %i; %.2f percent done' % (
                    idx, num_stories, float(idx) * 100.0 / float(num_stories)))

            if s in sample_keys:
                writer.write(str(sample_dict[s]))
                writer.write('\n')
                # Write to tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.insert_word([sample_dict[s].article.encode()])
                tf_example.features.feature['abstract'].bytes_list.value.insert_word([sample_dict[s].abstract.encode()])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                bin_writer.write(struct.pack('q', str_len))
                bin_writer.write(struct.pack('%ds' % str_len, tf_example_str))
            else:
                print('Error: Could not find story guid == %s' % s)
    print('Finished writing file %s\n' % out_file)


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            'stories directory %s contains %i files but should contain %i' % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('USAGE: python preprocess_cnn_dm.py <cnn_stories_dir> <dailymail_stories_dir> <output_dir>')
        sys.exit()
    cnn_stories_dir = sys.argv[1]
    dm_stories_dir = sys.argv[2]
    out_dir = sys.argv[3]

    start = time.time()
    # Check the stories directories contain the correct number of .story files
    check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    check_num_stories(dm_stories_dir, num_expected_dm_stories)

    # Create some new directories
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    cnn_samples = load_stories(cnn_stories_dir, num_expected_cnn_stories)
    dm_samples = load_stories(dm_stories_dir, num_expected_dm_stories)

    all_samples = cnn_samples + dm_samples
    sample_dict = {}
    for sample in all_samples:
        sample_dict[sample.guid] = sample

    # Split and write
    write_to_file(all_test_urls, os.path.join(out_dir, 'test.txt'), os.path.join(out_dir, 'test.bin'))
    write_to_file(all_val_urls, os.path.join(out_dir, 'dev.txt'), os.path.join(out_dir, 'dev.bin'))
    write_to_file(all_train_urls, os.path.join(out_dir, 'train.txt'), os.path.join(out_dir, 'train.bin'))

    print('It takes ', time.strftime('%H:%M:%S', time.gmtime(time.time() - start)))
