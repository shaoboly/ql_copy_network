import pyrouge
from utils.metric_utils import rouge_log
import os

def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir

    rouge_results = r.convert_and_evaluate()
    print(rouge_results)
    return r.output_to_dict(rouge_results)


# result_dict = rouge_eval(r"/home/v-boshao/code/gpt-2-Pytorch/output/ref",
#                          r"/home/v-boshao/code/gpt-2-Pytorch/output/pred")
# rouge_log(result_dict,os.path.dirname(r"/home/v-boshao/code/gpt-2-Pytorch/output/ref"))

data_dir = "/home/v-boshao/data//bert_s2l_copy-quora_tag-11-01-quora-tag2tag/train.csv.test.with_tag.predtag-results"

result_dict = rouge_eval(data_dir+"/ref",
                         data_dir+"/pred")
rouge_log(result_dict,os.path.dirname(data_dir+"/pred"))

def sample_part(ref_dir, dec_dir):
    import shutil, os
    for i in range(10000):
        filename = "%06d_decoded.txt" % i
        refname = "%06d_reference.txt" % i

        out_f_name = "%06d_decoded_sample.txt" % i
        out_refname = "%06d_reference_sample.txt" % i

        shutil.copy(os.path.join(dec_dir, filename), os.path.join(dec_dir, out_f_name))
        shutil.copy(os.path.join(ref_dir,refname),os.path.join(ref_dir,out_refname))

# sample_part(r"/home/v-boshao/data/test-results/ref",
#            r"/home/v-boshao/data/test-results/pred")