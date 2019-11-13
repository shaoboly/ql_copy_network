from data_reading.cnndm_read import CNNDailyProcessor
from data_reading.nyt_read import NYTProcessor
from data_reading.mspars_read import MSProcessor
from data_reading.wikipedia_read import WikiProcessor,WikiProcessorRevsese,\
    WikiProcessorNgramMask,WikiPassage2Sent,\
    WikiProcessorOnlyTarget,WikiProcessorNgramMaskEncoderDecoder,WikiProcessorNgramMaskEncoderDecoderDenoise,\
    WikiMultiTask,WikiMultiTaskP2S
from data_reading.nucle_read import NULEProcessor
from data_reading.noise_read import NoiseProcessor
from data_reading.chinese_reading import IceProcessor
from data_reading.disconfuse import DisConfuProcessor
from data_reading.qg_read import SQUADProcessor,SQUADProcessorAnswer,NQQGProcessorAnswer,BingQG
from data_reading.msn_read import MSNProcessor,MSNProcessorQG,QIProcessor,NewKeyMultiInput,CalloutRead,M2MKeyProcessor,Paraphrase,ParaphraseTagsProcessor,ParaphrasesrcTagsProcessor,ParaphrasesrcTag2Tag,ParaphraseSemanticTag2Tag,ParaphrasesrcTag2Tag_newvocab
from data_reading.translation_read import Tran_En2De
from data_reading.keywords_read import KeysProcessor
from data_reading.trie_experiment import CalloutReadNew,CalloutReadv4

processors = {
    'cnn_dm': CNNDailyProcessor,
    'nyt': NYTProcessor,
    'mspars':MSProcessor,
    'wiki':WikiProcessor,
    'wikire':WikiProcessorRevsese,
    'wikingram':WikiProcessorNgramMask,
    'wikisent':WikiPassage2Sent,
    'wikionlytarget':WikiProcessorOnlyTarget,
    'wikingramende':WikiProcessorNgramMaskEncoderDecoder,
    'wikingramendenoise':WikiProcessorNgramMaskEncoderDecoderDenoise,
    'wikumulti':WikiMultiTask,
    'wikumultip2s':WikiMultiTaskP2S,
    'nucle':NULEProcessor,
    'noise':NoiseProcessor,
    'chineseice':IceProcessor,
    'disconfuse':DisConfuProcessor,
    'squad':SQUADProcessorAnswer,
    'msn':MSNProcessor,
    'm2m':M2MKeyProcessor,
    'en2de':Tran_En2De,
    'keywords':KeysProcessor,
    'msnqg':MSNProcessorQG,
    'qip':QIProcessor,
    'new_key_multi':NewKeyMultiInput,
    "callout":CalloutReadNew,
    "calloutv4":CalloutReadv4,
    'nqqg':NQQGProcessorAnswer,
    'bingqg':BingQG,
    'paraphrase':Paraphrase,
    'quora_tag':ParaphraseTagsProcessor,
    'quora_srctag':ParaphrasesrcTagsProcessor,
    'tag2tag':ParaphrasesrcTag2Tag,
    'semantic2tag':ParaphraseSemanticTag2Tag,
    'new_vocab_tag2tag':ParaphrasesrcTag2Tag_newvocab


}


def abstract2sents_func(params):
    task_name = params.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)
    return processors[task_name].abstract2sents
