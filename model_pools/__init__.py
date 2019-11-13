from model_pools.bert_final_summarizer import BertFinalSummarizer
from model_pools.bert_sentence_summarizer import BertSentenceSummarizer
from model_pools.bert_sentence_summarizer_copy import BertSentenceSummarizerCopy
from model_pools.bert_sentence_summarizer_copy_rl import BertSentenceSummarizerCopyRL
from model_pools.bert_sentence_summarizer_copy_rl_v2 import BertSentenceSummarizerCopyRLV2
from model_pools.bert_sentence_summarizer_copy_v2 import BertSentenceSummarizerCopyV2
from model_pools.bert_summarizer import BertSummarizer
from model_pools.bert_summarizer_copy import BertSummarizerCopy
from model_pools.bert_summarizer_dec import BertSummarizerDec
from model_pools.bert_summarizer_dec_draft import BertSummarizerDecDraft
from model_pools.bert_summarizer_dec_v2 import BertSummarizerDecV2
from model_pools.bert_summarizer_dec_v3 import BertSummarizerDecV3
from model_pools.bert_summarizer_dec_v4 import BertSummarizerDecV4
from model_pools.bert_summarizer_dec_v4_1 import BertSummarizerDecV4V1
from model_pools.bert_summarizer_dec_v4_2 import BertSummarizerDecV4V2
from model_pools.bert_two_stage_summarizer import BertTwoStageSummarizer
from model_pools.bert_two_stage_summarizer_v2 import BertTwoStageSummarizerV2
from model_pools.bert_s2s_sepv_copy import BertS2LCopySep
from model_pools.bert_s2s_sepv import BertS2LSep
from model_pools.bert_s2s_sepv_copy_noise import BertS2LCopySepNoise
from model_pools.multi_gpu_baseline_copy import MultiGPUBaselineCopy
from model_pools.multi_gpu_baseline_copy_add import MultiGPUBaselineCopyAdd
from model_pools.multi_gpu_baseline_add_decoder import MultiGPUBaselineCopyAddOnlyDecoder
from model_pools.multi_gpu_add_improve_transformer import MultiGPUBaselineCopyAddImprove

model_pools = {
    'summarize_bert_baseline': BertSummarizer,
    'summarize_bert_baseline_copy': BertSummarizerCopy,
    'bert_two_stage_summarizer': BertTwoStageSummarizer,
    'bert_two_stage_summarizer_v2': BertTwoStageSummarizerV2,
    'bert_final_summarizer': BertFinalSummarizer,
    'bert_sentence_summarizer': BertSentenceSummarizer,
    'bert_sentence_summarizer_copy': BertSentenceSummarizerCopy,
    'bert_sentence_summarizer_copy_v2': BertSentenceSummarizerCopyV2,
    'bert_sentence_summarizer_copy_rl': BertSentenceSummarizerCopyRL,
    'bert_sentence_summarizer_copy_rl_v2': BertSentenceSummarizerCopyRLV2,
    'bert_summarizer_dec': BertSummarizerDec,
    'bert_summarizer_dec_v2': BertSummarizerDecV2,
    'bert_summarizer_dec_v3': BertSummarizerDecV3,
    'bert_summarizer_dec_v4': BertSummarizerDecV4,
    'bert_summarizer_dec_v4_1': BertSummarizerDecV4V1,
    'bert_summarizer_dec_v4_2': BertSummarizerDecV4V2,
    'bert_summarizer_dec_draft': BertSummarizerDecDraft,
    'bert_s2l':BertS2LSep,
    'bert_s2l_copy':BertS2LCopySep,
    'bert_s2l_copy_noise':BertS2LCopySepNoise,
    'multi_gpu_bert_s2l_copy':MultiGPUBaselineCopy,
    'multi_gpu_bert_s2l_copy_add':MultiGPUBaselineCopyAdd,
    'multi_gpu_bert_s2l_copy_add_decoder':MultiGPUBaselineCopyAddOnlyDecoder,
    'multi_gpu_bert_s2l_copy_improve_transformer':MultiGPUBaselineCopyAddImprove
}
