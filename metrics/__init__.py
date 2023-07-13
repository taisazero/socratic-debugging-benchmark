from .multi_reference_metrics import *
ALL_METRICS = {

    'bleu4': Bleu4Metric(),
    'rougeL': RougeLMetric(),
    'bertscore_f1': BertScoreF1Metric(),
    'rouge1': Rouge1Metric(),
    'rouge2': Rouge2Metric(),
}