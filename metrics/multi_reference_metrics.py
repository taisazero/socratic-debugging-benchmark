from evaluate import load
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.matching import max_weight_matching
from tqdm import tqdm


class BaseMetric:
    def __init__(self, metric_name):
        self.metric = load(metric_name)
        self.metric_name = metric_name

    def compute(self, predictions, references, **kwargs):
        raise NotImplementedError
    
    def compute_overall_score(self, predictions, references):
        assert len(predictions) == len(references), "Length of predictions and references must be the same."
        if isinstance(predictions[0], list):
            return self.compute_multiple_predictions(predictions, references)
        max_scores = []
        for pred, ref_list in zip(predictions, references):
            scores = [self.compute(predictions=[pred], references=[ref]) for ref in ref_list]
            max_score = max(scores, key=self._score_key)
            max_scores.append(max_score)
        
        return self._aggregate(max_scores)
    
    """
    description: Compute the score between each prediction and all references, then take the maximum score.
    param:
        predictions: A list of predictions.
        references: A list of references.
    return: The maximum score aggregated over all samples in predictions.
    """
    def compute_multiple_predictions(self, predictions, references):
        assert len(predictions) == len(references), "Length of predictions and references must be the same."

        max_scores = []
        for pred_list, ref_list in zip(predictions, references):
            scores = []
            for pred in pred_list:
                for ref in ref_list:
                    score = self.compute(predictions=[pred], references=[ref])
                    scores.append(score)
            max_score = max(scores, key=self._score_key)
            max_scores.append(max_score)
        
        return self._aggregate(max_scores)
    

    def compute_thoroughness(self, predictions, references, log=False):
        assert len(predictions) == len(references), "Length of predictions and references must be the same."
       
        # handle if both lists are empty
        if len(predictions) == 0 and len(references) == 0:
            return {
                'true_positives': [],
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        # ensure that predictions and references are lists of lists
        assert isinstance(predictions[0], list) and isinstance(references[0], list), "predictions and references must be lists of lists"
        true_positives = []
        precisions = []
        recalls = []
        f1s = []
        # if log use tqdm to show progress bar else use zip make it concise code-wise

        if log:
            lambda_func = tqdm(zip(predictions, references), total=len(predictions), desc="Computing thoroughness")
        else:
            lambda_func = zip(predictions, references)
        
        for pred_list, ref_list in lambda_func:
            # Create a bipartite graph
            B = nx.Graph()
            for i, p in enumerate(pred_list):
                B.add_node(f"0_{i}_{p}", bipartite=0)
            for i, r in enumerate(ref_list):
                B.add_node(f"1_{i}_{r}", bipartite=1)

            # Add edges with weight as the score between predictions and references
            for i, pred in enumerate(pred_list):
                for j, ref in enumerate(ref_list):
                    score = self.compute(predictions=[pred], references=[ref])
                    # use self._score_key(score) to get the score since score is a dict
                    pred_id = f"0_{i}_{pred}"
                    ref_id = f"1_{j}_{ref}"
                    B.add_edge(pred_id, ref_id, weight=self._score_key(score))


            # Find maximum bipartite matching using the weight as the score
            matching = max_weight_matching(B)
            # change matching to a dict of pred: ref
            matching_dict = {}
            for pred, ref in matching:
                if pred.startswith('0_'):
                    matching_dict[pred] = ref
                elif ref.startswith('0_'):
                    matching_dict[ref] = pred
            # Compute the total score of the maximum matching
            tp = sum(B[pred][ref]['weight'] for pred, ref in matching_dict.items() if pred.startswith('0_'))
            true_positives.append(tp)
            precision = tp / len(pred_list)
            precisions.append(precision)
            recall = tp / len(ref_list)
            recalls.append(recall)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1s.append(f1)

        return {
            'true_positives': true_positives,
            'precision': sum(precisions) / len(precisions),
            'recall': sum(recalls) / len(recalls),
            'f1': sum(f1s) / len(f1s)
        }

    def compute_single(self, prediction, references):
        if isinstance(prediction, list):
            return self.compute_single_for_multi_prediction(prediction, references)
        scores = [self.compute(predictions=[prediction], references=[ref]) for ref in references]
        max_score = max(scores, key=self._score_key)
        max_ref = references[scores.index(max_score)]
        return self._score_key(max_score), max_ref
    
    def compute_single_for_multi_prediction(self, predictions, references):
        scores = []
        for pred in predictions: 
            for ref in references: 
                score = self.compute(predictions=[pred], references=[ref])
                scores.append(score)
        max_score = max(scores, key=self._score_key)
        # scratch:
                # score index is 13
                # ref index is 13 % 4 = 1
                # pred index is 3 because 13 % 10 = 3 but 14 % 10 = 4 so % not work for pred
                # pred index is 3 when score index is 14
                # pred index is 4 when score index is 16
                # pred index is 0 when score index is 3
                # so pred index = score index // len(references)
                # pred index = 13 // 4 = 3
                # pred index = 14 // 4 = 3
                # pred index = 16 // 4 = 4
        max_ref = references[scores.index(max_score) % len(references)]
        max_pred = predictions[scores.index(max_score) // len(references)]
        return self._score_key(max_score), {'pred': max_pred, 'ref': max_ref}
    
    def _score_key(self, score):
        raise NotImplementedError("Subclasses should implement the _score_key method.")

    def _aggregate(self, scores):
        return sum(self._score_key(score) for score in scores) / len(scores)


class Bleu4Metric(BaseMetric):
    def __init__(self):
        super().__init__('bleu')

    def compute(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references, **kwargs)
    
    def _score_key(self, score):
        return score['bleu']


class RougeLMetric(BaseMetric):
    def __init__(self):
        super().__init__('rouge')

    def compute(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)
    
    def _score_key(self, score):
        return score['rougeL']
    


class Rouge1Metric(BaseMetric):
    def __init__(self):
        super().__init__('rouge')

    def compute(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)
    
    def _score_key(self, score):
        return score['rouge1']

    
class Rouge2Metric(BaseMetric):
    def __init__(self):
        super().__init__('rouge')

    def compute(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)

    def _score_key(self, score):
        return score['rouge2']

    

class BertScoreF1Metric(BaseMetric):
    def __init__(self, lang='en', model_type='microsoft/deberta-xlarge-mnli'):
        self.lang = lang
        self.model_type = model_type
        super().__init__('bertscore')

    def compute(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references,
                                    lang=self.lang, model_type=self.model_type)
    
    def _score_key(self, score):
        if len(score['f1']) > 1:
            raise ValueError("BERTScore F1 score has more than one value.")
        else:
            return score['f1'][0]
