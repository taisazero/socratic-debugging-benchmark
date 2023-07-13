from metrics.multi_reference_metrics import *
from metrics import ALL_METRICS
import prettytable as pt
import pandas as pd
from tqdm import tqdm
"""
    This class is used to compute the metrics for a given set of predictions and references.
    It is used in the evaluation script.
Args:
metrics: A dictionary of metric names and their corresponding Metric objects.
e.g. {'bleu4': Bleu4Metric(), 'rougeL': RougeLMetric()}
export_to_excel: A boolean indicating whether to export the results to an Excel file.
export_path: The path to the Excel file.
"""
class MetricComputer:
    def __init__(self, export_to_excel=False, metrics=None,
                 export_path="results.xlsx"):
        self.metrics = ALL_METRICS if metrics is None else metrics
        self.table = pt.PrettyTable()
        self.table.field_names = ["Metric", "Score"]
        self.table.align["Metric"] = "l"
        self.table.align["Score"] = "r"
        self.table.float_format = "0.2"
        self.export_to_excel = export_to_excel
        self.export_path = export_path
        

    def compute(self, predictions, references, contexts=None):
        # Compute overall scores
        scores = {}   
        for metric_name, metric in tqdm(self.metrics.items(), desc="Computing overall metrics"):
                score = metric.compute_overall_score(predictions, references)
                scores[metric_name] = score
                self.table.add_row([metric_name, score])

        print(self.table)
        # Compute scores for individual predictions
        if self.export_to_excel:
            self._export_excel(predictions, references, contexts)

        return scores
    
    def compute_thoroughness(self, predictions, references, contexts=None):
        # Compute thoroughness scores
        scores = {}
        self.table = pt.PrettyTable()
        self.table.field_names = ["Metric", "Precision", "Recall", "F1"]
        self.table.align["Metric"] = "l"
        self.table.align["Precision"] = "r"
        self.table.align["Recall"] = "r"
        self.table.align["F1"] = "r"
        self.table.float_format = "0.3"
        for metric_name, metric in tqdm(self.metrics.items(), desc="Computing overall metrics"):
                print(f'Computing {metric_name}...')
                score = metric.compute_thoroughness(predictions, references, log=True)
                precision = score['precision']
                recall = score['recall']
                f1 = score['f1']
                scores  [metric_name] = {'precision': precision, 'recall': recall, 'f1': f1} 
                self.table.add_row([metric_name, precision, recall, f1])


        print(self.table)
        # Compute scores for individual predictions
        if self.export_to_excel:
            self._export_excel(predictions, references, contexts, metric_type='thoroughness')

        return scores

    """
    Description: Export the results to an Excel file.
    Args:
        predictions: A list of predictions.
        references: A list of references.
        contexts: A list of contexts.
        metric_type: A string indicating whether to compute the scores using thoroughness (P, R, F1) or standard scoring.
    """
    def _export_excel(self, predictions, references, contexts=None, metric_type='single'):
        results = []
        if contexts is not None:
            assert len(predictions) == len(references) == len(contexts)
        else:
            assert len(predictions) == len(references)
            contexts = [None] * len(predictions)
        for metric_name, metric in self.metrics.items():
            for pred, ref_list, context in tqdm(zip(predictions, references, contexts), desc=f"Computing {metric_name} for individual predictions"):
                if metric_type == 'thoroughness':
                    f_scores = metric.compute_thoroughness([pred], [ref_list])
                    if isinstance(pred,list):
                        # get argmax of f_scores['true_positives']
                        max_f_score = max(f_scores['true_positives'])
                        max_f_score_index = f_scores['true_positives'].index(max_f_score)
                        single_score = f_scores['f1']
                        best_result = {'ref': ref_list[max_f_score_index], 'pred': pred[max_f_score_index]}

                else:
                    single_score, best_result = metric.compute_single(pred, ref_list)

                if isinstance(best_result, dict):
                    best_ref = best_result['ref']
                    best_pred = best_result['pred']
                else:
                    best_ref = best_result
                    best_pred = pred
                results.append({
                    'context': context if context is not None else "",
                    'prediction': best_pred,
                    'reference': best_ref,  # Assuming the first reference has the maximum score
                    'reference_list': ref_list,  # All references for this prediction
                    'prediction_list': pred,  # All predictions for this reference
                    'metric': metric_name,
                    'score': single_score
                })
        # Create an Excel file with a tab for each metric
        with pd.ExcelWriter(self.export_path) as writer:
            for metric_name, metric in self.metrics.items():
                metric_results = [r for r in results if r['metric'] == metric_name]
                df = pd.DataFrame(metric_results)
                df.to_excel(writer, sheet_name=metric_name, index=False)
        