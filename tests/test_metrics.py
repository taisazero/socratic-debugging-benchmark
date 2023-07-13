from metrics.multi_reference_metrics import Bleu4Metric, RougeLMetric, BertScoreF1Metric, BaseMetric
import pytest
import pytest
import networkx as nx

class MockMetric:
    def compute(self, predictions, references):
        return {"score": int(predictions[0] == references[0])}

class MockMetricRatio:
    def compute(self, predictions, references):
        return {"score": len(predictions[0]) / (len(predictions[0]) + len(references[0]))}
    
class MockMonk2 (BaseMetric):
    def __init__(self):
        self.metric = MockMetricRatio()

    def _score_key(self, score):
        return score['score']

    def compute(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)
    
# Mock the metric object for testing
class MockMonk (BaseMetric):
    def __init__(self):
        self.metric = MockMetric()

    def _score_key(self, score):
        return score['score']

    def compute(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)



# Test cases
@pytest.fixture
def some_class():
    return MockMonk(), MockMonk2()

def test_compute_thoroughness_empty_lists(some_class):
    
    result = some_class[0].compute_thoroughness([], [])
    assert result == {'true_positives': [], 'precision': 0, 'recall': 0, 'f1': 0}

def test_compute_thoroughness_type_1(some_class):
    with pytest.raises(AssertionError):
        result = some_class[0].compute_thoroughness([''], [''])

def test_compute_thoroughness_type_2(some_class):
    with pytest.raises(AssertionError):
        result = some_class[0].compute_thoroughness([['']], [''])

def test_compute_thoroughness_single_entry(some_class):
    result = some_class[0].compute_thoroughness([["a"]], [["a"]])
    assert result == {'true_positives': [1], 'precision': 1, 'recall': 1, 'f1': 1}

def test_compute_thoroughness_multiple_entries(some_class):
    result = some_class[0].compute_thoroughness([["a", "b", "c"]], [["b", "c", "d"]])
    assert result == {'true_positives': [2], 'precision': 2/3, 'recall': 2/3, 'f1': 2/3}

def test_compute_thoroughness_mismatched_lengths(some_class):
    with pytest.raises(AssertionError):
        some_class[0].compute_thoroughness([["a", "b", "c"]], [["b", "c"], ["d", "e"]])

def test_compute_thoroughness_multiple_lists(some_class):
    result = some_class[0].compute_thoroughness([["a", "b", "c"], ["d", "e"]], [["b", "c", "d"], ["e", "f"]])
    assert result == {
        'true_positives': [2, 1],
        'precision': (2/3 + 1/2) / 2,
        'recall': (2/3 + 1/2) / 2,
        'f1': (2/3 + 1/2) / 2
    }

def test_compute_thoroughness_no_common_elements(some_class):
    result = some_class[0].compute_thoroughness([["a", "b", "c"]], [["d", "e", "f"]])
    assert result == {'true_positives': [0], 'precision': 0, 'recall': 0, 'f1': 0}

    

def test_ratio_compute_thoroughness_empty_lists(some_class):
    result = some_class[1].compute_thoroughness([], [])
    assert result == {'true_positives': [], 'precision': 0, 'recall': 0, 'f1': 0}

def test_ratio_compute_thoroughness_single_entry(some_class):
    result = some_class[1].compute_thoroughness([["a"]], [["a"]])
    assert result == {'true_positives': [0.5], 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}

def test_ratio_compute_thoroughness_multiple_entries(some_class):
    result = some_class[1].compute_thoroughness([["a", "b", "c"]], [["b", "c", "d"]])
    assert result == {'true_positives': [1.5], 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}

def test_ratio_compute_thoroughness_mismatched_lengths(some_class):
    with pytest.raises(AssertionError):
        some_class[1].compute_thoroughness([["a", "b", "c"]], [["b", "c"], ["d", "e"]])

def test_ratio_compute_thoroughness_multiple_lists(some_class):
    result = some_class[1].compute_thoroughness([["a", "b", "c"], ["d", "e"]], [["b", "c", "d"], ["e", "f"]])
    assert result == {
        'true_positives': [1.5, 1],
        # tp / len(pred_list)
        'precision': (1.5 / 3 + 1 / 2) / 2,
        # tp / len(ref_list)
        'recall': (1.5 / 3 + 1 / 2) / 2,
        # 2 * precision * recall / (precision + recall)
        'f1': (0.5 + (2 * (1 / 2 * 1 / 2) / (1 / 2 + 1 / 2))) / 2
    }

def test_ratio_compute_thoroughness_distinct_ratios(some_class):
    result = some_class[1].compute_thoroughness([["a", "b", "c"], ["d", "e"]], [["b", "c", "d"], ["e", "f", "g", "h"]])
    assert result == {
        'true_positives': [1.5, 1],
        'precision': (1.5 / 3 + 1 / 2) / 2,
        'recall': (1.5 / 3 + 1 / 4) / 2,
        # 2 * precision * recall / (precision + recall)
        'f1': (0.5 + (2 * (1 / 2 * 1 / 4) / (1 / 2 + 1 / 4))) / 2
    }

def test_ratio_compute_thoroughness_no_common_elements(some_class):
    result = some_class[1].compute_thoroughness([["a", "b", "c"]], [["d", "e", "f"]])
    assert result == {'true_positives': [1.5], 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}

def test_metrics():
    bleu4 = Bleu4Metric()
    rougeL = RougeLMetric()
    bertscore_f1 = BertScoreF1Metric()

    predictions = ["hello there, I am badeed", "general kenobi kappacino frank"]
    references = [
        ["hello there, I am badeed", "hi here! you like cats?"],
        ["general kenobi kappacino frank", "general grievous badeed neice"]
    ]

    assert bleu4.compute_overall_score(predictions, references) == 1.0
    assert rougeL.compute_overall_score(predictions, references) == 1.0
    assert bertscore_f1.compute_overall_score(predictions, references) == 1.0



if __name__ == "__main__":
    test_metrics()
    test_compute_thoroughness_empty_lists()
    test_compute_thoroughness_type_1()
    test_compute_thoroughness_type_2()
    test_compute_thoroughness_single_entry()
    test_compute_thoroughness_multiple_entries()
    test_compute_thoroughness_mismatched_lengths()
    test_compute_thoroughness_multiple_lists()
    test_compute_thoroughness_no_common_elements()

    test_ratio_compute_thoroughness_empty_lists()
    test_ratio_compute_thoroughness_single_entry()
    test_ratio_compute_thoroughness_multiple_entries()
    test_ratio_compute_thoroughness_mismatched_lengths()
    test_ratio_compute_thoroughness_multiple_lists()
    test_ratio_compute_thoroughness_no_common_elements()
    test_ratio_compute_thoroughness_distinct_ratios()

    print("All tests passed!")