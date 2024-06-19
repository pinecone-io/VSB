from vsb.metrics import Recall
from vsb.vsb_types import SearchRequest


def test_recall_equal():
    # Test recall() for equal length actual and expected lists.
    assert Recall._calculate(["1"], ["1"]) == 1.0
    assert Recall._calculate(["0"], ["1"]) == 0
    assert Recall._calculate(["1", "3"], ["1", "2"]) == 0.5
    assert Recall._calculate(["3", "1"], ["1", "2"]) == 0.5
    assert Recall._calculate(["1", "2"], ["2", "1"]) == 1
    assert Recall._calculate(["2", "3", "4", "5"], ["1", "2", "3", "4"]) == 0.75


def test_recall_actual_fewer_expected():
    # Test recall() when actual matches is fewer than expected - i.e.
    # query returned less than requested top_k.
    assert Recall._calculate(["1"], ["1"]) == 1.0
    assert Recall._calculate(["1"], ["1", "2"]) == 0.5
    assert Recall._calculate(["3"], ["1", "2"]) == 0
    assert Recall._calculate(["1"], ["1", "2", "3", "4"]) == 0.25
    assert Recall._calculate(["1", "2"], ["1", "2", "3", "4"]) == 0.5


def test_recall_more_neighbors_than_topk():
    # Test Recall when the Request has more neighbors present than the specified top_k
    # - in which case we should only consider the first K elements when calculating
    # recall.
    request = SearchRequest(values=[], top_k=1, neighbors=["1", "2"])
    assert Recall.measure(request, ["1"]) == 1.0
    assert Recall.measure(request, ["2"]) == 0.0
