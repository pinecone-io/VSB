from vsb.metrics import Recall, AveragePrecision, ReciprocalRank
from vsb.vsb_types import SearchRequest
from math import isclose


def test_recall_equal():
    # Test recall() for equal length actual and expected lists.
    assert Recall._calculate([], []) == 1.0
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


def test_average_precision():
    # Test AveragePrecision() for actual and expected lists.
    assert AveragePrecision._calculate([], []) == 1.0
    assert AveragePrecision._calculate(["1"], ["1"]) == 1.0
    assert AveragePrecision._calculate(["0"], ["1"]) == 0.0
    assert AveragePrecision._calculate(["1", "3"], ["1", "2"]) == 0.75
    assert AveragePrecision._calculate(["3", "1"], ["1", "2"]) == 0.25
    assert AveragePrecision._calculate(["1", "2"], ["2", "1"]) == 0.5
    assert isclose(
        AveragePrecision._calculate(["2", "3", "4", "5"], ["1", "2", "3", "4"]),
        0.479166,
        abs_tol=0.001,
    )

    expected = ["a", "b", "c", "d", "e", "f"]
    request = SearchRequest(values=[], top_k=1, neighbors=expected)
    assert AveragePrecision.measure(request, ["a"]) == 1.0
    assert AveragePrecision._calculate(["a", "b"], expected) == 1.0
    assert AveragePrecision._calculate(["z"], expected) == 0.0
    assert AveragePrecision._calculate(expected, expected) == 1.0
    assert isclose(
        AveragePrecision._calculate(["a", "x"], expected), 0.75, abs_tol=0.001
    )
    assert isclose(
        AveragePrecision._calculate(["a", "c"], expected), 0.75, abs_tol=0.001
    )
    assert isclose(
        AveragePrecision._calculate(["b", "a", "c"], expected), 0.666, abs_tol=0.001
    )
    assert isclose(
        AveragePrecision._calculate(["c", "b", "a"], expected), 0.5, abs_tol=0.001
    )
    assert isclose(
        AveragePrecision._calculate(["a", "d", "c", "b"], expected),
        0.79166,
        abs_tol=0.001,
    )
    assert isclose(
        AveragePrecision._calculate(["a", "b", "c", "f", "e", "d"], expected),
        0.925,
        abs_tol=0.001,
    )
    assert isclose(
        AveragePrecision._calculate(["f", "e", "d", "c", "b", "a"], expected),
        0.3833,
        abs_tol=0.001,
    )
    assert isclose(
        AveragePrecision._calculate(["a", "z", "b", "d", "f", "x"], expected),
        0.6972,
        abs_tol=0.001,
    )


def test_average_precision_more_neighbors_than_topk():
    # Test AveragePrecision when the Request has more neighbors present than the specified top_k
    # - in which case we should only consider the first K elements when calculating
    # average precision.
    request = SearchRequest(values=[], top_k=1, neighbors=["1", "2"])
    assert AveragePrecision.measure(request, ["1"]) == 1.0
    assert AveragePrecision.measure(request, ["2"]) == 0.0


def test_reciprocal_rank_equal():
    # Test ReciprocalRank() for equal length actual and expected lists.
    assert ReciprocalRank._calculate([], []) == 1.0
    assert ReciprocalRank._calculate(["1"], ["1"]) == 1.0
    assert ReciprocalRank._calculate(["0"], ["1"]) == 0.0
    assert ReciprocalRank._calculate(["1", "3"], ["1", "2"]) == 1.0
    assert ReciprocalRank._calculate(["3", "1"], ["1", "2"]) == 0.5
    assert ReciprocalRank._calculate(["1", "2"], ["2", "1"]) == 1.0
    assert ReciprocalRank._calculate(["2", "3", "4", "5"], ["1", "2", "3", "4"]) == 1.0
    assert ReciprocalRank._calculate(["2", "3", "4", "5"], ["9", "6", "1", "5"]) == 0.25


def test_reciprocal_rank_actual_fewer_expected():
    # Test ReciprocalRank() when actual matches is fewer than expected - i.e.
    # query returned less than requested top_k.
    assert ReciprocalRank._calculate([], ["1"]) == 0.0
    assert ReciprocalRank._calculate(["1"], ["1", "2"]) == 1.0
    assert ReciprocalRank._calculate(["3"], ["1", "2"]) == 0.0
    assert ReciprocalRank._calculate(["1"], ["1", "2", "3", "4"]) == 1.0
    assert ReciprocalRank._calculate(["1", "2"], ["2", "3", "4"]) == 0.5


def test_reciprocal_rank_more_neighbors_than_topk():
    # Test ReciprocalRank when the Request has more neighbors present than the specified top_k
    # - in which case we should only consider the first K elements when calculating
    # reciprocal rank.
    request = SearchRequest(values=[], top_k=1, neighbors=["1", "2"])
    assert ReciprocalRank.measure(request, ["1"]) == 1.0
    assert ReciprocalRank.measure(request, ["2"]) == 0.0
    assert ReciprocalRank.measure(request, ["3"]) == 0.0
