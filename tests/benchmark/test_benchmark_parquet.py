import pytest

import vsb
from vsb.workloads.mnist.mnist import MnistTest
from vsb.workloads.nq_768_tasb.nq_768_tasb import Nq768TasbTest
from vsb.workloads.yfcc.yfcc import YFCCTest


def test_topk_recalc_mnist_euclidean(benchmark):
    # Benchmark recalculating the top-k nearest neighbors when loading a
    # subset of a parquet file - euclidean distance.
    # Use mnist-test as it is the smallest reduced parquet file & passages
    # (600) supporting euclidean, but should be sufficient to see interesting
    # performance characteristics.
    workload = MnistTest("topk_bench_euclidean", cache_dir=vsb.default_cache_dir)

    # Expected top_k nearest neighbors for the first and last query. Disable
    # black formatting as it tries to put every number on its own line (!)
    # fmt: off
    expected_neighbours_0 = ["522", "243", "349", "103", "288", "263", "301", "84",
                             "223", "307", "371", "505", "422", "337", "387", "258",
                             "377", "96", "133", "230", "52", "567", "353", "183",
                             "280", "441", "167", "101", "54", "419", "42", "45", "305",
                             "57", "389", "267", "562", "411", "413", "467", "500",
                             "247", "436", "115", "227", "383", "412", "478", "324",
                             "87", "123", "26", "285", "195", "15", "338", "172",
                             "599", "148", "297", "140", "154", "586", "423", "420",
                             "185", "417", "409", "454", "91", "142", "33", "124",
                             "362", "319", "402", "289", "71", "38", "364", "19", "272",
                             "461", "518", "484", "214", "257", "276", "270", "366",
                             "43", "574", "310", "346", "72", "514", "207", "418",
                             "482", "193"]

    expected_neighbours_19 = ["92", "580", "566", "336", "26", "162", "217", "116",
                              "170", "550", "442", "354", "369", "344", "338", "438",
                              "350", "402", "520", "372", "237", "314", "564", "280",
                              "226", "194", "289", "428", "271", "436", "275", "304",
                              "142", "373", "363", "115", "383", "379", "57", "45",
                              "418", "576", "176", "364", "384", "267", "362", "342",
                              "4", "247", "412", "183", "584", "459", "590", "585",
                              "476", "482", "334", "61", "374", "166", "110", "322",
                              "153", "434", "441", "460", "329", "58", "258", "592",
                              "160", "396", "389", "257", "148", "285", "297", "232",
                              "536", "313", "167", "474", "163", "541", "419", "502",
                              "409", "413", "104", "346", "292", "518", "282", "288",
                              "22", "227", "38", "48"]

    # fmt: on
    def benchmark_query_generation():
        query_iter = workload.get_query_iter(1, 0, 0)
        for i, (tenant, query) in enumerate(query_iter):
            # Sanity check the correct number of neighbors, and
            # first and last nearest queries neighbors are correct.
            assert len(query.neighbors) == 100
            match i:
                case 0:
                    assert query.neighbors == expected_neighbours_0
                case 19:
                    assert query.neighbors == expected_neighbours_19
            pass

    benchmark(benchmark_query_generation)


def test_topk_recalc_yfcc_euclidean(benchmark):
    # Benchmark recalculating the top-k nearest filtered neighbors when loading a
    # subset of a parquet file - euclidean distance.
    # Use yfcc-test to test filter performance for each query over records.
    workload = YFCCTest(
        "topk_bench_filtered_euclidean", cache_dir=vsb.default_cache_dir
    )

    # Expected top_k nearest neighbors for the first and last query. Disable
    # black formatting as it tries to put every number on its own line (!)
    # fmt: off
    expected_neighbours_0 = ['7829', '9712', '9020', '3244', '8794', '4140', '4973', '6922', '2140', '6978']

    expected_neighbours_499 = []

    # fmt: on
    def benchmark_query_generation():
        query_iter = workload.get_query_iter(1, 0, 0)
        for i, (tenant, query) in enumerate(query_iter):
            # Sanity check the correct number of neighbors, and
            # first and last nearest queries neighbors are correct.
            assert len(query.neighbors) <= 10
            match i:
                case 0:
                    assert query.neighbors == expected_neighbours_0
                case 499:
                    assert query.neighbors == expected_neighbours_499
            pass

    benchmark(benchmark_query_generation)
