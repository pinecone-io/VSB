from ..base import VectorWorkload
from ..dataset import Dataset


class MNIST(VectorWorkload):
    def __init__(self):
        print("MNIST::__init__")
        self.dataset = Dataset(name="mnist")
        self.dataset.load_documents()
        print(self.dataset.documents)
        self.records = Dataset.split_dataframe(self.dataset.documents, 100)
        print(self.records)
        self.operation_count = 0
        self.operation_limit = 10

    def next_record_batch(self):
        print("MNIST::next_record_batch")

    def execute_next_request(self, db) ->bool:
        print("MNIST::execute_next_request")
        self.operation_count += 1
        return self.operation_count < self.operation_limit

