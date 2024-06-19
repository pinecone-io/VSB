#!/usr/bin/env python3

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import sys


def rewrite_parquet_file_with_new_row_group_size(
    input_file, output_dir, row_group_size, max_rows_per_file
):
    # Open the existing Parquet file as Dataset
    in_dataset = ds.dataset(input_file)

    # Write the batches into the dataset
    pq.write_to_dataset(
        in_dataset,
        basename_template="part-{i}.parquet",
        root_path=output_dir,
        row_group_size=row_group_size,
        max_rows_per_file=max_rows_per_file,
    )


if len(sys.argv) != 5:
    print(
        f"Usage: {sys.argv[0]} <input_parquet> <output_parquet_dir> <row_group_size> <max_rows_per_file>"
    )
    sys.exit(1)

input_parquet_file = sys.argv[1]
output_parquet_dir = sys.argv[2]
row_group_size = int(sys.argv[3])
max_rows_per_file = int(sys.argv[4])

rewrite_parquet_file_with_new_row_group_size(
    input_parquet_file, output_parquet_dir, row_group_size, max_rows_per_file
)
