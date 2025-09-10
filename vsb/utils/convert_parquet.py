import argparse, os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

SPARSE_STRUCT = pa.struct([
    pa.field("indices", pa.list_(pa.uint32())),
    pa.field("values",  pa.list_(pa.float32())),
])

def cast_list_values(arr: pa.Array, to=pa.float32()) -> pa.Array:
    """Cast a list<T> to list<float32> regardless of inner int widths."""
    if not pa.types.is_list(arr.type):
        raise ValueError(f"'values' must be a list, got {arr.type}")
    target = pa.list_(to)
    try:
        return pc.cast(arr, target, safe=False)
    except pa.ArrowInvalid:
        # Fallback: rebuild list with explicit cast of child values
        offsets = arr.offsets if hasattr(arr, "offsets") else arr.value_offsets()
        child   = pc.cast(arr.values, to, safe=False)
        return pa.ListArray.from_arrays(offsets, child, type=target)

def ensure_column(table: pa.Table, name: str):
    try:
        return table.column(name)
    except KeyError:
        return None

def get_list_f32(col: pa.ChunkedArray | pa.Array) -> pa.ListArray:
    # Combine to one chunk
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if not pa.types.is_list(arr.type):
        raise ValueError(f"'values' must be list<T>, got {arr.type}")
    # Cast inner to float32
    child = pc.cast(arr.values, pa.float32(), safe=False)
    return pa.ListArray.from_arrays(arr.offsets, child, type=pa.list_(pa.float32()))


def rewrite_file(src: Path, dst: Path, expected_dim: int | None, normalize: bool):
    tbl = pq.read_table(src)

    # --- id ---
    id_col = ensure_column(tbl, "id")
    if id_col is None:
        raise RuntimeError(f"{src}: missing required 'id' column")
    if not pa.types.is_string(id_col.type):
        id_col = pc.cast(id_col, pa.string(), safe=False)

    # --- values ---
    # Your parquet meta printed leaf as "values.list.element"; Arrow exposes top-level "values".
    val_col = ensure_column(tbl, "values")
    if val_col is None:
        # Some writers flatten awkwardly; try to detect a stray "element" column and bail with clarity.
        if "element" in tbl.column_names:
            raise RuntimeError(f"{src}: saw 'element' leaf but no top-level 'values'. "
                               "Read with pyarrow (not pandas), or regenerate the source parquet properly.")
        raise RuntimeError(f"{src}: missing required 'values' column")

    # Convert list elements to float32
    val_col = tbl.column("values") if "values" in tbl.column_names else None
    if val_col is None:
        if "element" in tbl.column_names:
            raise RuntimeError(f"{src}: saw leaf 'element' but no top-level 'values' — read with PyArrow, not pandas.")
        raise RuntimeError(f"{src}: missing required 'values' column")

    vals = get_list_f32(val_col)  # ListArray<float32>

    # Optional: normalize uint8→[0,1] (works on the combined listarray)
    if normalize:
        child = pc.divide(vals.values, pa.scalar(255.0, type=pa.float32()))
        vals  = pa.ListArray.from_arrays(vals.offsets, child, type=pa.list_(pa.float32()))

    # Dim check: use first non-null list
    if expected_dim is not None:
        nn = pc.drop_null(vals)
        if len(nn) > 0:
            if len(nn[0]) != expected_dim:
                raise RuntimeError(f"{src}: values dim {len(nn[0])} != expected {expected_dim}")

    # --- metadata (STRING or NULL) ---
    meta_col = ensure_column(tbl, "metadata")
    if meta_col is None:
        meta_col = pa.nulls(len(tbl), type=pa.string())
    elif not pa.types.is_string(meta_col.type):
        meta_col = pc.cast(meta_col, pa.string(), safe=False)

    # --- sparse_values (STRUCT or NULL) ---
    # Your source has an INT32 placeholder; Pinecone expects STRUCT. We emit NULLs of the right STRUCT type.
    sparse_col = pa.nulls(len(tbl), type=SPARSE_STRUCT)

    # Assemble in required order
    new_tbl = pa.table({
        "id":            id_col,
        "values":        val_col,
        "sparse_values": sparse_col,
        "metadata":      meta_col,
    }, schema=pa.schema([
        pa.field("id", pa.string()),
        pa.field("values", pa.list_(pa.float32())),
        pa.field("sparse_values", SPARSE_STRUCT),
        pa.field("metadata", pa.string()),
    ]))

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_tbl, dst, compression="snappy")

def main():
    """
    Rewrite Parquet files for Pinecone bulk import
    """
    ap = argparse.ArgumentParser(description="Rewrite Parquet files for Pinecone bulk import")
    ap.add_argument("--input", required=True, help="Input file or directory")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--expected-dim", type=int, default=None, help="Assert vector length N")
    ap.add_argument("--normalize", action="store_true", help="Normalize uint8 → float32 in [0,1]")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    files = []
    if in_path.is_dir():
        files = list(in_path.rglob("*.parquet"))
    elif in_path.is_file():
        files = [in_path]
    else:
        raise SystemExit(f"Not found: {in_path}")

    if not files:
        raise SystemExit("No .parquet files found")

    for f in files:
        rel = f.name if in_path.is_file() else f.relative_to(in_path).as_posix()
        dst = out_dir.joinpath(rel)
        rewrite_file(f, dst, args.expected_dim, args.normalize)
        print(f"✅ {f} -> {dst}")

if __name__ == "__main__":
    main()
