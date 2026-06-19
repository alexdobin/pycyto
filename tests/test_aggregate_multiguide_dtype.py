"""Regression test for multi-guide (MOI>1) assignment-column dtype inference.

Bug: ``_load_assignments_for_experiment_sample`` reads cyto's per-barcode
``<bc>.assignments.tsv`` with ``pl.read_csv``. For cells assigned more than one
guide (MOI>1), the per-guide columns ``assignment``, ``guide_ids_original``,
``umis``, ``fdr`` and ``log_odds`` hold pipe-delimited values (e.g.
``guide_ids_original = "129|262"``). Polars infers each column's dtype from a
bounded row sample (``infer_schema_length``); when every row in that sample is a
single-guide cell, the numeric-looking columns are inferred as ``i64`` / ``f64``
and the first later multi-guide row blows up with::

    ComputeError: could not parse `129|262` as dtype `i64`
                  at column 'guide_ids_original' (column number 7)

The fix reads those per-guide columns as ``String`` regardless of the sample, so
the dtype is deterministic across every per-barcode file (and uniform for the
later ``pl.concat``).
"""

import polars as pl

from pycyto.aggregate import _load_assignments_for_experiment_sample

# cyto's assignments.tsv schema (one row per CRISPR-detected cell).
HEADER = [
    "cell_id",
    "submatrix_id",
    "cell",
    "moi",
    "n_umi",
    "assignment",
    "guide_ids_original",
    "umis",
    "fdr",
    "log_odds",
    "tested",
]

# The per-guide columns that are pipe-delimited for MOI>1 cells.
PER_GUIDE_COLS = ["assignment", "guide_ids_original", "umis", "fdr", "log_odds"]

# Push the first multi-guide row well past polars' default infer window (100),
# so a naive ``read_csv`` infers the per-guide columns as numeric and then
# crashes on the pipe-delimited value -- faithfully reproducing the GCP failure.
N_SINGLE_GUIDE_ROWS = 150


def _write_assignments_tsv(path, *, with_multiguide: bool) -> None:
    lines = ["\t".join(HEADER)]
    for i in range(N_SINGLE_GUIDE_ROWS):
        # single-guide cell: every per-guide column is a bare scalar.
        lines.append(
            "\t".join(
                [
                    str(1000 + i),  # cell_id
                    str(i),  # submatrix_id
                    f"CELL{i:05d}-D-A01",  # cell
                    "1",  # moi
                    "6",  # n_umi
                    "GENE_Protosp001_A",  # assignment
                    "7",  # guide_ids_original  -> inferred i64
                    "2",  # umis                -> inferred i64
                    "1.5e-06",  # fdr           -> inferred f64
                    "13.7",  # log_odds         -> inferred f64
                    "true",  # tested
                ]
            )
        )
    if with_multiguide:
        # MOI=3 cell appearing AFTER the inference window: pipe-delimited values.
        lines.append(
            "\t".join(
                [
                    "9999",
                    "210",
                    "CELLMULTI-D-A01",
                    "3",
                    "24",
                    "BLNK_Protosp104_B|APOE_Protosp168_B|PICALM_Protosp406_B",
                    "65|129|262",
                    "2|6|3",
                    "1.9e-06|3.4e-08|0.0002",
                    "6.1|17.1|8.2",
                    "true",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _load(tmp_path, *, with_multiguide: bool) -> pl.DataFrame:
    assignments_dir = tmp_path / "assignments"
    assignments_dir.mkdir()
    _write_assignments_tsv(
        assignments_dir / "BC001.assignments.tsv", with_multiguide=with_multiguide
    )
    dfs = _load_assignments_for_experiment_sample(
        root=str(tmp_path),
        crispr_bcs=["BC001"],
        lane_id="1",
        experiment="E1",
        sample="S1",
    )
    assert len(dfs) == 1
    return dfs[0]


def test_multiguide_assignments_load_without_crash(tmp_path):
    """A late multi-guide row must not crash the read; values survive intact."""
    df = _load(tmp_path, with_multiguide=True)
    assert df.height == N_SINGLE_GUIDE_ROWS + 1
    # The pipe-delimited multi-guide values are preserved verbatim.
    assert "65|129|262" in df["guide_ids_original"].to_list()
    assert "2|6|3" in df["umis"].to_list()


def test_per_guide_columns_are_strings(tmp_path):
    """Per-guide columns are read as String, deterministically (no numeric infer)."""
    df = _load(tmp_path, with_multiguide=True)
    for col in PER_GUIDE_COLS:
        assert df.schema[col] == pl.String, (
            f"{col} should be String, got {df.schema[col]}"
        )


def test_per_guide_columns_uniform_dtype_for_single_guide_only_file(tmp_path):
    """A barcode with NO multi-guide cell must still yield String per-guide columns.

    Otherwise it infers numeric dtypes and the later cross-barcode ``pl.concat``
    sees a mixed schema (some files i64, some String).
    """
    df = _load(tmp_path, with_multiguide=False)
    for col in PER_GUIDE_COLS:
        assert df.schema[col] == pl.String, (
            f"{col} should be String, got {df.schema[col]}"
        )
