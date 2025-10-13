#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PostgreSQL loader (accelerated) with:
- PostGIS POINTZ geometry (+ GIST) for coordinates
- FTS (tsvector) + trigger (+ GIN) for metadata
- Fast annotations_terms via NumPy + COPY
- Optional annotations_json aggregation (+ GIN) via --enable-json

Default schema: ns
"""

import argparse
import os
import io
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, PendingRollbackError
import time


# -----------------------------
# Args
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="PostgreSQL loader with PostGIS + FTS and accelerated annotations COPY.")
    ap.add_argument(
        "--url",
        required=False,
        default=os.getenv("DB_URL"),
        help="SQLAlchemy DB URL (postgresql://user:pass@host/db). Falls back to DB_URL or DATABASE_URL env var if omitted."
    )
    ap.add_argument("--data-dir", default="./", help="Directory containing Parquet files")
    ap.add_argument("--schema", default="ns", help="Target schema (default: ns)")
    ap.add_argument("--if-exists", choices=["replace", "append"], default="replace", help="Behavior for coordinates/metadata")
    ap.add_argument("--batch-cols", type=int, default=150, help="terms_* columns to melt per batch (smaller uses less RAM)")
    ap.add_argument("--stage-chunksize", type=int, default=50000, help="pandas.to_sql() chunksize for staging loads")
    ap.add_argument("--enable-json", action="store_true", help="Also build annotations_json (slow)")
    ap.add_argument("--srid", type=int, default=4326, help="SRID for geometry(POINTZ). Default 4326")
    return ap.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def ensure_schema(engine: Engine, schema: str):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        conn.execute(text(f"SET search_path TO {schema}, public;"))


def ensure_extensions(engine: Engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent;"))


def is_finite_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return np.isfinite(s.to_numpy(copy=False))


# -----------------------------
# Coordinates (POINTZ + GIST)
# -----------------------------
def build_coordinates(engine: Engine, df: pd.DataFrame, schema: str, chunksize: int, if_exists: str, srid: int):
    print("→ coordinates: preparing dataframe")
    must_have = ["study_id", "x", "y", "z"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise KeyError(f"coordinates missing columns: {missing}")
    df = df[must_have].copy()
    df["study_id"] = df["study_id"].astype(str)
    for c in ["x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    finite_mask = is_finite_series(df["x"]) & is_finite_series(df["y"]) & is_finite_series(df["z"])
    bad = (~finite_mask).sum()
    if bad:
        print(f"   … dropping {bad:,} non-finite rows from coordinates")
    df = df[finite_mask].reset_index(drop=True)

    with engine.begin() as conn:
        if if_exists == "replace":
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}.coordinates CASCADE;"))
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}.coordinates_stage CASCADE;"))
        conn.execute(text(f"""
            CREATE UNLOGGED TABLE IF NOT EXISTS {schema}.coordinates_stage (
                study_id TEXT NOT NULL,
                x DOUBLE PRECISION NOT NULL,
                y DOUBLE PRECISION NOT NULL,
                z DOUBLE PRECISION NOT NULL
            );
        """))

    print("→ coordinates: loading staging (to_sql)")
    # to_sql can leave connections in a bad transaction state on some DB errors; retry once
    def _to_sql_with_retry(engine_obj):
        try:
            df.to_sql("coordinates_stage", engine_obj, schema=schema, if_exists="append", index=False,
                      chunksize=chunksize, method="multi")
            return None
        except (OperationalError, PendingRollbackError) as e:
            return e

    err = _to_sql_with_retry(engine)
    if err is not None:
        print("   … detected DB transaction error during to_sql; retrying once after brief sleep")
        try:
            time.sleep(1)
            # Create a fresh engine and retry once
            new_engine = create_engine(engine.url, pool_pre_ping=True)
            err2 = _to_sql_with_retry(new_engine)
            if err2 is not None:
                raise err2
            engine = new_engine
        except Exception:
            raise

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {schema}.coordinates (
                study_id TEXT NOT NULL,
                geom geometry(POINTZ, {srid}) NOT NULL
            );
        """))
        if if_exists == "replace":
            conn.execute(text(f"TRUNCATE TABLE {schema}.coordinates;"))
        print("→ coordinates: populating geometry from staging")
        conn.execute(text(f"""
            INSERT INTO {schema}.coordinates (study_id, geom)
            SELECT study_id,
                   ST_SetSRID(ST_MakePoint(x, y, z), {srid})::geometry(POINTZ, {srid})
            FROM {schema}.coordinates_stage;
        """))
        print("→ coordinates: indexing & analyze")
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_coordinates_study ON {schema}.coordinates (study_id);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_coordinates_geom_gist ON {schema}.coordinates USING GIST (geom);"))
        conn.execute(text(f"ANALYZE {schema}.coordinates;"))
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.coordinates_stage;"))
    print("→ coordinates (POINTZ + GIST) done.")


# -----------------------------
# Metadata (FTS tsvector + trigger + GIN)
# -----------------------------
def build_metadata(engine: Engine, df: pd.DataFrame, schema: str, if_exists: str):
    print("→ metadata: preparing & creating table")
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    with engine.begin() as conn:
        if if_exists == "replace":
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}.metadata CASCADE;"))
        cols = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(f"{c} DOUBLE PRECISION")
            else:
                cols.append(f"{c} TEXT")
        cols.append("fts tsvector")
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {schema}.metadata ({', '.join(cols)});"))

    print("→ metadata: bulk inserting (to_sql)")
    # Same retry pattern as coordinates
    def _meta_to_sql_with_retry(engine_obj):
        try:
            df.to_sql("metadata", engine_obj, schema=schema, if_exists="append", index=False, chunksize=20000, method="multi")
            return None
        except (OperationalError, PendingRollbackError) as e:
            return e

    merr = _meta_to_sql_with_retry(engine)
    if merr is not None:
        print("   … detected DB transaction error during metadata to_sql; retrying once after brief sleep")
        time.sleep(1)
        new_engine = create_engine(engine.url, pool_pre_ping=True)
        merr2 = _meta_to_sql_with_retry(new_engine)
        if merr2 is not None:
            raise merr2
        engine = new_engine

    with engine.begin() as conn:
        res = conn.execute(text(f"""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = 'metadata';
        """), {"schema": schema}).fetchall()
        text_cols = [r[0] for r in res if r[0] != "fts" and r[1] in ("text", "character varying")]
        if text_cols:
            cols_expr = " || ' ' || ".join([f"coalesce({c},'')" for c in text_cols])
            print("→ metadata: computing tsvector over text columns")
            conn.execute(text(f"UPDATE {schema}.metadata SET fts = to_tsvector('pg_catalog.english', {cols_expr});"))
            print("→ metadata: creating GIN index & trigger")
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_metadata_fts ON {schema}.metadata USING GIN (fts);"))
            conn.execute(text(f"DROP TRIGGER IF EXISTS metadata_fts_update ON {schema}.metadata;"))
            conn.execute(text(f"""
                CREATE TRIGGER metadata_fts_update
                BEFORE INSERT OR UPDATE ON {schema}.metadata
                FOR EACH ROW EXECUTE FUNCTION tsvector_update_trigger(
                    'fts', 'pg_catalog.english', {", ".join(text_cols)}
                );
            """))
            conn.execute(text(f"ANALYZE {schema}.metadata;"))
    print("→ metadata (FTS + trigger) done.")


# -----------------------------
# Annotations -> sparse terms via NumPy + COPY (+ optional JSONB)
# -----------------------------
def copy_terms(engine: Engine, schema: str, rows: List[Tuple[str, str, str, float]]):
    buf = io.StringIO()
    for study_id, contrast_id, term, weight in rows:
        # COPY text format uses \N for NULL
        cval = r'\N' if contrast_id is None else str(contrast_id)
        buf.write(f"{study_id}\t{cval}\t{term}\t{weight}\n")
    buf.seek(0)
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("SET LOCAL synchronous_commit = off;")
            cur.copy_expert(
                f"COPY {schema}.annotations_terms (study_id, contrast_id, term, weight) FROM STDIN WITH (FORMAT text)",
                buf,
            )
        raw.commit()
    finally:
        raw.close()


def build_annotations(engine: Engine, df: pd.DataFrame, schema: str, batch_cols: int, enable_json: bool = False):
    print("→ annotations: preparing")
    fixed = {"id", "study_id", "contrast_id"}
    term_cols = [c for c in df.columns if c not in fixed and str(c).startswith("terms_")]
    if not term_cols:
        raise RuntimeError("No term columns found in annotations.*")

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.annotations_terms CASCADE;"))
        conn.execute(text(f"""
            CREATE UNLOGGED TABLE {schema}.annotations_terms (
                study_id    TEXT NOT NULL,
                contrast_id TEXT,
                term        TEXT NOT NULL,
                weight      DOUBLE PRECISION NOT NULL
            );
        """))
        if enable_json:
            conn.execute(text(f"DROP TABLE IF EXISTS {schema}.annotations_json CASCADE;"))
            conn.execute(text(f"""
                CREATE TABLE {schema}.annotations_json (
                    study_id    TEXT NOT NULL,
                    contrast_id TEXT,
                    terms       JSONB NOT NULL,
                    PRIMARY KEY (study_id, contrast_id)
                );
            """))

    id_vars = ["study_id", "contrast_id"]
    total_inserted = 0

    # Pre-extract ID arrays once
    sid_arr = df["study_id"].astype(str).to_numpy(copy=False)
    cid_series = df["contrast_id"] if "contrast_id" in df.columns else pd.Series([None]*len(df))
    cid_arr = cid_series.where(pd.notna(cid_series), None).astype(object).to_numpy(copy=False)

    for i in range(0, len(term_cols), batch_cols):
        cols = term_cols[i:i+batch_cols]

        # Filter out columns that are entirely null or <= 0 to avoid useless processing
        nonempty = [c for c in cols if (pd.to_numeric(df[c], errors="coerce") > 0).any()]
        if not nonempty:
            continue

        # Build rows with NumPy (avoid huge melted DataFrame)
        term_rows: List[Tuple[str, str, str, float]] = []
        for c in nonempty:
            col = pd.to_numeric(df[c], errors="coerce").to_numpy(copy=False)
            mask = np.isfinite(col) & (col > 0)
            if not mask.any():
                continue
            idx = np.nonzero(mask)[0]
            term = re.sub(r"^terms_[^_]*__", "", str(c)).strip().lower()
            term_rows.extend(zip(sid_arr[idx], cid_arr[idx], [term]*len(idx), col[idx].astype(float)))

        if term_rows:
            copy_terms(engine, schema, term_rows)
            total_inserted += len(term_rows)
            print(f"   … copied {len(term_rows):,} rows (cumulative {total_inserted:,})")

    # Indexes AFTER bulk load (faster)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_annotations_terms_term ON {schema}.annotations_terms (term);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_annotations_terms_study ON {schema}.annotations_terms (study_id);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_annotations_terms_term_study ON {schema}.annotations_terms (term, study_id);"))
        conn.execute(text(f"ANALYZE {schema}.annotations_terms;"))
        # Build PK/unique AFTER load to avoid per-row maintenance
        conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_annotations_terms ON {schema}.annotations_terms (study_id, contrast_id, term);"))
        conn.execute(text(f"ALTER TABLE {schema}.annotations_terms ADD CONSTRAINT pk_annotations_terms PRIMARY KEY USING INDEX ux_annotations_terms;"))

        if enable_json:
            print("→ annotations_json: aggregating (this may take a while)")
            conn.execute(text("SET LOCAL work_mem = '512MB';"))
            conn.execute(text("SET LOCAL maintenance_work_mem = '1GB';"))
            conn.execute(text(f"""
                INSERT INTO {schema}.annotations_json (study_id, contrast_id, terms)
                SELECT study_id, contrast_id, jsonb_object_agg(term, weight)
                FROM {schema}.annotations_terms
                GROUP BY study_id, contrast_id
                ON CONFLICT (study_id, contrast_id) DO UPDATE
                SET terms = EXCLUDED.terms;
            """))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_annotations_json_terms_gin ON {schema}.annotations_json USING GIN (terms);"))
            conn.execute(text(f"ANALYZE {schema}.annotations_json;"))

    print(f"→ annotations_terms total inserted: {total_inserted:,}")
    if enable_json:
        print("   … annotations_json populated and indexed.")
    print("   … annotations done.")


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    if not args.url:
        raise SystemExit("Missing DB URL. Provide --url or set DB_URL environment variable.")
    engine = create_engine(args.url, pool_pre_ping=True)

    ensure_schema(engine, args.schema)
    ensure_extensions(engine)

    # --- Basic connectivity sanity checks & status prints ---
    with engine.begin() as conn:
        sv = conn.execute(text("SELECT version();")).fetchone()
        db = conn.execute(text("SELECT current_database();")).fetchone()
        sch = conn.execute(text("SELECT current_schema();")).fetchone()
    print("✅ server_version:", sv[0].splitlines()[0])
    print("✅ current_database:", db[0])
    print("✅ current_schema:", sch[0])

    # Load Parquet files
    print("📦 loading Parquet files...")
    coords = load_parquet(os.path.join(args.data_dir, "coordinates.parquet"))
    meta   = load_parquet(os.path.join(args.data_dir, "metadata.parquet"))
    ann    = load_parquet(os.path.join(args.data_dir, "annotations.parquet"))
    print(f"📐 shapes -> coordinates: {coords.shape}, metadata: {meta.shape}, annotations: {ann.shape}")

    # Build
    print("\n=== Build: coordinates ===")
    build_coordinates(engine, coords, args.schema, args.stage_chunksize, args.if_exists, args.srid)

    print("\n=== Build: metadata ===")
    build_metadata(engine, meta, args.schema, args.if_exists)

    print("\n=== Build: annotations ===")
    build_annotations(engine, ann, args.schema, args.batch_cols, enable_json=args.enable_json)

    print("\n=== Ready ===")
    print(f"- coordinates  : {args.schema}.coordinates (geometry(POINTZ,{args.srid}) + GIST)")
    print(f"- metadata     : {args.schema}.metadata (FTS + trigger + GIN)")
    print(f"- annotations  : {args.schema}.annotations_terms (sparse via COPY)" + (" + annotations_json (GIN)" if args.enable_json else ""))


if __name__ == "__main__":
    main()
