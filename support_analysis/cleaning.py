"""Tidies up the ticket table. `basic_clean` is the original cleanup (column names,
duplicates, blanks); `clean_tickets` adds the ticket stuff on top: clean text, fix
dates and priorities, sanity-check CSAT, and toss empty or repeated tickets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .text_cleaning import clean_ticket_text

NULL_TOKENS = {"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "null": np.nan, "N/A": np.nan, "n/a": np.nan}

PRIORITY_MAP = {
    "p1": "urgent", "critical": "urgent", "urgent": "urgent",
    "p2": "high", "high": "high",
    "p3": "medium", "medium": "medium", "normal": "medium",
    "p4": "low", "low": "low",
}


def normalize_column_name(c: object) -> str:
    return str(c).strip().lower().replace(" ", "_")


@dataclass
class CleaningReport:
    n_rows_in: int
    n_rows_out: int
    columns_in: List[str]
    columns_out: List[str]
    missing_by_col: Dict[str, int]
    duplicates_removed: int


def basic_clean(
    df: pd.DataFrame,
    id_col: str = "id",
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, CleaningReport]:
    df0 = df.copy()
    df0.columns = [normalize_column_name(c) for c in df0.columns]

    if drop_columns:
        drop_columns = [normalize_column_name(c) for c in drop_columns]
        df0 = df0.drop(columns=[c for c in drop_columns if c in df0.columns])

    df0 = df0.dropna(axis=1, how="all")

    # Trim strings and turn null-like tokens into real NaN *before* de-duplicating,
    # so " T-001" and "T-001" count as the same ID.
    for c in df0.select_dtypes(include=["object", "string"]).columns:
        df0[c] = df0[c].astype(str).str.strip()
    df0 = df0.replace(NULL_TOKENS)

    if id_col not in df0.columns:
        df0[id_col] = np.arange(1, len(df0) + 1)

    # Rows with a missing ID are kept; only repeated non-null IDs are duplicates.
    dup_mask = df0[id_col].notna() & df0[id_col].duplicated(keep="first")
    duplicates_removed = int(dup_mask.sum())
    df0 = df0.loc[~dup_mask].reset_index(drop=True)

    report = CleaningReport(
        n_rows_in=len(df),
        n_rows_out=len(df0),
        columns_in=[str(c) for c in df.columns],
        columns_out=list(df0.columns),
        missing_by_col={k: int(v) for k, v in df0.isna().sum().items()},
        duplicates_removed=duplicates_removed,
    )
    return df0, report


@dataclass
class TicketCleaningReport:
    basic: CleaningReport
    tickets_with_html: int = 0
    quoted_replies_removed: int = 0
    signatures_removed: int = 0
    tickets_with_pii: int = 0
    pii_redactions: int = 0
    invalid_timestamps: int = 0
    negative_resolution_times: int = 0
    invalid_csat: int = 0
    empty_or_short_dropped: int = 0
    content_duplicates_removed: int = 0
    n_rows_out: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_tickets(
    df: pd.DataFrame,
    id_col: str = "ticket_id",
    text_cols: Sequence[str] = ("subject", "body"),
    created_col: str = "created_at",
    resolved_col: str = "resolved_at",
    min_text_chars: int = 15,
) -> Tuple[pd.DataFrame, TicketCleaningReport]:
    out, base = basic_clean(df, id_col=id_col)
    rep = TicketCleaningReport(basic=base)

    present = [c for c in text_cols if c in out.columns]
    if not present:
        raise ValueError(
            f"None of the text columns {list(text_cols)} were found. Columns after cleaning: {list(out.columns)}"
        )
    missing = [c for c in text_cols if c not in out.columns]
    if missing:
        rep.notes.append(f"Text columns not found and skipped: {missing}")

    # --- Text: combine subject + body, then clean (HTML, replies, signatures, PII) ---
    raw_text = out[present].fillna("").astype(str).agg("\n".join, axis=1)
    results = [clean_ticket_text(t) for t in raw_text]
    out["text"] = [r.text for r in results]
    rep.tickets_with_html = sum(r.had_html for r in results)
    rep.quoted_replies_removed = sum(r.removed_quoted_reply for r in results)
    rep.signatures_removed = sum(r.removed_signature for r in results)
    rep.tickets_with_pii = sum(r.pii_redactions > 0 for r in results)
    rep.pii_redactions = sum(r.pii_redactions for r in results)
    # The raw body may contain PII; keep only the redacted text downstream.
    out = out.drop(columns=[c for c in present if c != "subject"])
    if "subject" in out.columns:
        out["subject"] = [clean_ticket_text(s).text for s in out["subject"]]

    # --- Timestamps and resolution time ---
    for col in (created_col, resolved_col):
        if col in out.columns:
            had_value = out[col].notna()
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed", utc=True)
            rep.invalid_timestamps += int((had_value & out[col].isna()).sum())
    if created_col in out.columns and resolved_col in out.columns:
        hours = (out[resolved_col] - out[created_col]).dt.total_seconds() / 3600
        negative = hours < 0
        rep.negative_resolution_times = int(negative.sum())
        out.loc[negative, resolved_col] = pd.NaT
        out["resolution_hours"] = hours.where(~negative).round(2)

    # --- Categorical normalisation ---
    for col in ("priority", "channel", "status", "product"):
        if col in out.columns:
            out[col] = out[col].str.strip().str.lower()
    if "priority" in out.columns:
        out["priority"] = out["priority"].map(lambda v: PRIORITY_MAP.get(v, v) if isinstance(v, str) else v)

    # --- CSAT must be 1-5 ---
    if "csat" in out.columns:
        csat = pd.to_numeric(out["csat"], errors="coerce")
        bad = csat.notna() & ~csat.between(1, 5)
        rep.invalid_csat = int(bad.sum())
        out["csat"] = csat.where(~bad)

    # --- Drop empty / too-short tickets, then duplicate content from the same customer ---
    short = out["text"].str.len() < min_text_chars
    rep.empty_or_short_dropped = int(short.sum())
    out = out.loc[~short]

    dup_subset = ["text"] + (["customer_id"] if "customer_id" in out.columns else [])
    content_dup = out.duplicated(subset=dup_subset, keep="first")
    rep.content_duplicates_removed = int(content_dup.sum())
    out = out.loc[~content_dup]

    if created_col in out.columns:
        out = out.sort_values(created_col, na_position="last")
    out = out.reset_index(drop=True)
    rep.n_rows_out = len(out)
    return out, rep
