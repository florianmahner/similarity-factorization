from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pysrf import SRF
from scipy.stats import hypergeom


def run_srf(drug_drug: np.ndarray, rank: int = 20, rho: float = 3.0) -> np.ndarray:
    srf = SRF(rank=rank, rho=rho, max_outer=30, max_inner=20, verbose=0)
    srf.fit(drug_drug)
    return srf.w_


def _enrichment(
    w: np.ndarray,
    drug_info: pd.DataFrame,
    annotation_df: pd.DataFrame,
    value_column: str,
    label: str,
    top_n: int,
    min_count: int,
    pvalue_thresh: float,
    fold_thresh: float,
) -> pd.DataFrame:
    drug_ann = drug_info.merge(annotation_df, on="drug_id", how="left")
    results = []
    for dim in range(w.shape[1]):
        top_idx = np.argsort(w[:, dim])[-top_n:][::-1]
        top_drugs = drug_info.iloc[top_idx]["drug_id"].values
        top_ann = drug_ann[drug_ann["drug_id"].isin(top_drugs)]

        for value in annotation_df[value_column].dropna().unique():
            n_total = (drug_ann[value_column] == value).sum()
            if n_total < min_count:
                continue
            n_top = (top_ann[value_column] == value).sum()
            pval = hypergeom.sf(n_top - 1, len(drug_info), n_total, top_n)
            fold = (n_top / top_n) / (n_total / len(drug_info))
            if pval < pvalue_thresh and fold > fold_thresh:
                results.append(
                    {
                        "dimension": dim,
                        label: value,
                        "n_top": n_top,
                        "n_total": n_total,
                        "fold_enrichment": fold,
                        "pvalue": pval,
                    }
                )
    return pd.DataFrame(results)


def compute_atc_enrichment(
    w: np.ndarray,
    drug_info: pd.DataFrame,
    atc_codes: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    atc_codes = atc_codes.copy()
    atc_codes["atc_level1"] = atc_codes["atc_code"].str[0]
    return _enrichment(
        w,
        drug_info,
        atc_codes,
        value_column="atc_level1",
        label="atc_class",
        top_n=top_n,
        min_count=1,
        pvalue_thresh=0.05,
        fold_thresh=1.0,
    )


def compute_target_enrichment(
    w: np.ndarray,
    drug_info: pd.DataFrame,
    drug_targets: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    return _enrichment(
        w,
        drug_info,
        drug_targets,
        value_column="target_id",
        label="target_id",
        top_n=top_n,
        min_count=5,
        pvalue_thresh=0.01,
        fold_thresh=2.0,
    )


def compute_side_effect_enrichment(
    w: np.ndarray,
    drug_info: pd.DataFrame,
    side_effects: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    if side_effects.empty:
        return pd.DataFrame()
    se_counts = side_effects["side_effect"].value_counts()
    common = se_counts[se_counts >= 20].index
    filtered = side_effects[side_effects["side_effect"].isin(common)].copy()
    return _enrichment(
        w,
        drug_info,
        filtered,
        value_column="side_effect",
        label="side_effect",
        top_n=top_n,
        min_count=20,
        pvalue_thresh=0.01,
        fold_thresh=2.0,
    )


def save_enrichment_results(
    w: np.ndarray,
    drug_info: pd.DataFrame,
    atc_enrich: pd.DataFrame,
    target_enrich: pd.DataFrame,
    se_enrich: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "srf_factors.npy", w)
    atc_enrich.to_csv(output_dir / "atc_enrichment.csv", index=False)
    target_enrich.to_csv(output_dir / "target_enrichment.csv", index=False)
    if not se_enrich.empty:
        se_enrich.to_csv(output_dir / "side_effect_enrichment.csv", index=False)

    top_drugs = []
    for dim in range(w.shape[1]):
        top_idx = np.argsort(w[:, dim])[-10:][::-1]
        for rank_idx, idx in enumerate(top_idx):
            top_drugs.append(
                {
                    "dimension": dim,
                    "rank": rank_idx,
                    "drug_id": drug_info.iloc[idx]["drug_id"],
                    "drug_name": drug_info.iloc[idx].get("name", ""),
                    "score": w[idx, dim],
                }
            )
    pd.DataFrame(top_drugs).to_csv(
        output_dir / "top_drugs_per_dimension.csv", index=False
    )

    summary = {
        "n_dimensions": w.shape[1],
        "n_drugs": w.shape[0],
        "sig_atc_dims": int(
            atc_enrich[atc_enrich["pvalue"] < 0.05]["dimension"].nunique()
        ),
        "sig_target_dims": int(
            target_enrich[target_enrich["pvalue"] < 0.01]["dimension"].nunique()
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
