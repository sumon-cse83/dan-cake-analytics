"""Layer 6c: descriptive machine learning - segmentation with clustering.

Two segmentations that management can act on:
  1. Product portfolio (52 products) - which products behave alike (size, growth, price, seasonality)?
  2. Market segments (region x channel x customer type = 220 cells) - where and how do we sell?
K-Means is compared with Agglomerative (Ward) and Gaussian Mixture clustering; k is chosen with the
silhouette score, and the Davies-Bouldin and Calinski-Harabasz indices are reported as checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config as C
from .utils import plt, save_json, savefig, to_latex_table


def product_profile(d: pd.DataFrame) -> pd.DataFrame:
    ym = d.Date.dt.to_period("M").dt.to_timestamp()
    g = d.groupby("Product_ID", observed=True)
    prof = pd.DataFrame({
        "Product_Name": g.Product_Name.first().astype(str), "Category": g.Category.first().astype(str),
        "net_sales": g.Net_Sales_BDT.sum(), "avg_price": g.Unit_Selling_Price_BDT.mean(),
        "units_per_order": g.Quantity.mean(),
        "distributor_share": d.assign(x=d.Channel == "Distributor").groupby("Product_ID", observed=True).x.mean(),
    })
    y = d.groupby(["Product_ID", "Year"], observed=True).Quantity.sum().unstack()
    prof["unit_growth_23_25"] = ((y[2025] / y[2023]) ** 0.5 - 1) * 100      # 2-year CAGR, all products sold since 2022
    m = d.groupby(["Product_ID", ym], observed=True).Quantity.sum().unstack(0)
    m = m[m.index.year.isin([2023, 2024, 2025])]
    seas = m.groupby(m.index.month).mean() / m.mean()
    prof["december_index"] = seas.loc[12] * 100
    detr = m / m.rolling(12, center=True, min_periods=6).mean()
    prof["volatility"] = detr.std() * 100
    prof.index = prof.index.astype(str)
    return prof


def segment_profile(d: pd.DataFrame) -> pd.DataFrame:
    key = ["Region", "Channel", "Customer_Type"]
    g = d.groupby(key, observed=True)
    prof = pd.DataFrame({"net_sales": g.Net_Sales_BDT.sum(), "orders": g.Quantity.size(), "aov": g.Net_Sales_BDT.mean(),
                         "units_per_order": g.Quantity.mean(), "avg_price": g.Unit_Selling_Price_BDT.mean(),
                         "discount": g["Discount_%"].mean()})
    y = d.groupby(key + ["Year"], observed=True).Net_Sales_BDT.sum().unstack()
    prof["growth_23_25"] = ((y[2025] / y[2023]) ** 0.5 - 1) * 100
    prem = d.assign(gift=d.Pack_Size.astype(str).str.contains(r"x\s*\d+|12 pcs")).groupby(key, observed=True).gift.mean()
    prof["gift_share"] = prem * 100
    prof = prof.reset_index()
    for c in key:
        prof[c] = prof[c].astype(str)
    return prof


def choose_k(X, ks=range(2, 9)):
    rows = []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=C.SEED).fit(X)
        ag = AgglomerativeClustering(k, linkage="ward").fit(X)
        gm = GaussianMixture(k, n_init=5, random_state=C.SEED).fit(X)
        for name, lab in [("K-Means", km.labels_), ("Agglomerative (Ward)", ag.labels_), ("Gaussian Mixture", gm.predict(X))]:
            if len(set(lab)) < 2:
                continue
            rows.append({"Algorithm": name, "k": k, "Silhouette": silhouette_score(X, lab),
                         "Davies-Bouldin": davies_bouldin_score(X, lab), "Calinski-Harabasz": calinski_harabasz_score(X, lab),
                         "inertia": km.inertia_ if name == "K-Means" else np.nan})
    return pd.DataFrame(rows)


def name_product_clusters(prof: pd.DataFrame) -> dict:
    """Business names from cluster averages: price tier first, then revenue size."""
    m = prof.groupby("cluster").agg(price=("avg_price", "mean"), sales=("net_sales", "mean"), growth=("unit_growth_23_25", "mean"))
    names = {m.price.idxmax(): "Premium & gift packs", m.price.idxmin(): "Single-serve value packs"}
    rest = m.drop(index=list(names))
    if len(rest):
        names[rest.sales.idxmax()] = "Core family packs"
        rest = rest.drop(index=rest.sales.idxmax())
    if len(rest):
        names[rest.sales.idxmin()] = "Niche & new lines"
        rest = rest.drop(index=rest.sales.idxmin())
    for i in rest.index:
        names[i] = "High-growth lines" if rest.loc[i, "growth"] >= m.growth.median() else "Mature slow-growth lines"
    return names


def name_market_clusters(sp: pd.DataFrame) -> dict:
    m = sp.groupby("cluster").agg(upo=("units_per_order", "mean"), sales=("net_sales", "sum"), growth=("growth_23_25", "mean"))
    names = {m.upo.idxmax(): "Bulk distribution"}
    rest = m.drop(index=list(names))
    names[rest.sales.idxmax()] = "Core retail trade"
    rest = rest.drop(index=rest.sales.idxmax())
    if len(rest):
        names[rest.growth.idxmax()] = "Emerging high-growth cells"
        rest = rest.drop(index=rest.growth.idxmax())
    for i in rest.index:
        names[i] = "Small-ticket online & regional" if len(rest) == 1 else f"Small-ticket group {i}"
    return names


def run(d: pd.DataFrame) -> dict:
    out: dict = {}
    # ======================================================== 1. product portfolio
    prof = product_profile(d)
    # Order size and channel mix turned out almost identical for every product (see EDA), so they carry no
    # segmentation signal; the portfolio is segmented on economics: size, price, growth, seasonality, stability.
    feats = ["net_sales", "avg_price", "unit_growth_23_25", "december_index", "volatility"]
    Z = prof[feats].copy()
    Z["net_sales"], Z["avg_price"] = np.log(Z.net_sales), np.log(Z.avg_price)
    X = StandardScaler().fit_transform(Z)
    sel = choose_k(X, range(2, 9))
    km_sel = sel[sel.Algorithm == "K-Means"].set_index("k")
    k = int(km_sel.loc[3:5, "Silhouette"].idxmax())          # business-usable range: 3-5 segments
    km = KMeans(k, n_init=50, random_state=C.SEED).fit(X)
    prof["cluster"] = km.labels_
    prof["segment"] = prof.cluster.map(name_product_clusters(prof))
    out["product_k"] = k
    out["product_scores"] = {"silhouette": float(silhouette_score(X, km.labels_)),
                             "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
                             "calinski_harabasz": float(calinski_harabasz_score(X, km.labels_))}
    summ = prof.groupby("segment").agg(Products=("net_sales", "size"), Sales=("net_sales", "sum"), Price=("avg_price", "mean"),
                                       UPO=("units_per_order", "mean"), Growth=("unit_growth_23_25", "mean"),
                                       Dec=("december_index", "mean"), Vol=("volatility", "mean")).sort_values("Sales", ascending=False)
    summ["Share"] = summ.Sales / summ.Sales.sum() * 100
    summ["Examples"] = [", ".join(prof[prof.segment == s].sort_values("net_sales", ascending=False).Product_Name.head(2))
                        for s in summ.index]
    out["product_segments"] = summ.reset_index().to_dict("records")
    t = summ.reset_index()[["segment", "Products", "Share", "Price", "UPO", "Growth", "Dec"]]
    t.columns = ["Segment", "Products", "Sales share %", "Avg price", "Units/order", "Unit growth %/yr", "Dec. index"]
    to_latex_table(t, "product_clusters", f"Product portfolio segments (K-Means, k={k})", "tab:productclusters",
                   fmt={"Products": "{:.0f}", "Sales share %": "{:.1f}", "Avg price": "{:.0f}", "Units/order": "{:.2f}",
                        "Unit growth %/yr": "{:+.1f}", "Dec. index": "{:.0f}"},
                   note="Growth is the compound annual growth of units between 2023 and 2025. December index = "
                        "average December units relative to the product's average month (100 = no seasonality).")
    prof.reset_index().rename(columns={"index": "Product_ID"}).to_csv(C.TABLES / "product_segments.csv", index=False)
    prof.reset_index().rename(columns={"index": "Product_ID"}).to_parquet(C.DASHBOARD_DATA / "product_segments.parquet", index=False)

    # ======================================================== 2. market segments
    sp = segment_profile(d)
    sf = ["net_sales", "aov", "units_per_order", "avg_price", "discount", "growth_23_25", "gift_share"]
    Zs = sp[sf].copy(); Zs["net_sales"] = np.log(Zs.net_sales)
    Xs = StandardScaler().fit_transform(Zs)
    sel_s = choose_k(Xs, range(2, 9))
    ks_sel = sel_s[sel_s.Algorithm == "K-Means"].set_index("k")
    k2 = int(ks_sel.loc[3:6, "Silhouette"].idxmax())
    km2 = KMeans(k2, n_init=50, random_state=C.SEED).fit(Xs)
    sp["cluster"] = km2.labels_
    out["segment_k"] = k2
    out["segment_scores"] = {"silhouette": float(silhouette_score(Xs, km2.labels_)),
                             "davies_bouldin": float(davies_bouldin_score(Xs, km2.labels_)),
                             "calinski_harabasz": float(calinski_harabasz_score(Xs, km2.labels_))}
    s2 = sp.groupby("cluster").agg(Cells=("net_sales", "size"), Sales=("net_sales", "sum"), AOV=("aov", "mean"),
                                   UPO=("units_per_order", "mean"), Growth=("growth_23_25", "mean"),
                                   Channels=("Channel", lambda s: ", ".join(s.value_counts().index[:2])),
                                   Regions=("Region", lambda s: ", ".join(s.value_counts().index[:3]))).sort_values("Sales", ascending=False)
    s2["Share"] = s2.Sales / s2.Sales.sum() * 100
    nm = name_market_clusters(sp)
    sp["segment"] = sp.cluster.map(nm); s2["segment"] = s2.index.map(nm)
    out["market_segments"] = s2.reset_index().to_dict("records")
    t2 = s2[["segment", "Cells", "Share", "AOV", "UPO", "Growth", "Channels", "Regions"]].copy()
    t2.columns = ["Segment", "Cells", "Sales share %", "AOV (BDT)", "Units/order", "Growth %/yr", "Main channels", "Main regions"]
    to_latex_table(t2, "market_clusters", f"Market segments of region $\\times$ channel $\\times$ customer-type cells (K-Means, k={k2})",
                   "tab:marketclusters", fmt={"Cells": "{:.0f}", "Sales share %": "{:.1f}", "AOV (BDT)": "{:.0f}",
                                              "Units/order": "{:.2f}", "Growth %/yr": "{:+.1f}"},
                   col_align="lrrrrrll", font=r"\scriptsize")
    sp.to_parquet(C.DASHBOARD_DATA / "market_segments.parquet", index=False)
    sp.to_csv(C.TABLES / "market_segments.csv", index=False)

    # algorithm comparison table at the chosen k
    comp = pd.concat([sel[sel.k == k].assign(Problem="Products"), sel_s[sel_s.k == k2].assign(Problem="Market cells")])
    comp = comp[["Problem", "Algorithm", "k", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]]
    out["algorithm_comparison"] = comp.to_dict("records")
    to_latex_table(comp, "cluster_comparison", "Comparison of clustering algorithms at the selected number of clusters",
                   "tab:clustercompare", fmt={"k": "{:.0f}", "Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}",
                                              "Calinski-Harabasz": "{:.1f}"}, col_align="llrrrr",
                   note="Higher silhouette and Calinski--Harabasz and lower Davies--Bouldin values indicate better-separated clusters.")

    # ======================================================== figures
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.6))
    ax[0].plot(km_sel.index, km_sel.Silhouette, "o-", label="products"); ax[0].plot(ks_sel.index, ks_sel.Silhouette, "s-", label="market cells")
    ax[0].axvline(k, color=C.PALETTE[0], ls=":", lw=.8); ax[0].axvline(k2, color=C.PALETTE[1], ls=":", lw=.8)
    ax[0].set_title("Silhouette by k"); ax[0].set_xlabel("k"); ax[0].legend(fontsize=7)
    pcs = PCA(2, random_state=C.SEED).fit(X); P = pcs.transform(X)
    for i, s in enumerate(prof.segment.unique()):
        mk = (prof.segment == s).values
        ax[1].scatter(P[mk, 0], P[mk, 1], s=30, label=s, color=C.PALETTE[i])
    ax[1].set_title(f"Products (PCA, {pcs.explained_variance_ratio_.sum():.0%} var.)")
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    Ps = PCA(2, random_state=C.SEED).fit_transform(Xs)
    for i, s in enumerate(sp.segment.unique()):
        mk = (sp.segment == s).values
        ax[2].scatter(Ps[mk, 0], Ps[mk, 1], s=14, label=s, color=C.PALETTE[i + 4])
    ax[2].set_title("Market cells (PCA)"); ax[2].set_xlabel("PC1")
    ax[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    savefig(fig, "ml_clustering")

    save_json(out, "clustering")
    print(f"[cluster] products k={k} sil={out['product_scores']['silhouette']:.3f}; market k={k2} "
          f"sil={out['segment_scores']['silhouette']:.3f}")
    return out
