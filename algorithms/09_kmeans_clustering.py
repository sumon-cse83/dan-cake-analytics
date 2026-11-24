# %% [markdown]
# # Algorithm 9 – K-Means Clustering
# QUESTION: which products SELL IN A SIMILAR WAY?  There is no "right answer" column -> UNSUPERVISED learning.
# IDEA (repeat until nothing changes):
#     1. put k centre points at random
#     2. give every product to its NEAREST centre
#     3. move each centre to the AVERAGE of its products
# Result: k groups (clusters) where products inside a group are close to each other.
# Good:  simple, fast, easy to explain to managers ("we have 3 types of products").
# Weak:  YOU choose k; needs scaling; assumes round, similar-sized groups; sensitive to outliers.
# Run:   python algorithms/09_kmeans_clustering.py

# %% Step 1. Build one row per product (a "profile") from 2020-2024
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import ROOT, SEED, read_orders, save_fact, save_result, savefig

train, _ = read_orders()          # clusters are learned from 2020-2024 only

def profile(d):
    months = d.groupby("Product_ID").month_index.nunique()             # months the product was on sale
    p = d.groupby("Product_ID").agg(
        avg_price=("log_price", lambda x: np.exp(x).mean()),           # BDT per unit
        units_per_month=("Quantity", "sum"),
        avg_order_qty=("Quantity", "mean"),
        avg_discount=("Discount_%", "mean"),
        large_order_share=("large_order", "mean"),
        december_share=("is_december", "mean"),                         # how much it depends on the December peak
    )
    p["units_per_month"] = p["units_per_month"] / months                # per month -> new products are compared fairly
    return p

prof = profile(train)
names = pd.read_csv(ROOT / "Dan_Cake_Bangladesh_500000_Transactions_2020_2026.csv",
                    usecols=["Product_ID", "Product_Name", "Category"]).drop_duplicates("Product_ID").set_index("Product_ID")
print("products:", len(prof))
print(prof.head().round(2).to_string())

# %% Step 2. Scale (K-Means uses distance, like KNN)
COLS = list(prof.columns)
scaler = StandardScaler().fit(prof[COLS])
X = scaler.transform(prof[COLS])

# %% Step 3. K-Means BY HAND for k=3 - watch the centres settle
rng = np.random.default_rng(SEED)
centres = X[rng.choice(len(X), 3, replace=False)]
for it in range(1, 11):
    dist = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)    # squared distance to each centre
    label = dist.argmin(axis=1)                                          # step 2: nearest centre
    inertia = dist.min(axis=1).sum()                                     # total squared distance (lower = tighter)
    new_centres = np.array([X[label == c].mean(axis=0) for c in range(3)])  # step 3: move centres
    print(f"iteration {it}: inertia = {inertia:.1f}, group sizes = {np.bincount(label, minlength=3).tolist()}")
    if np.allclose(new_centres, centres):
        print("centres stopped moving -> done")
        break
    centres = new_centres

# %% Step 4. Choose k: elbow (inertia) and silhouette
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)   # n_init: try 20 random starts, keep the best
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
print("\n" + pd.DataFrame({"k": ks, "inertia": np.round(inertia, 1), "silhouette": np.round(sil, 3)}).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow: look for the bend"); ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette: higher = better separated"); ax[1].set_xlabel("k")
savefig(fig, "09_choose_k.png")
# Silhouette for one product = (distance to the nearest OTHER cluster - distance inside its own) / the larger one.
# Between -1 and 1. Above ~0.25 = some structure, above ~0.5 = clear groups.
best_k = list(ks)[int(np.argmax(sil))]
print("best k by silhouette:", best_k)
# On this data k=2 wins clearly (0.54): a small group of bread/breakfast products sells very differently.
# The other cakes are similar to each other (silhouette ~0.2 for k>=3 = weak sub-groups). Try best_k = 4 to explore them.

# %% Step 5. Final K-Means and what each cluster means
km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(X)
prof["cluster"] = km.labels_
summary = prof.groupby("cluster")[COLS].mean().round(2)
summary["products"] = prof.cluster.value_counts().sort_index()
print("\nAverage profile per cluster:\n" + summary.to_string())
# Compare each cluster with the overall average to NAME it (e.g. "high price, low volume = premium").
z = pd.DataFrame(km.cluster_centers_, columns=COLS).round(2)
print("\nCluster centres in standard deviations from the average (+ = above average, - = below):\n" + z.to_string())
for c in range(best_k):
    members = names.loc[prof.index[prof.cluster == c]]
    print(f"\ncluster {c}: {members.Category.value_counts().to_dict()}")
    print("   e.g.", ", ".join(members.Product_Name.head(6)))

# %% Step 6. Picture: squeeze the columns into 2 with PCA to draw the clusters
pca = PCA(n_components=2, random_state=SEED).fit(X)
xy = pca.transform(X)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sc = ax.scatter(xy[:, 0], xy[:, 1], c=prof.cluster, cmap="tab10", s=70)
cxy = pca.transform(km.cluster_centers_)
ax.scatter(cxy[:, 0], cxy[:, 1], marker="X", s=250, c="black", label="centres")
for pid, (a, b) in zip(prof.index, xy):
    ax.annotate(pid, (a, b), fontsize=6, alpha=0.7)
ax.legend(*sc.legend_elements(), title="cluster", loc="best")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variation)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variation)")
ax.set_title(f"K-Means product clusters (k={best_k})")
savefig(fig, "09_clusters.png")

# %% Step 7. Use the model: which group does a BRAND-NEW product belong to?
# km.predict() measures the distance to each learned centre (after the SAME scaling) and picks the nearest.
idea = pd.DataFrame([
    {"avg_price": 120, "units_per_month": 700, "avg_order_qty": 6.0, "avg_discount": 2.3, "large_order_share": 0.18, "december_share": 0.08},
    {"avg_price": 40, "units_per_month": 300, "avg_order_qty": 6.5, "avg_discount": 2.6, "large_order_share": 0.22, "december_share": 0.09},
], index=["new cake (like a pound cake)", "new bread (like Milk Bread)"])
idea["cluster"] = km.predict(scaler.transform(idea[COLS]))
print("\nNew product ideas and their cluster:\n" + idea.to_string())
# A planner can now treat the new product like the existing products in that cluster (pricing, stock, promotions).
# Note: do NOT compare clusters of 2020-24 with profiles from 2025-26 directly - 2026 has no December yet and
# large orders became more common, so every product's profile shifts. Re-cluster each full year instead.

# %% Step 8. Save the result (and key numbers for the report)
save_fact(km_manual_iters=it, km_best_k=best_k, km_sil=max(sil), km_sil_k4=sil[list(ks).index(4)],
          km_small_cluster=int(prof.cluster.value_counts().min()), km_products=len(prof))
save_result("K-Means", "C_clustering_products", {"k": best_k, "silhouette": round(float(max(sil)), 3),
                                                 "cluster_sizes": prof.cluster.value_counts().sort_index().tolist()})
