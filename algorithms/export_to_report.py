"""Copy the algorithm lessons' results into the LaTeX report (chapter "Algorithm-Level Study").

Run after the 9 lessons (python algorithms/0X_*.py):
    python algorithms/export_to_report.py
Writes:
    report/tables/algo_regression.tex      problem A comparison table
    report/tables/algo_classification.tex  problem B comparison table
    report/generated/algo_numbers.tex      every number quoted in the chapter, as LaTeX macros
    report/figures/algo/*.png              the lessons' charts
"""
import json
import shutil

from common import OUT, ROOT

REPORT = ROOT / "report"
summary = json.loads((OUT / "summary.json").read_text())
facts = json.loads((OUT / "facts.json").read_text())

REG = "A_regression_monthly_units"
CLS = "B_classification_large_order"
ORDER = ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting",
         "XGBoost", "SVM", "KNN"]


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(problem, cols, fmt, sort_key, reverse, caption, label, baseline_name):
    rows = {k: v for k, v in summary[problem].items()}
    base = rows.pop(baseline_name)
    ranked = sorted(rows.items(), key=lambda kv: kv[1][sort_key], reverse=reverse)
    best = ranked[0][0]
    head = " & ".join([r"\textbf{Algorithm}"] + [rf"\textbf{{{tex_escape(c)}}}" for c in cols])
    lines = [r"\begin{table}[htbp]\centering\small",
             rf"\caption{{{caption}}}\label{{{label}}}",
             r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", r"\toprule", head + r" \\ \midrule"]
    for name, sc in ranked:
        cells = [fmt[c].format(sc[c]) for c in cols]
        if name == best:
            cells = [rf"\textbf{{{c}}}" for c in cells]
            name = rf"\textbf{{{name}}}"
        lines.append(" & ".join([name] + cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join([rf"\emph{{{baseline_name}}}"] + [fmt[c].format(base[c]) for c in cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n", ranked, base


reg_tex, reg_rank, reg_base = table(
    REG, ["MAE", "RMSE", "MAPE_%", "R2"], {"MAE": "{:.1f}", "RMSE": "{:.1f}", "MAPE_%": "{:.1f}", "R2": "{:.3f}"},
    "MAE", False, "Problem A, monthly units per product: test results 2025--2026 of the algorithm lessons "
    "(sorted by MAE; best in bold)", "tab:algoreg", "Baseline: same as last month")
cls_tex, cls_rank, cls_base = table(
    CLS, ["accuracy", "precision", "recall", "F1", "ROC_AUC"], {c: "{:.3f}" for c in
                                                                ["accuracy", "precision", "recall", "F1", "ROC_AUC"]},
    "ROC_AUC", True, "Problem B, large order ($\\geq 10$ units): test results 2025--2026 of the algorithm lessons "
    "(sorted by ROC-AUC; best in bold)", "tab:algocls", "Baseline: always small")
(REPORT / "tables" / "algo_regression.tex").write_text(reg_tex.replace("R2", "$R^2$").replace("MAPE\\_\\%", "MAPE (\\%)"))
(REPORT / "tables" / "algo_classification.tex").write_text(cls_tex.replace("ROC\\_AUC", "ROC-AUC"))

# ---------------------------------------------------------------- macros
m = {}
def put(name, value, fmt="{}"):
    m[name] = fmt.format(value) if not isinstance(value, str) else tex_escape(value)

r, c = summary[REG], summary[CLS]
put("AlgRegBest", reg_rank[0][0]); put("AlgRegBestMAE", reg_rank[0][1]["MAE"], "{:.1f}")
put("AlgRegBestRtwo", reg_rank[0][1]["R2"], "{:.3f}")
put("AlgRegWorst", reg_rank[-1][0]); put("AlgRegWorstMAE", reg_rank[-1][1]["MAE"], "{:.1f}")
put("AlgRegBaseMAE", reg_base["MAE"], "{:.1f}")
put("AlgRegGain", 100 * (1 - reg_rank[0][1]["MAE"] / reg_base["MAE"]), "{:.0f}")
put("AlgClsBest", cls_rank[0][0]); put("AlgClsBestAUC", cls_rank[0][1]["ROC_AUC"], "{:.3f}")
put("AlgClsWorst", cls_rank[-1][0]); put("AlgClsWorstAUC", cls_rank[-1][1]["ROC_AUC"], "{:.3f}")
put("AlgClsBaseAcc", 100 * cls_base["accuracy"], "{:.1f}")
for key, name in [("Linear Regression", "Lin"), ("Decision Tree", "DT"), ("Random Forest", "RF"),
                  ("Gradient Boosting", "GB"), ("XGBoost", "XGB"), ("SVM", "SVM"), ("KNN", "KNN")]:
    put(f"Alg{name}MAE", r[key]["MAE"], "{:.1f}"); put(f"Alg{name}Rtwo", r[key]["R2"], "{:.3f}")
for key, name in [("Logistic Regression", "Log"), ("Decision Tree", "DT"), ("Random Forest", "RF"),
                  ("Gradient Boosting", "GB"), ("XGBoost", "XGB"), ("SVM", "SVM"), ("KNN", "KNN")]:
    put(f"Alg{name}AUC", c[key]["ROC_AUC"], "{:.3f}"); put(f"Alg{name}Fone", c[key]["F1"], "{:.3f}")
    put(f"Alg{name}Recall", c[key]["recall"], "{:.3f}"); put(f"Alg{name}Prec", c[key]["precision"], "{:.3f}")
    put(f"Alg{name}Acc", c[key]["accuracy"], "{:.3f}")
km = summary["C_clustering_products"]["K-Means"]

f = facts
spec = {  # macro: (fact key, format)
    "LinBzero": ("lin_b0", "{:.1f}"), "LinBone": ("lin_b1", "{:.3f}"), "LinOneMAE": ("lin_one_mae", "{:.1f}"),
    "LinTopCoef": ("lin_top_coef", "{:.1f}"), "LinTopFeature": ("lin_top_feature", None),
    "LogBone": ("logit_one_b1", "{:.3f}"), "LogPfive": ("logit_p5", "{:.2f}"), "LogPten": ("logit_p10", "{:.2f}"),
    "LogPfifteen": ("logit_p15", "{:.2f}"), "LogThreshold": ("logit_threshold", "{:.2f}"),
    "LogTopOR": ("logit_top_or", "{:.2f}"), "LargeShareTrain": ("large_share_train", "{:.1%}"),
    "LargeShareTest": ("large_share_test", "{:.1%}"),
    "DTBestDepth": ("dt_best_depth", "{:.0f}"), "DTTopShare": ("dt_top_share", "{:.0%}"),
    "DTFitAUCDeep": ("dt_fit_auc_d15", "{:.3f}"), "DTValAUCDeep": ("dt_val_auc_d15", "{:.3f}"),
    "DTValAUCBest": ("dt_val_auc_best", "{:.3f}"),
    "RFOneTreeMAE": ("rf_one_tree_val_mae", "{:.1f}"), "RFValMAEone": ("rf_val_mae_1", "{:.1f}"),
    "RFValMAEhundred": ("rf_val_mae_100", "{:.1f}"), "RFOOB": ("rf_oob_r2", "{:.3f}"),
    "RFTopFeature": ("rf_top_feature", None), "RFTopPerm": ("rf_top_perm", "{:.1f}"),
    "GBManualStart": ("gb_manual_start", "{:.1f}"), "GBManualFive": ("gb_manual_after5", "{:.1f}"),
    "GBRate": ("gb_rate", "{:g}"), "GBTrees": ("gb_trees", "{:.0f}"), "GBTopFeature": ("gb_top_feature", None),
    "XGBBestN": ("xgb_best_n", "{:.0f}"), "XGBValAUC": ("xgb_val_auc", "{:.3f}"),
    "XGBPosWeight": ("xgb_pos_weight", "{:.2f}"), "XGBRegTrees": ("xgb_reg_trees", "{:.0f}"),
    "XGBTopGain": ("xgb_top_gain", "{:.0%}"),
    "SVMRawAUC": ("svm_raw_auc", "{:.3f}"), "SVMScaledAUC": ("svm_scaled_auc", "{:.3f}"),
    "SVMLinAUC": ("svm_linear_auc", "{:.3f}"), "SVMRbfAUC": ("svm_rbf_auc", "{:.3f}"),
    "SVMBestC": ("svm_best_C", "{:g}"), "SVMBestGamma": ("svm_best_gamma", None),
    "SVMBestValAUC": ("svm_best_val_auc", "{:.3f}"), "SVMNSupport": ("svm_n_support", "{:,.0f}"),
    "SVMTrainRows": ("svm_train_rows", "{:,.0f}"), "SVRBest": ("svr_best", None),
    "KNNRawAUC": ("knn_raw_auc", "{:.3f}"), "KNNScaledAUC": ("knn_scaled_auc", "{:.3f}"),
    "KNNBestK": ("knn_best_k", "{:.0f}"), "KNNOneFitAUC": ("knn_k1_fit_auc", "{:.3f}"),
    "KNNOneValAUC": ("knn_k1_val_auc", "{:.3f}"), "KNNRegK": ("knn_reg_k", "{:.0f}"),
    "KMIters": ("km_manual_iters", "{:.0f}"), "KMBestK": ("km_best_k", "{:.0f}"), "KMSil": ("km_sil", "{:.2f}"),
    "KMSilFour": ("km_sil_k4", "{:.2f}"), "KMSmall": ("km_small_cluster", "{:.0f}"),
    "KMProducts": ("km_products", "{:.0f}"),
}
for name, (key, fmt) in spec.items():
    v = f[key]
    put(name, v if fmt is None else fmt.format(v))       # put() escapes % and _ for LaTeX

lines = ["% Generated by algorithms/export_to_report.py - do not edit by hand"]
for k, v in sorted(m.items()):
    lines.append(rf"\newcommand{{\{k}}}{{{v}\xspace}}")
(REPORT / "generated" / "algo_numbers.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- figures
dst = REPORT / "figures" / "algo"
dst.mkdir(exist_ok=True)
for png in OUT.glob("*.png"):
    shutil.copy(png, dst / png.name)
print(f"wrote {len(m)} macros, 2 tables, {len(list(OUT.glob('*.png')))} figures")
