"""Bangla (বাংলা) version support for the LaTeX report.

The generated English tables in report/tables/ are copied to report/tables_bn/ with Bangla captions
and notes (column headers, model names and metrics stay in English, as agreed), and the automatic
insights are written in Bangla to report/generated/insights_bn.tex. Everything is rebuilt from the
same metric files, so the Bangla report stays identical in content to the English one.
"""
from __future__ import annotations

import re

from . import config as C
from .utils import load_json

BS = "\\"
TABLES_BN = C.REPORT / "tables_bn"
MONTHS_BN = {"Jan": "জানুয়ারি", "Feb": "ফেব্রুয়ারি", "Mar": "মার্চ", "Apr": "এপ্রিল", "May": "মে", "Jun": "জুন",
             "Jul": "জুলাই", "Aug": "আগস্ট", "Sep": "সেপ্টেম্বর", "Oct": "অক্টোবর", "Nov": "নভেম্বর", "Dec": "ডিসেম্বর"}


def pct(s: str) -> str:
    """Escape % for LaTeX."""
    return s.replace("%", BS + "%")


def table_texts() -> dict:
    """table name -> (Bangla caption, Bangla note or None)."""
    c, k, e = load_json("classification"), load_json("clustering"), load_json("eda")
    f = load_json("forecasting")
    first, last = e["first_full_year"], e["last_full_year"]
    return {
        "annual_summary": ("বার্ষিক কর্মদক্ষতার সারসংক্ষেপ, 2020--2026",
                           "* 2026 সালে কেবল জানুয়ারি--সেপ্টেম্বর অন্তর্ভুক্ত, তাই এর প্রবৃদ্ধি হিসাব করা হয়নি। "
                           "AOV = গড় অর্ডার-মূল্য; Avg price = গড় একক বিক্রয়মূল্য।"),
        "category_summary": ("পণ্য-ক্যাটাগরি অনুযায়ী আয় ও প্রবৃদ্ধি",
                             f"CAGR প্রথম ও শেষ পূর্ণ বছরের মধ্যে হিসাবকৃত। {first}-এর পরে চালু হওয়া ক্যাটাগরির প্রবৃদ্ধি "
                             "তাদের চালুর বছর থেকে হিসাব করা।"),
        "top_products": ("নিট বিক্রয় অনুযায়ী শীর্ষ দশটি পণ্য (Pareto/ABC বিশ্লেষণ)", None),
        "segment_summary": ("অঞ্চল, চ্যানেল ও গ্রাহকের ধরন অনুযায়ী কর্মদক্ষতা", None),
        "stat_tests": ("প্রধান অনুসন্ধানমূলক হাইপোথিসিসের পরিসংখ্যানিক টেস্ট",
                       "Effect size: Kruskal--Wallis-এর জন্য $" + BS + "varepsilon^2$, Spearman-এর জন্য $" + BS +
                       "rho^2$ এবং chi-square-এর জন্য Cram" + BS + "'er's $V$। $n=500{,}000$ হলে প্রায় যেকোনো পার্থক্যই "
                       "`তাৎপর্যপূর্ণ', তাই ব্যবহারিক গুরুত্ব বিচার করে effect size।"),
        "price_volume": ("বার্ষিক বিক্রয় প্রবৃদ্ধির পরিমাণ (volume) ও মূল্য (price) প্রভাবে বিভাজন",
                         "$(1+g_{sales}) = (1+g_{volume})(1+g_{price})$; এখানে মূল্য মানে ইউনিটপ্রতি নিট আয়, তাই এতে "
                         "পণ্য-মিশ্রণের পরিবর্তনও প্রতিফলিত।"),
        "forecast_models": ("মাসিক পণ্য-চাহিদা পূর্বাভাস: পরীক্ষাকালের কর্মদক্ষতা (এক-ধাপ-অগ্রিম, জানুয়ারি 2025--সেপ্টেম্বর 2026)",
                            "ত্রুটি প্রতি পণ্য-মাসে ইউনিটে। CV RMSE হলো টিউনিংয়ে ব্যবহৃত 2022, 2023 ও 2024-এর "
                            "expanding-window ভাঁজে log স্কেলে গড় validation RMSE। বেসলাইনের কোনো fitting লাগে না।"),
        "forecast_recursive": ("ডিসেম্বর 2024 থেকে recursive 21-মাসের পূর্বাভাস (প্রশিক্ষণের পরে কোনো প্রকৃত ডেটা ব্যবহৃত হয়নি)", None),
        "forecast_company": ("কোম্পানি-স্তরের মাসিক ইউনিট পূর্বাভাস, পরীক্ষাকাল জানুয়ারি 2025--সেপ্টেম্বর 2026", None),
        "future_forecast": ("বারো মাসের অগ্রিম পূর্বাভাস, অক্টোবর 2026 -- সেপ্টেম্বর 2027",
                            f"Units ML raw = reconciliation-এর আগে {f['best_recursive_model']} পণ্য-পূর্বাভাসের যোগফল। চূড়ান্ত "
                            "মোট এসেছে কোম্পানির ইউনিট ও নিট বিক্রয়ের (মিলিয়ন টাকা) Holt-Winters মডেল থেকে; ব্যবধান হলো 2,000টি "
                            "সিমুলেটেড ভবিষ্যৎ-পথের 10তম ও 90তম শতাংশক। পণ্যের পূর্বাভাস এই মোটের সঙ্গে মেলানো হয়েছে।"),
        "classification_models": ("বড় অর্ডার ক্লাসিফিকেশন ($" + BS + "geq$10 ইউনিট): পরীক্ষাকালের কর্মদক্ষতা, 2025--2026",
                                  f"পরীক্ষা-বছরে বড় অর্ডারের হার: {c['positive_rate_test']:.1%}; যে ক্লাসিফায়ার সবসময় "
                                  f"`বড় নয়' বলে তার accuracy হতো {c['majority_baseline_accuracy']:.1%}, তাই কেবল accuracy "
                                  "বিভ্রান্তিকর। Threshold 2024 validation-বছরে F1 সর্বোচ্চ করে। KNN 60,000-সারির প্রশিক্ষণ-নমুনা "
                                  "ব্যবহার করে। Lift@10% = সর্বোচ্চ স্কোরের 10% অর্ডারে বড় অর্ডারের হার ভাগ সামগ্রিক হার।"),
        "product_clusters": (f"পণ্য-পোর্টফোলিওর বিভাগ (K-Means, k={k['product_k']})",
                             "প্রবৃদ্ধি হলো 2023 থেকে 2025-এর মধ্যে ইউনিটের চক্রবৃদ্ধি বার্ষিক হার। ডিসেম্বর সূচক = পণ্যের গড় "
                             "মাসের তুলনায় গড় ডিসেম্বরের ইউনিট (100 = মৌসুমিতা নেই)।"),
        "market_clusters": ("অঞ্চল $" + BS + "times$ চ্যানেল $" + BS + "times$ গ্রাহকের ধরন কোষের বাজার-বিভাগ "
                            f"(K-Means, k={k['segment_k']})", None),
        "cluster_comparison": ("নির্বাচিত ক্লাস্টার-সংখ্যায় ক্লাস্টারিং অ্যালগরিদমগুলোর তুলনা",
                               "উচ্চতর silhouette ও Calinski--Harabasz এবং নিম্নতর Davies--Bouldin মান বেশি পৃথক ক্লাস্টার নির্দেশ করে।"),
        "daily_anomalies": ("দশটি সবচেয়ে অস্বাভাবিক ব্যবসায়িক দিন (মৌসুমি-সমন্বিত)", None),
        "anomaly_detection": ("500,000 অর্ডারে রোপিত 1,000টি লেনদেন-ত্রুটির শনাক্তকরণ",
                              "শেষ চারটি কলাম দেখায় প্রতিটি ধরনের ত্রুটির কত অংশ শনাক্ত হয়েছে। ডেটার একটি অনুলিপিতে প্রতিটি "
                              "ধরনের 250টি ত্রুটি রোপণ করা হয়েছে।"),
        "hyperparameters": ("Expanding-window যাচাইসহ randomised search-এ নির্বাচিত hyper-parameter (চাহিদা পূর্বাভাস)", None),
    }


def build_tables():
    TABLES_BN.mkdir(exist_ok=True)
    for name, (cap, note) in table_texts().items():
        src = C.TEX_TABLES / f"{name}.tex"
        if not src.exists():
            continue
        out = []
        for line in src.read_text(encoding="utf-8").split("\n"):
            if line.startswith(BS + "caption{"):
                line = BS + "caption{" + cap + "}"
            elif line.startswith(BS + "par" + BS + "smallskip"):
                if note is None:
                    continue
                line = (BS + "par" + BS + "smallskip" + BS + "parbox{0.92" + BS + "linewidth}{" + BS + "footnotesize" + BS +
                        "emph{টীকা:} " + re.sub(r"(?<!\\)%", lambda m: BS + "%", note) + "}")
            out.append(line)
        (TABLES_BN / f"{name}.tex").write_text("\n".join(out), encoding="utf-8")


def build_insights():
    e, f, c, k, a = (load_json(n) for n in ["eda", "forecasting", "classification", "clustering", "anomaly"])
    fu = f["future"]
    best_wape = [r for r in f["one_step"] if r["Model"] == f["best_model"]][0]["WAPE"]
    seg = sorted(k["product_segments"], key=lambda s: -s["Growth"])[0]
    mk = sorted(k["market_segments"], key=lambda s: -s["Growth"])[0]
    lo, hi = e["elasticity_ci"]
    items = [
        ("প্রবৃদ্ধি", "কী ঘটেছে",
         f"2020 থেকে 2025 পর্যন্ত নিট বিক্রয় বছরে {e['cagr_sales']:.1%} হারে বেড়েছে, কিন্তু ইউনিট বেড়েছে মাত্র {e['cagr_units']:.1%}; "
         f"গড়ে বার্ষিক প্রবৃদ্ধির {e['avg_price_growth']:.1f} শতাংশ-বিন্দু এসেছে মূল্য ও মিশ্রণ থেকে এবং {e['avg_volume_growth']:.1f} পরিমাণ থেকে।",
         "পরিমাণ-প্রবৃদ্ধিকে আলাদা KPI হিসেবে অনুসরণ করুন; মূল্য-নির্ভর প্রবৃদ্ধি অনির্দিষ্টকাল টেকসই নয়।"),
        ("মূল্য নির্ধারণ", "কেন ঘটেছে",
         f"পণ্য, বছর ও মৌসুম নিয়ন্ত্রণ করার পর চাহিদার নির্ণীত মূল্য-স্থিতিস্থাপকতা {e['elasticity']:+.2f} (95% CI {lo:+.2f} থেকে {hi:+.2f}), "
         f"যা শূন্য থেকে তাৎপর্যপূর্ণভাবে আলাদা নয়। ডিসকাউন্টের সঙ্গে অর্ডারের আকারের কোনো সম্পর্ক নেই "
         f"(Spearman rho = {e['tests']['Discount % vs units/order']['stat']:.3f})।",
         f"ডিসকাউন্ট বাজেট পুনর্বিবেচনা করুন (এই সময়কালে {e['discount_cost'] / 1e6:,.1f} মিলিয়ন টাকা): ডিসকাউন্ট বিক্রির পরিমাণ বাড়ায় বলে মনে হয় না।"),
        ("মৌসুমিতা", "কী ঘটেছে",
         f"{MONTHS_BN[e['peak_month']]} শীর্ষ মাস (সূচক {e['peak_index']:.0f}, গড় মাস = 100) এবং {MONTHS_BN[e['trough_month']]} সবচেয়ে দুর্বল "
         f"({e['trough_index']:.0f})। মৌসুমি প্রভাব পরিসংখ্যানিকভাবে শক্তিশালী (Kruskal-Wallis p < 0.001)।",
         "অক্টোবর থেকে মজুদ ও ডিস্ট্রিবিউটর-সক্ষমতা বাড়ান; ফেব্রুয়ারিতে প্রমোশন ও রক্ষণাবেক্ষণের পরিকল্পনা করুন।"),
        ("পোর্টফোলিও", "কী ঘটেছে",
         f"{e['top_category']} সবচেয়ে বড় ক্যাটাগরি (বিক্রয়ের {e['top_category_share']:.1f}%); 52টি পণ্যের শীর্ষ 10টি আয়ের "
         f"{e['share_top10_products']:.1f}% দেয় এবং {e['abc_counts'].get('A', 0)}টি পণ্য `A' শ্রেণি (বিক্রয়ের 80%) গঠন করে।",
         "প্রথমে A-শ্রেণির পণ্যের প্রাপ্যতা নিশ্চিত করুন; C-শ্রেণির পণ্য যৌক্তিকীকরণ বা পুনঃস্থাপনের জন্য পর্যালোচনা করুন।"),
        ("পোর্টফোলিও", "বিভাজন",
         f"ক্লাস্টারিং {k['product_k']}টি পণ্য-বিভাগ খুঁজে পেয়েছে; `{seg['segment']}' সবচেয়ে দ্রুত বাড়ছে "
         f"(বছরে ইউনিট {seg['Growth']:+.1f}%, বিক্রয়ের {seg['Share']:.1f}%)।",
         f"`{seg['segment']}'-এর জন্য তাকের জায়গা ও বিপণনে অগ্রাধিকার দিন।"),
        ("বাজার", "কী ঘটেছে",
         f"ঢাকা একাই বিক্রয়ের {e['top_region_share']:.1f}% এবং ঢাকা ও চট্টগ্রাম মিলে {e['dhaka_ctg_share']:.1f}% আনে। General Trade সবচেয়ে "
         f"বড় চ্যানেল ({e['channel_share']['General Trade']:.1f}%), আর E-commerce {e['channel_share']['E-commerce']:.1f}%।",
         "দ্বিতীয় সারির শহর গড়ে তুলে এবং ছোট ভিত্তি থেকে বাড়তে থাকা E-commerce প্রসারিত করে কেন্দ্রীভবনের ঝুঁকি কমান।"),
        ("বাজার", "বিভাজন",
         f"`{mk['segment']}' বিভাগটি ({int(mk['Cells'])}টি অঞ্চল-চ্যানেল-গ্রাহক কোষ, মূলত {mk['Regions']}) বছরে {mk['Growth']:+.1f}% হারে "
         f"বাড়ছে, যা {k['segment_k']}টি বাজার-বিভাগের মধ্যে দ্রুততম।",
         "মাঠ-পর্যায়ের বিক্রয় ও ডিজিটাল বিপণনের প্রচেষ্টা এই উদীয়মান কোষগুলোতে বরাদ্দ করুন।"),
        ("অর্ডার", "পূর্বাভাস",
         f"একটি অর্ডার বড় হবে কি না (>= {C.LARGE_ORDER_UNITS} ইউনিট) তা মাঝারি মাত্রায় পূর্বাভাসযোগ্য (সেরা ROC-AUC "
         f"{c['results'][0]['ROC-AUC']:.3f}, {c['results'][0]['Model']}); চ্যানেলই সবচেয়ে শক্তিশালী চালিকাশক্তি: ডিস্ট্রিবিউটর অর্ডারের "
         f"{c['large_rate_by_channel']['Distributor']:.0%} বড়, অথচ অনলাইনে {c['large_rate_by_channel']['E-commerce']:.0%}।",
         "যানবাহন ও মজুদ বণ্টনে বড়-অর্ডার স্কোর ব্যবহার করুন, মূলত চ্যানেল-স্তরে।"),
        ("পূর্বাভাস", "সামনে কী ঘটবে",
         f"{f['best_model']} মডেল পরের মাসের পণ্য-চাহিদা {best_wape:.1f}% weighted ত্রুটিতে পূর্বাভাস করে, যা seasonal-naive নিয়মের চেয়ে "
         f"{f['best_vs_seasonal_naive_pct']:.0f}% কম RMSE। অক্টোবর 2026 থেকে সেপ্টেম্বর 2027-এর নিট বিক্রয়ের পূর্বাভাস "
         f"{fu['net_sales'] / 1e6:,.1f} মিলিয়ন টাকা (আগের 12 মাসের চেয়ে {fu['growth_sales_pct']:+.1f}%; 80% ব্যবধান "
         f"{fu['net_sales_lo'] / 1e6:,.1f} -- {fu['net_sales_hi'] / 1e6:,.1f} মিলিয়ন টাকা)।",
         "মাসিক উৎপাদন পরিকল্পনায় পণ্য-স্তরের পূর্বাভাস এবং নিরাপত্তা-মজুদে পূর্বাভাস-ব্যবধান ব্যবহার করুন।"),
        ("ঝুঁকি", "অস্বাভাবিকতা",
         f"{a['daily']['days']:,}টি ব্যবসায়িক দিনের {a['daily']['flag_any']}টি অস্বাভাবিক হিসেবে চিহ্নিত। একটি নিয়ন্ত্রিত পরীক্ষায় নিয়ম ও "
         f"Isolation Forest মিলে রোপিত 1,000টি ডেটা-ত্রুটির {a['transaction_detection'][3]['Recall']:.0%} ধরেছে।",
         "প্রতিদিন অ্যানোমালি যাচাই চালান এবং মাস-শেষের প্রতিবেদনের আগে চিহ্নিত দিন ও অর্ডার তদন্ত করুন।"),
    ]
    esc = lambda s: (s.replace("&", BS + "&").replace("%", BS + "%").replace("_", BS + "_")
                     .replace(">=", "$" + BS + "geq$"))
    lines = [BS + "begin{enumerate}[leftmargin=*,itemsep=4pt]"]
    for area, kind, text, action in items:
        lines.append(BS + "item " + BS + "textbf{" + esc(area) + " (" + esc(kind) + ")।} " + esc(text) + " " +
                     BS + "emph{সুপারিশ:} " + esc(action))
    lines.append(BS + "end{enumerate}")
    (C.TEX_GENERATED / "insights_bn.tex").write_text("\n".join(lines), encoding="utf-8")


def run():
    build_tables()
    build_insights()
    print("[report_bn] Bangla tables -> report/tables_bn, insights -> report/generated/insights_bn.tex")


if __name__ == "__main__":
    run()
