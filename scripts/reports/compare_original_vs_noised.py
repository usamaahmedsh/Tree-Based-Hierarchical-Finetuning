"""
Comparison: original 335k chunks vs noised_chunks_hpc.jsonl (250k HPC snapshot)
Outputs a self-contained HTML report to outputs/reports/
"""
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import statistics

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ORIGINAL_PATH = Path("data/chunks/chunks.jsonl")
NOISED_PATH   = Path("noised_chunks_hpc.jsonl")
OUT_DIR       = Path("outputs/reports")


# ── loaders ──────────────────────────────────────────────────────────────────

def load_original():
    records = []
    with open(ORIGINAL_PATH) as f:
        for line in f:
            d = json.loads(line)
            records.append({
                "topic": d.get("topic_name", "Unknown"),
                "token_count": d.get("token_count", 0),
                "text_len": len(d.get("text", "")),
            })
    return records


def load_noised():
    records = []
    with open(NOISED_PATH) as f:
        for line in f:
            d = json.loads(line)
            records.append({
                "topic":          d.get("topic_name", "Unknown"),
                "noise_tier":     d.get("noise_tier", "?"),
                "majority_score": d.get("majority_score", 0.0),
                "orig_len":       len(d.get("original_text", "")),
                "noised_len":     len(d.get("noised_text", "")),
                "instr_len":      len(d.get("instruction", "")),
                "leaf_id":        d.get("leaf_id", ""),
                "label_path":     d.get("label_path", ""),
            })
    return records


# ── analysis helpers ──────────────────────────────────────────────────────────

def topic_counts(records, key="topic"):
    c = Counter(r[key] for r in records)
    return dict(sorted(c.items(), key=lambda x: -x[1]))


def safe_pct(a, b):
    return round(100 * a / b, 1) if b else 0


def text_length_stats(values):
    if not values:
        return {}
    return {
        "min":    min(values),
        "max":    max(values),
        "mean":   round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "stdev":  round(statistics.stdev(values), 1) if len(values) > 1 else 0,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_topic_comparison(orig_counts, noised_counts):
    all_topics = sorted(set(orig_counts) | set(noised_counts))
    df = pd.DataFrame({
        "topic":    all_topics,
        "original": [orig_counts.get(t, 0) for t in all_topics],
        "noised":   [noised_counts.get(t, 0) for t in all_topics],
    }).sort_values("original", ascending=False)

    fig = go.Figure()
    fig.add_bar(name="Original (335k)", x=df["topic"], y=df["original"], marker_color="#4C78A8")
    fig.add_bar(name="Noised HPC (250k)", x=df["topic"], y=df["noised"],  marker_color="#F58518", opacity=0.85)
    fig.update_layout(
        barmode="group",
        title="Topic Distribution: Original vs Noised",
        xaxis_title="Topic", yaxis_title="Chunk Count",
        template="plotly_white", height=500,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def plot_noise_tier_donut(noised_records):
    counts = Counter(r["noise_tier"] for r in noised_records)
    df = pd.DataFrame({"tier": list(counts.keys()), "count": list(counts.values())})
    color_map = {"HIGH": "#E45756", "MED": "#F58518", "LOW": "#72B7B2"}
    fig = px.pie(df, values="count", names="tier", hole=0.45,
                 title="Noise Tier Distribution (HPC 250k snapshot)",
                 color="tier", color_discrete_map=color_map)
    fig.update_layout(template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_text_length_change(noised_records):
    df = pd.DataFrame({
        "original_len": [r["orig_len"]  for r in noised_records],
        "noised_len":   [r["noised_len"] for r in noised_records],
    })
    df["delta"] = df["noised_len"] - df["original_len"]
    df["pct_change"] = 100 * df["delta"] / df["original_len"].replace(0, 1)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Original vs Noised Text Length (chars)", "% Length Change Distribution"))

    fig.add_trace(go.Histogram(x=df["original_len"], name="Original", marker_color="#4C78A8",
                               opacity=0.7, nbinsx=80), row=1, col=1)
    fig.add_trace(go.Histogram(x=df["noised_len"],   name="Noised",   marker_color="#F58518",
                               opacity=0.7, nbinsx=80), row=1, col=1)
    fig.add_trace(go.Histogram(x=df["pct_change"].clip(-100, 200), name="% Change",
                               marker_color="#72B7B2", nbinsx=80), row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(barmode="overlay", template="plotly_white", height=420,
                      title="Text Length Before vs After Noising")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_majority_score_dist(noised_records):
    scores = [r["majority_score"] for r in noised_records]
    fig = px.histogram(scores, nbins=120,
                       title="Majority Score Distribution (noised chunks)",
                       labels={"value": "Majority Score", "count": "# Chunks"},
                       color_discrete_sequence=["#4C78A8"])
    fig.add_vline(x=1.5,  line_dash="dash", line_color="red",    annotation_text="HIGH (1.5)")
    fig.add_vline(x=0.5,  line_dash="dash", line_color="orange",  annotation_text="MED (0.5)")
    fig.update_layout(template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_coverage(orig_counts, noised_counts):
    rows = []
    for t, oc in orig_counts.items():
        nc = noised_counts.get(t, 0)
        rows.append({"topic": t, "original": oc, "noised": nc,
                     "coverage_pct": safe_pct(nc, oc)})
    df = pd.DataFrame(rows).sort_values("coverage_pct")
    fig = px.bar(df, x="coverage_pct", y="topic", orientation="h",
                 title="HPC Coverage per Topic (% of original chunks processed)",
                 labels={"coverage_pct": "% Coverage", "topic": "Topic"},
                 color="coverage_pct",
                 color_continuous_scale=["#E45756", "#F58518", "#72B7B2"],
                 range_color=[0, 100])
    fig.add_vline(x=100, line_dash="dash", line_color="green")
    fig.update_layout(template="plotly_white", height=650)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_tier_by_topic(noised_records):
    rows = defaultdict(lambda: Counter())
    for r in noised_records:
        rows[r["topic"]][r["noise_tier"]] += 1
    data = []
    for topic, tiers in rows.items():
        for tier, cnt in tiers.items():
            data.append({"topic": topic, "tier": tier, "count": cnt})
    df = pd.DataFrame(data)
    color_map = {"HIGH": "#E45756", "MED": "#F58518", "LOW": "#72B7B2"}
    fig = px.bar(df, x="topic", y="count", color="tier",
                 title="Noise Tier Breakdown by Topic",
                 barmode="stack",
                 color_discrete_map=color_map)
    fig.update_layout(template="plotly_white", height=480,
                      xaxis_tickangle=-45,
                      legend=dict(orientation="h", y=-0.35))
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── html assembly ─────────────────────────────────────────────────────────────

def build_html(orig, noised, plots, stats):
    orig_topic  = topic_counts(orig,   key="topic")
    noised_topic = topic_counts(noised, key="topic")

    tier_counts = Counter(r["noise_tier"] for r in noised)
    total_orig   = len(orig)
    total_noised = len(noised)

    coverage_rows = ""
    for t, oc in sorted(orig_topic.items(), key=lambda x: -x[1]):
        nc  = noised_topic.get(t, 0)
        pct = safe_pct(nc, oc)
        bar_color = "#72B7B2" if pct >= 75 else ("#F58518" if pct >= 40 else "#E45756")
        coverage_rows += f"""
        <tr>
          <td>{t}</td>
          <td>{oc:,}</td>
          <td>{nc:,}</td>
          <td>
            <div style="background:#eee;border-radius:4px;width:160px;display:inline-block">
              <div style="background:{bar_color};width:{min(pct,100):.0f}%;height:14px;border-radius:4px"></div>
            </div>
            &nbsp;<b>{pct}%</b>
          </td>
        </tr>"""

    missing_topics = [t for t in orig_topic if t not in noised_topic]

    orig_len_stats   = text_length_stats([r["text_len"]   for r in orig])
    noised_len_stats = text_length_stats([r["noised_len"] for r in noised])
    orig_in_noised   = text_length_stats([r["orig_len"]   for r in noised])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Original vs Noised Comparison Report</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {{ background:#f8f9fa; font-family:'Segoe UI',sans-serif; }}
    .wrap {{ max-width:1400px; margin:30px auto; background:#fff; padding:36px; border-radius:12px; box-shadow:0 4px 16px rgba(0,0,0,.07); }}
    h1,h2,h3 {{ color:#2c3e50; }}
    .kpi {{ background:#fff; padding:18px 22px; border-radius:8px; border-left:5px solid #007bff; box-shadow:0 2px 6px rgba(0,0,0,.06); }}
    .kpi .num {{ font-size:26px; font-weight:700; color:#007bff; }}
    .kpi .lbl {{ color:#6c757d; font-size:12px; text-transform:uppercase; letter-spacing:1px; }}
    .chart {{ margin:28px 0; padding:16px; border:1px solid #eee; border-radius:8px; }}
    .insight {{ background:#f1f8ff; border:1px solid #cfe2ff; padding:18px; border-radius:8px; margin:16px 0; }}
    table {{ font-size:13px; }}
    .warn {{ background:#fff3cd; border:1px solid #ffc107; padding:14px; border-radius:8px; }}
  </style>
</head>
<body><div class="wrap">
  <h1 class="mb-1">Original vs Noised — Comparison Report</h1>
  <p class="text-muted mb-4">Original 335k corpus &nbsp;↔&nbsp; HPC noise injection snapshot (250k processed so far)</p>

  <!-- KPIs -->
  <div class="row g-3 mb-4">
    <div class="col-md-3"><div class="kpi">
      <div class="num">{total_orig:,}</div><div class="lbl">Original Chunks</div>
    </div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#F58518">
      <div class="num" style="color:#F58518">{total_noised:,}</div><div class="lbl">Noised Chunks (HPC snapshot)</div>
    </div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#72B7B2">
      <div class="num" style="color:#72B7B2">{safe_pct(total_noised, total_orig)}%</div><div class="lbl">Overall Pipeline Progress</div>
    </div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#E45756">
      <div class="num" style="color:#E45756">{total_orig - total_noised:,}</div><div class="lbl">Chunks Remaining</div>
    </div></div>
  </div>
  <div class="row g-3 mb-5">
    <div class="col-md-3"><div class="kpi" style="border-color:#E45756">
      <div class="num" style="color:#E45756">{tier_counts.get('HIGH',0):,}</div><div class="lbl">HIGH Noise Chunks</div>
    </div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#F58518">
      <div class="num" style="color:#F58518">{tier_counts.get('MED',0):,}</div><div class="lbl">MED Noise Chunks</div>
    </div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#72B7B2">
      <div class="num" style="color:#72B7B2">{tier_counts.get('LOW',0):,}</div><div class="lbl">LOW Noise Chunks</div>
    </div></div>
    <div class="col-md-3"><div class="kpi">
      <div class="num">{len(noised_topic)}</div><div class="lbl">Topics Covered So Far</div>
    </div></div>
  </div>

  {'<div class="warn mb-4"><b>⚠ Missing Topics:</b> The following topics from the original corpus have not yet appeared in the HPC output: <code>' + ", ".join(missing_topics) + "</code></div>" if missing_topics else ""}

  <hr class="my-4">
  <h2>1. Topic Distribution Comparison</h2>
  <div class="chart">{plots['topic_bar']}</div>

  <h2>2. HPC Coverage per Topic</h2>
  <div class="insight">How much of each topic's original chunks have been processed so far. Green = done, red = barely started.</div>
  <div class="chart">{plots['coverage']}</div>

  <h2>3. Coverage Table</h2>
  <div class="table-responsive mb-5">
    <table class="table table-sm table-striped table-hover">
      <thead class="table-dark"><tr><th>Topic</th><th>Original</th><th>Noised</th><th>Coverage</th></tr></thead>
      <tbody>{coverage_rows}</tbody>
    </table>
  </div>

  <hr class="my-4">
  <h2>4. Noise Tier Distribution</h2>
  <div class="row">
    <div class="col-md-5 chart">{plots['tier_donut']}</div>
    <div class="col-md-7 chart">{plots['tier_by_topic']}</div>
  </div>
  <div class="insight">
    <b>Note:</b> {safe_pct(tier_counts.get('LOW',0), total_noised)}% of chunks are LOW noise.
    This is expected if most chunks fall below the majority score threshold of 0.5.
    Verify that <code>majority_score</code> values are being correctly passed from the tree to the Stage 3 script — if most scores are near 0, the z-score normalisation may need rechecking on the full 335k population.
  </div>

  <hr class="my-4">
  <h2>5. Majority Score Distribution</h2>
  <div class="chart">{plots['score_dist']}</div>

  <hr class="my-4">
  <h2>6. Text Length: Before vs After Noising</h2>
  <div class="chart">{plots['len_change']}</div>
  <div class="row g-3 mb-4">
    <div class="col-md-4">
      <div class="insight"><b>Original corpus (all 335k)</b><br>
        Mean: {orig_len_stats['mean']:,} chars &nbsp;|&nbsp; Median: {orig_len_stats['median']:,}<br>
        Min: {orig_len_stats['min']:,} &nbsp;|&nbsp; Max: {orig_len_stats['max']:,}
      </div>
    </div>
    <div class="col-md-4">
      <div class="insight"><b>Original text (within noised 250k)</b><br>
        Mean: {orig_in_noised['mean']:,} chars &nbsp;|&nbsp; Median: {orig_in_noised['median']:,}<br>
        Min: {orig_in_noised['min']:,} &nbsp;|&nbsp; Max: {orig_in_noised['max']:,}
      </div>
    </div>
    <div class="col-md-4">
      <div class="insight"><b>Noised text (250k)</b><br>
        Mean: {noised_len_stats['mean']:,} chars &nbsp;|&nbsp; Median: {noised_len_stats['median']:,}<br>
        Min: {noised_len_stats['min']:,} &nbsp;|&nbsp; Max: {noised_len_stats['max']:,}
      </div>
    </div>
  </div>

</div></body></html>"""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    logging.info("Loading original chunks …")
    orig = load_original()

    logging.info("Loading noised HPC chunks …")
    noised = load_noised()

    logging.info("Building plots …")
    orig_topic   = topic_counts(orig,   key="topic")
    noised_topic = topic_counts(noised, key="topic")

    plots = {
        "topic_bar":   plot_topic_comparison(orig_topic, noised_topic),
        "tier_donut":  plot_noise_tier_donut(noised),
        "len_change":  plot_text_length_change(noised),
        "score_dist":  plot_majority_score_dist(noised),
        "coverage":    plot_coverage(orig_topic, noised_topic),
        "tier_by_topic": plot_tier_by_topic(noised),
    }

    stats = {}
    html = build_html(orig, noised, plots, stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "comparison_original_vs_noised.html"
    out_path.write_text(html, encoding="utf-8")
    logging.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
