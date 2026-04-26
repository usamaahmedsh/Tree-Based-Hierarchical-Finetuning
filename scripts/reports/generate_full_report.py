"""
generate_full_report.py
=======================
Three-way comparison: original 335k → HPC noised → relabeled
Plus fine-tuning readiness analysis.
Output: outputs/reports/full_analysis_report.html
"""

import json, statistics, re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ORIGINAL_PATH  = Path("data/chunks/chunks.jsonl")
NOISED_PATH    = Path("noised_chunks_hpc.jsonl")
RELABELED_PATH = Path("noised_chunks_hpc_relabeled.jsonl")
OUT            = Path("outputs/reports/full_analysis_report.html")

HIGH_T, MED_T = 1.5, 0.5


# ── loaders ───────────────────────────────────────────────────────────────────

def load_original():
    rows = []
    with open(ORIGINAL_PATH) as f:
        for line in f:
            d = json.loads(line)
            rows.append({"topic": d.get("topic_name","?"),
                         "token_count": d.get("token_count", 0),
                         "text_len": len(d.get("text",""))})
    return rows


def load_noised(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ot, nt, ins = d.get("original_text",""), d.get("noised_text",""), d.get("instruction","")
            rows.append({
                "topic":           d.get("topic_name","?"),
                "leaf_id":         d.get("leaf_id",""),
                "noise_tier":      d.get("noise_tier","?"),
                "noise_tier_corr": d.get("noise_tier_corrected", d.get("noise_tier","?")),
                "majority_score":  d.get("majority_score", 0.0),
                "score_corr":      d.get("majority_score_corrected", d.get("majority_score", 0.0)),
                "orig_len":        len(ot),
                "noised_len":      len(nt),
                "instr_len":       len(ins),
                "instr_words":     len(ins.split()),
                "has_instruction": len(ins.strip()) > 0,
                "noised_text":     nt,
                "instruction":     ins,
                # fine-tuning token estimate (~4 chars/token)
                "ft_input_tokens":  len(ins)  // 4,
                "ft_output_tokens": len(nt)   // 4,
                "ft_total_tokens":  (len(ins) + len(nt)) // 4,
            })
    return rows


# ── helpers ───────────────────────────────────────────────────────────────────

def pct(a, b):   return round(100*a/b, 1) if b else 0
def med(vals):   return round(statistics.median(vals), 1) if vals else 0
def avg(vals):   return round(statistics.mean(vals), 1)   if vals else 0
def topic_dist(rows, key="topic"): return dict(Counter(r[key] for r in rows).most_common())


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_tier_comparison(hpc, relabeled):
    hpc_tiers = Counter(r["noise_tier"]      for r in hpc)
    rel_tiers = Counter(r["noise_tier_corr"] for r in relabeled)
    tiers = ["HIGH","MED","LOW"]
    fig = go.Figure()
    fig.add_bar(name="HPC (original labels)",  x=tiers,
                y=[hpc_tiers.get(t,0) for t in tiers], marker_color="#E45756")
    fig.add_bar(name="Relabeled (corrected)", x=tiers,
                y=[rel_tiers.get(t,0) for t in tiers], marker_color="#4C78A8")
    fig.update_layout(barmode="group", title="Noise Tier: Before vs After Relabeling",
                      template="plotly_white", height=400)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def plot_topic_triple(orig, hpc, relabeled):
    all_topics = sorted(set(r["topic"] for r in orig))
    orig_c = topic_dist(orig)
    hpc_c  = topic_dist(hpc)
    rel_c  = topic_dist(relabeled)
    df = pd.DataFrame({"topic": all_topics,
                       "Original (335k)":  [orig_c.get(t,0) for t in all_topics],
                       "HPC noised":       [hpc_c.get(t,0)  for t in all_topics],
                       "Relabeled":        [rel_c.get(t,0)  for t in all_topics],
                       }).sort_values("Original (335k)", ascending=False)
    fig = go.Figure()
    colors = ["#4C78A8","#F58518","#72B7B2"]
    for col, color in zip(["Original (335k)","HPC noised","Relabeled"], colors):
        fig.add_bar(name=col, x=df["topic"], y=df[col], marker_color=color)
    fig.update_layout(barmode="group", title="Topic Distribution — Three-Way",
                      template="plotly_white", height=480,
                      xaxis_tickangle=-45, legend=dict(orientation="h", y=-0.35))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_text_lengths(orig, hpc):
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("Original text length (chars)",
                        "Noised text length (chars)",
                        "% length change after noising"))
    orig_lens   = [r["text_len"]   for r in orig]
    noised_lens = [r["noised_len"] for r in hpc]
    pcts        = [pct(r["noised_len"]-r["orig_len"], r["orig_len"]) for r in hpc if r["orig_len"]>0]

    fig.add_trace(go.Histogram(x=orig_lens,   nbinsx=80, marker_color="#4C78A8", name="Original"), row=1,col=1)
    fig.add_trace(go.Histogram(x=noised_lens, nbinsx=80, marker_color="#F58518", name="Noised"),   row=1,col=2)
    fig.add_trace(go.Histogram(x=[min(max(p,-100),200) for p in pcts],
                               nbinsx=80, marker_color="#72B7B2", name="% change"),               row=1,col=3)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=3)
    fig.update_layout(template="plotly_white", height=400, showlegend=False,
                      title="Text Length Distributions")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_token_dist(hpc):
    totals = [r["ft_total_tokens"] for r in hpc if r["has_instruction"]]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=[min(t,2048) for t in totals], nbinsx=80,
                               marker_color="#4C78A8", name="Total tokens"))
    for thresh, col, label in [(512,"green","512"), (1024,"orange","1024"), (2048,"red","2048")]:
        fig.add_vline(x=thresh, line_dash="dash", line_color=col,
                      annotation_text=f"{label} tok", annotation_position="top right")
    pct_under_512  = pct(sum(1 for t in totals if t<=512),  len(totals))
    pct_under_1024 = pct(sum(1 for t in totals if t<=1024), len(totals))
    fig.update_layout(title=f"Fine-Tuning Token Length (instruction+noised_text) — "
                            f"{pct_under_512}% ≤512 | {pct_under_1024}% ≤1024",
                      xaxis_title="Estimated tokens", yaxis_title="# samples",
                      template="plotly_white", height=400)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_instruction_len(hpc):
    instrs = [r["instr_words"] for r in hpc if r["has_instruction"]]
    fig = px.histogram(instrs, nbins=60,
                       title="Instruction Length Distribution (words)",
                       labels={"value":"Words in instruction"},
                       color_discrete_sequence=["#54A24B"])
    fig.update_layout(template="plotly_white", height=380)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_tier_by_topic_corrected(relabeled):
    rows = defaultdict(Counter)
    for r in relabeled:
        rows[r["topic"]][r["noise_tier_corr"]] += 1
    data = [{"topic":t, "tier":tier, "count":cnt}
            for t, tiers in rows.items() for tier, cnt in tiers.items()]
    df = pd.DataFrame(data)
    color_map = {"HIGH":"#E45756","MED":"#F58518","LOW":"#72B7B2"}
    fig = px.bar(df, x="topic", y="count", color="tier", barmode="stack",
                 title="Corrected Noise Tier Breakdown by Topic",
                 color_discrete_map=color_map)
    fig.update_layout(template="plotly_white", height=480,
                      xaxis_tickangle=-45, legend=dict(orientation="h", y=-0.35))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def plot_score_corr_dist(relabeled):
    scores = [r["score_corr"] for r in relabeled]
    capped = [min(max(s,-5),30) for s in scores]
    fig = px.histogram(capped, nbins=100,
                       title="Corrected Majority Score Distribution (capped at [-5, 30])",
                       labels={"value":"Corrected score"},
                       color_discrete_sequence=["#B279A2"])
    fig.add_vline(x=HIGH_T, line_dash="dash", line_color="red",   annotation_text="HIGH (1.5)")
    fig.add_vline(x=MED_T,  line_dash="dash", line_color="orange",annotation_text="MED (0.5)")
    fig.update_layout(template="plotly_white", height=400)
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── fine-tuning readiness ────────────────────────────────────────────────────

def ft_readiness(hpc, relabeled):
    total = len(hpc)
    empty_instr  = sum(1 for r in hpc if not r["has_instruction"])
    empty_noised = sum(1 for r in hpc if r["noised_len"] < 50)
    very_short   = sum(1 for r in hpc if r["ft_total_tokens"] < 32)
    over_2k      = sum(1 for r in hpc if r["ft_total_tokens"] > 2048)
    usable       = [r for r in hpc if r["has_instruction"] and r["noised_len"] >= 50]

    tok_totals   = [r["ft_total_tokens"] for r in usable]
    instr_words  = [r["instr_words"]     for r in usable]

    tier_corr    = Counter(r["noise_tier_corr"] for r in relabeled)
    tier_orig    = Counter(r["noise_tier"]      for r in hpc)

    # duplication proxy: count repeated instructions (exact)
    instr_counts = Counter(r["instruction"] for r in hpc if r["has_instruction"])
    exact_dupes  = sum(v-1 for v in instr_counts.values() if v > 1)

    # quality checks
    truncated    = sum(1 for r in hpc if r["noised_text"].endswith(("...", "…")))
    has_encoding = sum(1 for r in hpc if "â€" in r["noised_text"] or "Ã" in r["noised_text"])

    return {
        "total": total,
        "empty_instr": empty_instr,
        "empty_noised": empty_noised,
        "very_short": very_short,
        "over_2k": over_2k,
        "usable": len(usable),
        "avg_total_tokens": avg(tok_totals),
        "med_total_tokens": med(tok_totals),
        "avg_instr_words": avg(instr_words),
        "exact_dupes": exact_dupes,
        "truncated": truncated,
        "has_encoding": has_encoding,
        "tier_orig": dict(tier_orig),
        "tier_corr": dict(tier_corr),
        "pct_under_512":  pct(sum(1 for t in tok_totals if t<=512),  len(tok_totals)),
        "pct_under_1024": pct(sum(1 for t in tok_totals if t<=1024), len(tok_totals)),
    }


def sample_table(hpc, tier, n=5):
    samples = [r for r in hpc if r["noise_tier_corr"] == tier and r["has_instruction"]][:n]
    rows = ""
    for r in samples:
        instr = r["instruction"][:120].replace("<","&lt;").replace(">","&gt;")
        noised = r["noised_text"][:180].replace("<","&lt;").replace(">","&gt;")
        rows += f"<tr><td>{r['topic']}</td><td>{instr}…</td><td>{noised}…</td><td>{r['ft_total_tokens']}</td></tr>"
    return f"""<table class="table table-sm table-bordered" style="font-size:12px">
        <thead class="table-dark"><tr><th>Topic</th><th>Instruction</th><th>Noised text (snippet)</th><th>~Tokens</th></tr></thead>
        <tbody>{rows}</tbody></table>"""


# ── html builder ──────────────────────────────────────────────────────────────

def build_html(orig, hpc, relabeled, plots, ft):
    orig_topics = topic_dist(orig)
    hpc_topics  = topic_dist(hpc)
    missing = sorted(set(orig_topics) - set(hpc_topics))

    cov_rows = ""
    for t, oc in sorted(orig_topics.items(), key=lambda x: -x[1]):
        nc  = hpc_topics.get(t, 0)
        cp  = pct(nc, oc)
        col = "#72B7B2" if cp>=75 else ("#F58518" if cp>=40 else "#E45756")
        cov_rows += f"""<tr><td>{t}</td><td>{oc:,}</td><td>{nc:,}</td>
            <td><div style="background:#eee;width:140px;display:inline-block;border-radius:3px">
            <div style="background:{col};width:{min(cp,100):.0f}%;height:13px;border-radius:3px"></div></div>
            &nbsp;<b>{cp}%</b></td></tr>"""

    usable_pct = pct(ft["usable"], ft["total"])
    ft_verdict = "GOOD" if usable_pct >= 85 and ft["avg_total_tokens"] < 1200 else \
                 "NEEDS FILTERING" if usable_pct >= 70 else "NEEDS WORK"
    verdict_color = {"GOOD":"#198754","NEEDS FILTERING":"#fd7e14","NEEDS WORK":"#dc3545"}[ft_verdict]

    tc_orig = ft["tier_orig"]
    tc_corr = ft["tier_corr"]
    n = ft["total"]

    samples_high = sample_table(relabeled, "HIGH")
    samples_med  = sample_table(relabeled, "MED")
    samples_low  = sample_table(relabeled, "LOW")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Full Pipeline Analysis Report</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body{{background:#f8f9fa;font-family:'Segoe UI',sans-serif;}}
    .wrap{{max-width:1400px;margin:28px auto;background:#fff;padding:36px;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,.07);}}
    h1,h2,h3{{color:#2c3e50;}} hr{{margin:40px 0;}}
    .kpi{{background:#fff;padding:16px 20px;border-radius:8px;border-left:5px solid #007bff;box-shadow:0 2px 6px rgba(0,0,0,.06);}}
    .kpi .num{{font-size:24px;font-weight:700;color:#007bff;}}
    .kpi .lbl{{color:#6c757d;font-size:11px;text-transform:uppercase;letter-spacing:1px;}}
    .chart{{margin:24px 0;padding:14px;border:1px solid #eee;border-radius:8px;}}
    .info{{background:#f1f8ff;border:1px solid #cfe2ff;padding:16px;border-radius:8px;margin:14px 0;}}
    .warn{{background:#fff3cd;border:1px solid #ffc107;padding:14px;border-radius:8px;margin:12px 0;}}
    .good{{background:#d1e7dd;border:1px solid #a3cfbb;padding:14px;border-radius:8px;margin:12px 0;}}
    .bad {{background:#f8d7da;border:1px solid #f1aeb5;padding:14px;border-radius:8px;margin:12px 0;}}
    .verdict{{font-size:22px;font-weight:700;color:{verdict_color};}}
  </style>
</head>
<body><div class="wrap">

  <h1>Full Pipeline Analysis Report</h1>
  <p class="text-muted">Original 335k &nbsp;→&nbsp; HPC noised (250k snapshot) &nbsp;→&nbsp; Relabeled</p>

  <!-- ── SECTION 1: PIPELINE PROGRESS ── -->
  <h2>1. Pipeline Progress</h2>
  <div class="row g-3 mb-4">
    <div class="col-md-3"><div class="kpi"><div class="num">{len(orig):,}</div><div class="lbl">Original chunks</div></div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#F58518"><div class="num" style="color:#F58518">{len(hpc):,}</div><div class="lbl">HPC processed (snapshot)</div></div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#72B7B2"><div class="num" style="color:#72B7B2">{pct(len(hpc),len(orig))}%</div><div class="lbl">Pipeline progress</div></div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#E45756"><div class="num" style="color:#E45756">{len(orig)-len(hpc):,}</div><div class="lbl">Remaining on HPC</div></div></div>
  </div>
  {"" if not missing else f'<div class="warn"><b>Topics not yet in HPC output:</b> {", ".join(missing)}</div>'}

  <!-- ── SECTION 2: NOISE TIER RELABELING ── -->
  <hr>
  <h2>2. Noise Tier — Before vs After Relabeling</h2>
  <div class="info">
    <b>Why relabeling was needed:</b> The original majority scores were computed as a global z-score across all 42,453 leaves.
    The 61 T0 (BERTopic catch-all) leaves have 2,000–23,000 docs each, inflating the global sigma to 235.
    This flattened all named-topic leaves to near-zero scores, making 99.5% of chunks LOW tier regardless of their actual dominance.<br><br>
    <b>Fix:</b> Two-group normalisation — T0 leaves scored within T0 population (σ=5,658), non-T0 leaves scored within non-T0 population (σ=7.03).
    non-T0 HIGH threshold: ≥15 docs/leaf &nbsp;|&nbsp; MED: ≥8 docs/leaf.
  </div>

  <div class="row g-3 mb-4">
    <div class="col-md-2"><div class="kpi" style="border-color:#E45756">
      <div class="num" style="color:#E45756">{tc_orig.get('HIGH',0):,}</div><div class="lbl">HIGH (original)</div></div></div>
    <div class="col-md-2"><div class="kpi" style="border-color:#F58518">
      <div class="num" style="color:#F58518">{tc_orig.get('MED',0):,}</div><div class="lbl">MED (original)</div></div></div>
    <div class="col-md-2"><div class="kpi" style="border-color:#72B7B2">
      <div class="num" style="color:#72B7B2">{tc_orig.get('LOW',0):,}</div><div class="lbl">LOW (original)</div></div></div>
    <div class="col-md-2"><div class="kpi" style="border-color:#E45756">
      <div class="num" style="color:#E45756">{tc_corr.get('HIGH',0):,}</div><div class="lbl">HIGH (corrected)</div></div></div>
    <div class="col-md-2"><div class="kpi" style="border-color:#F58518">
      <div class="num" style="color:#F58518">{tc_corr.get('MED',0):,}</div><div class="lbl">MED (corrected)</div></div></div>
    <div class="col-md-2"><div class="kpi" style="border-color:#72B7B2">
      <div class="num" style="color:#72B7B2">{tc_corr.get('LOW',0):,}</div><div class="lbl">LOW (corrected)</div></div></div>
  </div>

  <div class="chart">{plots["tier_compare"]}</div>
  <div class="chart">{plots["tier_by_topic"]}</div>
  <div class="chart">{plots["score_corr"]}</div>

  <!-- ── SECTION 3: TOPIC DISTRIBUTION ── -->
  <hr>
  <h2>3. Topic Distribution — Three-Way Comparison</h2>
  <div class="chart">{plots["topic_triple"]}</div>

  <h3>Coverage Table</h3>
  <div class="table-responsive mb-4">
    <table class="table table-sm table-striped table-hover">
      <thead class="table-dark"><tr><th>Topic</th><th>Original</th><th>HPC Noised</th><th>Coverage</th></tr></thead>
      <tbody>{cov_rows}</tbody>
    </table>
  </div>

  <!-- ── SECTION 4: TEXT QUALITY ── -->
  <hr>
  <h2>4. Text Quality</h2>
  <div class="chart">{plots["text_lengths"]}</div>
  <div class="row g-3 mb-4">
    <div class="col-md-4"><div class="info">
      <b>Original corpus</b><br>
      Mean: {avg([r["text_len"] for r in orig]):,} chars &nbsp;|&nbsp; Median: {med([r["text_len"] for r in orig]):,}<br>
      Token mean: {avg([r["token_count"] for r in orig]):,} &nbsp;|&nbsp; Median: {med([r["token_count"] for r in orig]):,}
    </div></div>
    <div class="col-md-4"><div class="info">
      <b>Original text (in noised 250k)</b><br>
      Mean: {avg([r["orig_len"] for r in hpc]):,} chars &nbsp;|&nbsp; Median: {med([r["orig_len"] for r in hpc]):,}
    </div></div>
    <div class="col-md-4"><div class="info">
      <b>Noised text</b><br>
      Mean: {avg([r["noised_len"] for r in hpc]):,} chars &nbsp;|&nbsp; Median: {med([r["noised_len"] for r in hpc]):,}<br>
      Net change: {avg([r["noised_len"]-r["orig_len"] for r in hpc]):+,.0f} chars avg
    </div></div>
  </div>
  {"" if ft["truncated"] < 1000 else f'<div class="warn">⚠ {ft["truncated"]:,} noised texts end with "…" — possible truncation during LLaMA generation.</div>'}
  {"" if ft["has_encoding"] < 100 else f'<div class="warn">⚠ {ft["has_encoding"]:,} noised texts contain encoding artefacts (â€, Ã etc.) — check UTF-8 handling in stage3.</div>'}

  <!-- ── SECTION 5: FINE-TUNING READINESS ── -->
  <hr>
  <h2>5. Fine-Tuning Readiness</h2>
  <div class="mb-3">Overall verdict: <span class="verdict">{ft_verdict}</span></div>

  <div class="row g-3 mb-4">
    <div class="col-md-3"><div class="kpi"><div class="num">{ft["usable"]:,}</div><div class="lbl">Usable training pairs ({usable_pct}%)</div></div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#E45756">
      <div class="num" style="color:#E45756">{ft["empty_instr"]:,}</div><div class="lbl">Empty instructions ({pct(ft["empty_instr"],ft["total"])}%) — must filter</div></div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#F58518">
      <div class="num" style="color:#F58518">{ft["over_2k"]:,}</div><div class="lbl">Samples >2048 tokens ({pct(ft["over_2k"],ft["total"])}%) — check context window</div></div></div>
    <div class="col-md-3"><div class="kpi" style="border-color:#72B7B2">
      <div class="num" style="color:#72B7B2">{ft["avg_total_tokens"]:,}</div><div class="lbl">Avg tokens per sample (instruction+output)</div></div></div>
  </div>

  <div class="chart">{plots["token_dist"]}</div>
  <div class="chart">{plots["instr_len"]}</div>

  <div class="row g-3 mb-4">
    <div class="col-md-6">
      <div class="{"good" if ft["pct_under_512"]>=60 else "warn"}">
        <b>{ft["pct_under_512"]}%</b> of usable samples fit within 512 tokens (instruction + noised text combined).
      </div>
      <div class="{"good" if ft["pct_under_1024"]>=85 else "warn"}">
        <b>{ft["pct_under_1024"]}%</b> fit within 1024 tokens — {"suitable for Phi-3-mini / Qwen2.5 default context." if ft["pct_under_1024"]>=85 else "some samples will be truncated at 1024-token context."}
      </div>
    </div>
    <div class="col-md-6">
      <div class="{"warn" if ft["exact_dupes"]>1000 else "good"}">
        <b>Exact duplicate instructions: {ft["exact_dupes"]:,}</b>
        {"— consider deduplicating before fine-tuning." if ft["exact_dupes"]>1000 else " — acceptable level."}
      </div>
      <div class="info">
        Average instruction length: <b>{ft["avg_instr_words"]} words</b>.
        {"Good — instructions are substantive." if ft["avg_instr_words"] >= 12 else "Short — may not provide enough signal for instruction following."}
      </div>
    </div>
  </div>

  <h3>Checklist before fine-tuning</h3>
  <table class="table table-sm table-bordered mb-4">
    <thead class="table-dark"><tr><th>Check</th><th>Status</th><th>Action</th></tr></thead>
    <tbody>
      <tr><td>Empty instructions ({pct(ft["empty_instr"],ft["total"])}%)</td>
          <td><span class="badge bg-danger">FILTER</span></td>
          <td>Drop {ft["empty_instr"]:,} rows where <code>instruction == ""</code></td></tr>
      <tr><td>Noise tier labels</td>
          <td><span class="badge bg-warning text-dark">FIXED</span></td>
          <td>Use <code>noise_tier_corrected</code> field from relabeled file</td></tr>
      <tr><td>Token length >2048 ({pct(ft["over_2k"],ft["total"])}%)</td>
          <td><span class="badge bg-{"warning text-dark" if ft["over_2k"]>5000 else "success"}">{"TRUNCATE" if ft["over_2k"]>5000 else "OK"}</span></td>
          <td>{"Truncate or drop samples exceeding your model context window" if ft["over_2k"]>5000 else "Minimal — no action needed"}</td></tr>
      <tr><td>Noised text too short (<50 chars) ({pct(ft["empty_noised"],ft["total"])}%)</td>
          <td><span class="badge bg-{"danger" if ft["empty_noised"]>500 else "success"}">{"FILTER" if ft["empty_noised"]>500 else "OK"}</span></td>
          <td>{"Drop degenerate outputs" if ft["empty_noised"]>500 else "Minimal — no action needed"}</td></tr>
      <tr><td>Exact duplicate instructions ({ft["exact_dupes"]:,})</td>
          <td><span class="badge bg-{"warning text-dark" if ft["exact_dupes"]>1000 else "success"}">{"DEDUP" if ft["exact_dupes"]>1000 else "OK"}</span></td>
          <td>{"Deduplicate on instruction field" if ft["exact_dupes"]>1000 else "Acceptable"}</td></tr>
      <tr><td>Training format</td>
          <td><span class="badge bg-success">READY</span></td>
          <td>Use <code>instruction</code> → input, <code>noised_text</code> → output</td></tr>
      <tr><td>Tier-weighted sampling</td>
          <td><span class="badge bg-info">RECOMMENDED</span></td>
          <td>Oversample HIGH/MED; LOW can be subsampled 50% to improve balance</td></tr>
      <tr><td>Remaining 85k chunks</td>
          <td><span class="badge bg-warning text-dark">PENDING</span></td>
          <td>Run <code>relabel_noise_tiers.py</code> on full 335k file once HPC finishes</td></tr>
    </tbody>
  </table>

  <h3>Sample Training Pairs by Tier</h3>
  <h5>HIGH tier samples (corrected)</h5>
  {samples_high}
  <h5 class="mt-4">MED tier samples (corrected)</h5>
  {samples_med}
  <h5 class="mt-4">LOW tier samples</h5>
  {samples_low}

</div></body></html>"""


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading original …");  orig     = load_original()
    print("Loading noised …");    hpc      = load_noised(NOISED_PATH)
    print("Loading relabeled …"); relabeled = load_noised(RELABELED_PATH)
    print("Computing FT stats …"); ft       = ft_readiness(hpc, relabeled)

    print("Building plots …")
    plots = {
        "tier_compare":  plot_tier_comparison(hpc, relabeled),
        "topic_triple":  plot_topic_triple(orig, hpc, relabeled),
        "text_lengths":  plot_text_lengths(orig, hpc),
        "token_dist":    plot_token_dist(hpc),
        "instr_len":     plot_instruction_len(hpc),
        "tier_by_topic": plot_tier_by_topic_corrected(relabeled),
        "score_corr":    plot_score_corr_dist(relabeled),
    }

    print("Assembling HTML …")
    html = build_html(orig, hpc, relabeled, plots, ft)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Saved → {OUT}")

    print()
    print("=== FINE-TUNING SUMMARY ===")
    print(f"  Total          : {ft['total']:,}")
    print(f"  Usable         : {ft['usable']:,}  ({pct(ft['usable'],ft['total'])}%)")
    print(f"  Empty instrs   : {ft['empty_instr']:,}")
    print(f"  Avg tokens     : {ft['avg_total_tokens']}")
    print(f"  ≤512 tokens    : {ft['pct_under_512']}%")
    print(f"  ≤1024 tokens   : {ft['pct_under_1024']}%")
    print(f"  Exact dupes    : {ft['exact_dupes']:,}")
    print(f"  Verdict        : {('GOOD' if pct(ft['usable'],ft['total'])>=85 and ft['avg_total_tokens']<1200 else 'NEEDS FILTERING')}")


if __name__ == "__main__":
    main()
