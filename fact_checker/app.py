# =============================================================================
# app.py
# Antarmuka web interaktif (Gradio) untuk Indonesian News Fact-Checker.
#
# Jalankan:  python -m fact_checker app            (http://127.0.0.1:7860)
#            python -m fact_checker app --share    (link publik sementara)
# =============================================================================

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Optional

from fact_checker.aggregation import verdict_explanation
from fact_checker.config import (
    PROJECT_ROOT,
    VERDICT_DISPLAY,
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    Config,
)
from fact_checker.explainability import highlight_html, plot_token_importance
from fact_checker.pipeline import FactChecker
from fact_checker.schemas import FactCheckResult

logger = logging.getLogger(__name__)

EXAMPLES = [
    "Presiden Jokowi memerintahkan Wakil Presiden Ma'ruf Amin meninjau lokasi kebakaran Depo Pertamina Plumpang.",
    "Ketua Panpel Arema FC Abdul Haris divonis bebas dalam kasus Tragedi Kanjuruhan.",
    "FIFA mencabut status Indonesia sebagai tuan rumah Piala Dunia U-20 2023.",
    "Pemerintah menetapkan 1 Ramadan 1444 H jatuh pada hari Kamis, 23 Maret 2023.",
    "Bencana longsor di Pulau Serasan tidak menimbulkan korban jiwa.",
    "Timnas sepak bola Indonesia dipastikan menjuarai Piala Dunia 2030.",
]

VERDICT_COLOR = {
    VERDICT_SUPPORTED: ("16, 185, 129", "✔"),      # Emerald Green
    VERDICT_REFUTED: ("239, 68, 68", "✘"),         # Rose Red
    VERDICT_NEI: ("245, 158, 11", "?"),            # Amber
}
NLI_COLOR = {
    "ENTAILMENT": "16, 185, 129",
    "NEUTRAL": "245, 158, 11",
    "CONTRADICTION": "239, 68, 68",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --fc-font: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --fc-font-mono: 'JetBrains Mono', monospace;
}

body, .gradio-container {
  font-family: var(--fc-font) !important;
}

/* Header Banner */
.fc-header {
  position: relative;
  overflow: hidden;
  padding: 30px 34px;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.90) 45%, rgba(30, 58, 138, 0.85) 100%);
  border: 1px solid rgba(255, 255, 255, 0.14);
  box-shadow: 0 16px 40px -10px rgba(15, 23, 42, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.06);
  color: #ffffff;
  margin-bottom: 16px;
}
.fc-header::before {
  content: "";
  position: absolute;
  top: -60px;
  right: -60px;
  width: 240px;
  height: 240px;
  background: radial-gradient(circle, rgba(59, 130, 246, 0.35) 0%, rgba(147, 51, 234, 0.15) 60%, transparent 80%);
  border-radius: 50%;
  pointer-events: none;
}
.fc-header h1 {
  margin: 0 0 8px;
  font-size: 2.1em;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.fc-header p {
  margin: 0;
  color: #cbd5e1;
  font-size: 1.05em;
  line-height: 1.55;
  max-width: 900px;
}
.fc-chips {
  margin-top: 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.fc-chip {
  font-size: 0.82em;
  font-weight: 600;
  padding: 5px 13px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(255, 255, 255, 0.08);
  color: #f1f5f9;
  backdrop-filter: blur(8px);
  transition: all 0.2s ease;
}
.fc-chip:hover {
  background: rgba(255, 255, 255, 0.16);
  border-color: rgba(255, 255, 255, 0.3);
}

/* Button & Form Accents */
button.primary, .primary-btn {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #4338ca 100%) !important;
  color: white !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 12px 24px !important;
  box-shadow: 0 6px 20px -3px rgba(37, 99, 235, 0.45) !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button.primary:hover, .primary-btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 28px -3px rgba(37, 99, 235, 0.6) !important;
}

/* Verdict Display Card */
.fc-verdict {
  border-radius: 18px;
  padding: 22px 26px;
  border: 2px solid;
  position: relative;
  overflow: hidden;
  box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.12);
  transition: all 0.3s ease;
}
.fc-verdict-title {
  display: flex;
  align-items: center;
  gap: 10px;
}
.fc-pulse-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
  animation: fc-pulse 1.8s infinite;
}
@keyframes fc-pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7); }
  70% { transform: scale(1.15); box-shadow: 0 0 0 9px rgba(255, 255, 255, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
}
.fc-verdict .big {
  font-size: 1.8em;
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1.2;
}
.fc-verdict .conf {
  font-size: 2.6em;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.02em;
}
.fc-muted {
  opacity: 0.78;
  font-size: 0.92em;
}
.fc-reason-box {
  margin: 14px 0 10px;
  padding: 12px 16px;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.04);
  border-left: 4px solid;
}
.fc-conflict-alert {
  padding: 10px 14px;
  border-radius: 10px;
  background: rgba(245, 158, 11, 0.15);
  border: 1px solid rgba(245, 158, 11, 0.4);
  color: #b45309;
  font-weight: 600;
  margin: 10px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Progress Bars */
.fc-bar {
  height: 10px;
  border-radius: 99px;
  background: rgba(127, 127, 127, 0.16);
  overflow: hidden;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.fc-bar-fill {
  height: 100%;
  border-radius: 99px;
  transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}
.fc-row {
  display: grid;
  grid-template-columns: 180px 1fr 60px;
  gap: 12px;
  align-items: center;
  margin: 8px 0;
  font-size: 0.95em;
}

/* Evidence Cards */
.fc-card {
  border: 1px solid var(--border-color-primary, rgba(226, 232, 240, 0.8));
  border-radius: 16px;
  padding: 18px 22px;
  margin-bottom: 14px;
  background: var(--block-background-fill, #ffffff);
  box-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.05);
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}
.fc-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px -4px rgba(0, 0, 0, 0.1);
}
.fc-card.decisive {
  border: 2px solid #3b82f6 !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25), 0 8px 25px -4px rgba(59, 130, 246, 0.2) !important;
  position: relative;
}
.fc-card.decisive::after {
  content: "👑 EVIDENCE PENENTU";
  position: absolute;
  top: 14px;
  right: 18px;
  font-size: 0.72em;
  font-weight: 800;
  letter-spacing: 0.05em;
  color: #2563eb;
  background: rgba(37, 99, 235, 0.12);
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid rgba(37, 99, 235, 0.3);
}
.fc-card h4 {
  margin: 10px 0 8px;
  font-size: 1.12em;
  font-weight: 700;
  line-height: 1.4;
}
.fc-card a {
  color: #2563eb;
  text-decoration: none;
  transition: color 0.15s ease;
}
.fc-card a:hover {
  text-decoration: underline;
  color: #1d4ed8;
}
.fc-passage {
  line-height: 1.75;
  font-size: 1.02em;
  color: var(--body-text-color, #334155);
  padding: 10px 14px;
  border-radius: 10px;
  background: rgba(127, 127, 127, 0.05);
  border-left: 3px solid #94a3b8;
  margin: 10px 0;
}
.fc-portal-badge {
  display: inline-block;
  font-size: 0.75em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 2px 8px;
  border-radius: 6px;
  background: rgba(59, 130, 246, 0.12);
  color: #2563eb;
}
.fc-badge {
  display: inline-block;
  font-size: 0.78em;
  font-weight: 700;
  padding: 3px 11px;
  border-radius: 999px;
  letter-spacing: 0.02em;
}
.fc-stack {
  display: flex;
  height: 8px;
  border-radius: 99px;
  overflow: hidden;
  margin-top: 8px;
  background: rgba(127, 127, 127, 0.15);
}

/* KPI Summary Metric Cards */
.fc-kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
  margin: 16px 0 22px;
}
.fc-kpi-card {
  padding: 18px 20px;
  border-radius: 16px;
  background: var(--block-background-fill, #ffffff);
  border: 1px solid var(--border-color-primary, rgba(226, 232, 240, 0.8));
  box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.05);
  transition: all 0.25s ease;
}
.fc-kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px -3px rgba(0, 0, 0, 0.1);
}
.fc-kpi-val {
  font-size: 2.1em;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
  margin: 4px 0 2px;
}
.fc-kpi-sub {
  font-size: 0.82em;
  opacity: 0.72;
}

/* Saliency Tokens (SHAP) */
.fc-tok {
  display: inline-block;
  padding: 1px 4px;
  margin: 1px 2px;
  border-radius: 4px;
  font-weight: 500;
  transition: transform 0.15s ease;
}
.fc-tok:hover {
  transform: scale(1.08);
}
.fc-tok.pos {
  background: rgba(16, 185, 129, 0.25);
  color: #065f46;
  border-bottom: 2px solid #10b981;
}
.fc-tok.neg {
  background: rgba(239, 68, 68, 0.25);
  color: #991b1b;
  border-bottom: 2px solid #ef4444;
}
.fc-legend {
  display: flex;
  gap: 14px;
  align-items: center;
  font-size: 0.85em;
  margin: 8px 0 12px;
}
.fc-legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.fc-legend-swatch {
  width: 14px;
  height: 14px;
  border-radius: 4px;
  display: inline-block;
}

/* Methodology & Architecture Styles */
.fc-method-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin: 8px 0 20px;
}
.fc-method-card {
  border: 1px solid var(--border-color-primary, rgba(226, 232, 240, 0.8));
  border-radius: 16px;
  padding: 20px 24px;
  background: var(--block-background-fill, #ffffff);
  box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.04);
  transition: all 0.2s ease;
}
.fc-method-card:hover {
  box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.08);
  border-color: rgba(59, 130, 246, 0.4);
}
.fc-method-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.fc-method-badge {
  font-size: 0.76em;
  font-weight: 800;
  padding: 4px 12px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.12);
  color: #2563eb;
  border: 1px solid rgba(59, 130, 246, 0.25);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.fc-method-title {
  margin: 0;
  font-size: 1.18em;
  font-weight: 700;
  color: var(--body-text-color, #0f172a);
}
.fc-math-box {
  background: rgba(15, 23, 42, 0.04);
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 12px;
  padding: 14px 20px;
  margin: 12px 0;
  font-family: var(--fc-font-mono);
  font-size: 1.02em;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
  color: var(--body-text-color, #0f172a);
}
.fc-rule-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin: 14px 0 6px;
}
.fc-rule-card {
  border-radius: 12px;
  padding: 14px 16px;
  border: 1.5px solid;
  background: rgba(127, 127, 127, 0.04);
}
.fc-rule-card.supported {
  border-color: #10b981;
  background: rgba(16, 185, 129, 0.06);
}
.fc-rule-card.refuted {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.06);
}
.fc-rule-card.nei {
  border-color: #f59e0b;
  background: rgba(245, 158, 11, 0.06);
}
.fc-rule-card.gate {
  border-color: #3b82f6;
  background: rgba(59, 130, 246, 0.06);
}
.fc-rule-head {
  font-weight: 700;
  font-size: 0.9em;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.fc-rule-body {
  font-family: var(--fc-font-mono);
  font-size: 0.88em;
  line-height: 1.55;
  color: var(--body-text-color, #1e293b);
}
.fc-alert-box {
  border: 1.5px solid #f59e0b;
  background: rgba(245, 158, 11, 0.08);
  border-radius: 16px;
  padding: 18px 22px;
  margin-top: 10px;
}
.fc-alert-box h4 {
  margin: 0 0 8px;
  font-size: 1.08em;
  color: #b45309;
  display: flex;
  align-items: center;
  gap: 8px;
}
"""


def _bar(value: float, rgb: str) -> str:
    pct = max(0.0, min(100.0, value * 100))
    return (
        f'<div class="fc-bar">'
        f'<div class="fc-bar-fill" style="width:{pct:.1f}%;background:linear-gradient(90deg, rgba({rgb},0.75), rgb({rgb}));"></div>'
        f'</div>'
    )


def render_verdict(result: FactCheckResult) -> str:
    v = result.verdict
    rgb, icon = VERDICT_COLOR[v.label]
    rows = "".join(
        f'<div class="fc-row">'
        f'<span>{VERDICT_DISPLAY[lbl]}</span>'
        f'{_bar(score, VERDICT_COLOR[lbl][0])}'
        f'<b style="text-align:right">{score:.0%}</b>'
        f'</div>'
        for lbl, score in v.label_scores.items()
    )
    conflict = (
        '<div class="fc-conflict-alert">'
        '<span>⚠</span> Bukti saling bertentangan — disarankan memeriksa evidence secara manual.'
        '</div>'
    ) if v.conflicting else ""

    t_total = result.timings.get("total", 0)
    t_ret = result.timings.get("retrieval", 0)
    t_nli = result.timings.get("nli", 0)
    t_shap = result.timings.get("explain_shap", 0)

    timing_chips = f"⏱ <b>{t_total:.1f}s total</b>"
    if t_ret > 0 or t_nli > 0:
        timing_chips += f" (retrieval {t_ret:.2f}s · NLI {t_nli:.2f}s"
        if t_shap > 0:
            timing_chips += f" · SHAP {t_shap:.1f}s"
        timing_chips += ")"

    return f"""
    <div class="fc-verdict" style="border-color:rgb({rgb});background:linear-gradient(145deg, rgba({rgb},0.09) 0%, rgba({rgb},0.03) 100%);">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap">
        <div>
          <div class="fc-muted" style="letter-spacing:0.06em;font-weight:700;font-size:0.78em;text-transform:uppercase">HASIL KEPUTUSAN (VERDICT)</div>
          <div class="fc-verdict-title">
            <span class="fc-pulse-dot" style="background:rgb({rgb});"></span>
            <div class="big" style="color:rgb({rgb});">{icon} {VERDICT_DISPLAY[v.label]}</div>
          </div>
          <div class="fc-muted" style="margin-top:2px;font-weight:500;">Label Teknis: <code>{v.label}</code></div>
        </div>
        <div style="text-align:right">
          <div class="fc-muted" style="letter-spacing:0.06em;font-weight:700;font-size:0.78em;text-transform:uppercase">CONFIDENCE</div>
          <div class="conf" style="color:rgb({rgb});">{v.confidence:.0%}</div>
        </div>
      </div>
      <div class="fc-reason-box" style="border-color:rgb({rgb});">
        <p style="margin:0 0 4px;font-weight:700;font-size:1.02em;">{html.escape(v.reason)}</p>
        <p class="fc-muted" style="margin:0;font-size:0.92em;">{verdict_explanation(v.label)}</p>
      </div>
      {conflict}
      <div style="margin:14px 0 10px;">
        {rows}
      </div>
      <div class="fc-muted" style="margin-top:12px;font-size:0.86em;display:flex;justify-content:space-between;border-top:1px solid rgba({rgb},0.2);padding-top:8px;">
        <span>{timing_chips}</span>
        <span><b>{len(result.evidences)}</b> evidence diverifikasi</span>
      </div>
    </div>"""


def render_evidences(result: FactCheckResult) -> str:
    if not result.evidences:
        return "<div class='fc-card fc-muted'><p>Tidak ada evidence yang ditemukan dalam korpus berita.</p></div>"
    cards = []
    for i, ev in enumerate(result.evidences):
        s = ev.nli_scores
        stack = "".join(
            f'<div title="{lbl}: {s.get(lbl, 0):.1%}" style="width:{s.get(lbl, 0)*100:.1f}%;'
            f'background:rgb({NLI_COLOR[lbl]})"></div>'
            for lbl in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")
        )
        rgb = NLI_COLOR.get(ev.nli_label, "100, 116, 139")
        decisive = (i == result.verdict.decisive_index)
        title = html.escape(ev.title)
        link = f'<a href="{html.escape(ev.url)}" target="_blank" rel="noopener noreferrer">{title} ↗</a>' if ev.url else title
        portal_name = html.escape(ev.source.upper() if ev.source else "PORTAL")
        date_str = html.escape(ev.date[:10] if ev.date else "-")

        cards.append(f"""
        <div class="fc-card {'decisive' if decisive else ''}">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
            <div style="display:flex;align-items:center;gap:8px;">
              <span class="fc-portal-badge">{portal_name}</span>
              <span class="fc-muted" style="font-size:0.86em;">#{i+1} · {date_str}</span>
            </div>
            <span class="fc-badge" style="background:rgba({rgb},0.15);color:rgb({rgb});border:1px solid rgba({rgb},0.3);">
              {ev.nli_label} {ev.nli_confidence:.0%}
            </span>
          </div>
          <h4>{link}</h4>
          <div class="fc-passage">"{html.escape(ev.passage)}"</div>
          <div class="fc-row" style="grid-template-columns:120px 1fr 60px;margin-top:10px;font-size:0.88em;">
            <span class="fc-muted">Relevansi Dokumen</span>{_bar(ev.relevance, '37, 99, 235')}<b style="text-align:right">{ev.relevance:.0%}</b>
          </div>
          <div class="fc-stack" title="Breakdown Probabilitas NLI">{stack}</div>
          <div class="fc-muted" style="margin-top:6px;font-size:0.82em;display:flex;justify-content:space-between;">
            <span>Entailment: <b>{s.get('ENTAILMENT',0):.1%}</b></span>
            <span>Neutral: <b>{s.get('NEUTRAL',0):.1%}</b></span>
            <span>Contradiction: <b>{s.get('CONTRADICTION',0):.1%}</b></span>
          </div>
        </div>""")
    return "".join(cards)


def render_explanation(result: FactCheckResult) -> str:
    if not result.explanation:
        return (
            "<div class='fc-card fc-muted'>"
            "<p>Aktifkan opsi <b>Explainability (SHAP)</b> di Pengaturan Lanjutan untuk melihat kontribusi tiap kata.</p>"
            "</div>"
        )
    ev = result.decisive_evidence
    target = result.explanation_target
    return f"""
    <div class="fc-card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <h4 style="margin:0;">🔍 Analisis Kata Kunci (Shapley Values)</h4>
        <span class="fc-badge" style="background:rgba(147,51,234,0.14);color:#7e22ce;border:1px solid rgba(147,51,234,0.3);">
          Target Prediksi: P({target})
        </span>
      </div>
      <div class="fc-legend">
        <div class="fc-legend-item">
          <span class="fc-legend-swatch" style="background:rgba(16,185,129,0.35);border:1px solid #10b981;"></span>
          <span>Hijau: Mendorong prediksi P({target})</span>
        </div>
        <div class="fc-legend-item">
          <span class="fc-legend-swatch" style="background:rgba(239,68,68,0.35);border:1px solid #ef4444;"></span>
          <span>Merah: Melawan prediksi</span>
        </div>
        <span class="fc-muted" style="margin-left:auto;font-size:0.85em;">(Arahkan kursor ke kata untuk melihat nilai SHAP numerik)</span>
      </div>
      <p class="fc-muted" style="margin:6px 0 10px;font-size:0.92em;">
        Evidence Penentu: <b>[{html.escape(ev.source.upper() if ev else '')}] {html.escape(ev.title if ev else '')}</b>
      </p>
      <div class="fc-passage" style="line-height:2.0;font-size:1.05em;border-left:4px solid #8b5cf6;">
        {highlight_html(result.explanation)}
      </div>
    </div>"""


def render_html(result: FactCheckResult) -> str:
    """Menggabungkan seluruh visualisasi HTML (verdict, evidence, explainability) lengkap dengan CSS."""
    return f"""
    <style>{CSS}</style>
    <div style="font-family:var(--fc-font);">
      {render_verdict(result)}
      <div style="margin-top:16px;">
        {render_explanation(result) if result.explanation else ''}
      </div>
      <div style="margin-top:16px;">
        <h3 style="margin:0 0 10px;font-size:1.2em;">📰 Daftar Bukti Berita (Evidences):</h3>
        {render_evidences(result)}
      </div>
    </div>
    """


def render_methodology_html() -> str:
    """Komponen visual arsitektur sistem dengan kartu formula matematis dan alur kerja."""
    return """
    <div class="fc-method-grid">
      <!-- Tahap 1 -->
      <div class="fc-method-card">
        <div class="fc-method-header">
          <span class="fc-method-badge">TAHAP 1</span>
          <h3 class="fc-method-title">🧹 Preprocessing Dokumen Berita</h3>
        </div>
        <p style="margin:4px 0 10px;color:var(--body-text-color);">Pembersihan data mentah hasil scraping 7 portal berita nasional (Detik, Kompas, Tempo, CNN Indonesia, Republika, Antara, Kumparan):</p>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">
          <span class="fc-chip" style="color:var(--body-text-color);background:rgba(127,127,127,0.08);border-color:rgba(127,127,127,0.2);">Pembersihan Dateline &amp; Boilerplate Jurnalisme</span>
          <span class="fc-chip" style="color:var(--body-text-color);background:rgba(127,127,127,0.08);border-color:rgba(127,127,127,0.2);">Pelindung Singkatan Bahasa Indonesia (dr., prof., kpu., dpr.)</span>
          <span class="fc-chip" style="color:var(--body-text-color);background:rgba(127,127,127,0.08);border-color:rgba(127,127,127,0.2);">Deduplikasi Artikel &amp; Filter Teks Pendek (&lt; 30 kata)</span>
        </div>
      </div>

      <!-- Tahap 2 -->
      <div class="fc-method-card">
        <div class="fc-method-header">
          <span class="fc-method-badge">TAHAP 2</span>
          <h3 class="fc-method-title">⚡ Hybrid Document Retrieval &amp; Rank Fusion</h3>
        </div>
        <p style="margin:4px 0 10px;color:var(--body-text-color);">Menggabungkan kekuatan pencarian leksikal (nama entitas, angka) dan semantik (pemahaman konteks &amp; parafrase):</p>
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));gap:10px;margin-bottom:12px;">
          <div style="padding:12px 14px;border-radius:10px;background:rgba(127,127,127,0.05);border:1px solid rgba(127,127,127,0.15);">
            <div style="font-weight:700;margin-bottom:4px;color:#2563eb;">① Okapi BM25 (Sparse Leksikal)</div>
            <div style="font-size:0.88em;opacity:0.85;">Pencocokan term frekuensi dan IDF atas teks judul + isi artikel lengkap.</div>
          </div>
          <div style="padding:12px 14px;border-radius:10px;background:rgba(127,127,127,0.05);border:1px solid rgba(127,127,127,0.15);">
            <div style="font-weight:700;margin-bottom:4px;color:#8b5cf6;">② Sentence-BERT (Dense Semantik)</div>
            <div style="font-size:0.88em;opacity:0.85;">Embedding 384-D <code>paraphrase-multilingual-MiniLM-L12-v2</code> dihitung dengan Cosine Similarity.</div>
          </div>
        </div>
        <div class="fc-math-box">
          <span style="font-weight:800;color:#2563eb;letter-spacing:0.02em;">RRF(d)</span>
          <span style="font-weight:700;">=</span>
          <span style="font-size:1.4em;line-height:1;">∑</span>
          <span style="font-size:0.85em;margin-top:8px;">r ∈ {BM25, dense}</span>
          <span style="display:inline-flex;flex-direction:column;align-items:center;vertical-align:middle;margin:0 6px;">
            <span style="border-bottom:1.5px solid currentColor;padding:0 8px;font-weight:700;">1</span>
            <span style="font-size:0.95em;">k + rank<sub>r</sub>(d)</span>
          </span>
          <span style="margin-left:12px;font-size:0.9em;opacity:0.85;border-left:1.5px solid rgba(127,127,127,0.3);padding-left:12px;">k = 60 &nbsp;──▶&nbsp; <b>Top-8 Artikel Emas</b></span>
        </div>
      </div>

      <!-- Tahap 3 -->
      <div class="fc-method-card">
        <div class="fc-method-header">
          <span class="fc-method-badge">TAHAP 3</span>
          <h3 class="fc-method-title">✂️ Sentence-Window Passage Selection</h3>
        </div>
        <p style="margin:4px 0;color:var(--body-text-color);">Artikel berita lengkap rata-rata terdiri dari 300–600 kata, melampaui batas representasi efektif model NLI. Sistem memotong artikel menggunakan <b>Sliding Window 3 Kalimat</b> (stride 2 kalimat) lalu mengurutkan kemiripan semantiknya terhadap klaim untuk mengambil <b>Top-5 Bukti Terfokus</b>.</p>
      </div>

      <!-- Tahap 4 -->
      <div class="fc-method-card">
        <div class="fc-method-header">
          <span class="fc-method-badge">TAHAP 4</span>
          <h3 class="fc-method-title">🧠 Natural Language Inference (Cross-Lingual mDeBERTa-v3)</h3>
        </div>
        <p style="margin:4px 0 10px;color:var(--body-text-color);">Model Transformer canggih <code>MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</code> membaca pasangan <i>[Premis: Bukti]</i> dan <i>[Hipotesis: Klaim]</i> secara serentak untuk menghasilkan distribusi probabilitas terkalibrasi:</p>
        <div class="fc-math-box">
          <span style="font-weight:600;">Distribusi Softmax:</span>
          <span class="fc-badge" style="background:rgba(16,185,129,0.15);color:#065f46;border:1px solid rgba(16,185,129,0.3);">P(Entailment)</span>
          <span>·</span>
          <span class="fc-badge" style="background:rgba(245,158,11,0.15);color:#92400e;border:1px solid rgba(245,158,11,0.3);">P(Neutral)</span>
          <span>·</span>
          <span class="fc-badge" style="background:rgba(239,68,68,0.15);color:#991b1b;border:1px solid rgba(239,68,68,0.3);">P(Contradiction)</span>
        </div>
      </div>

      <!-- Tahap 5 -->
      <div class="fc-method-card" style="border:2px solid #3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,0.15);">
        <div class="fc-method-header">
          <span class="fc-method-badge" style="background:#2563eb;color:#ffffff;border-color:#2563eb;">TAHAP 5 (UTAMA)</span>
          <h3 class="fc-method-title">⚖️ Multi-Evidence Aggregation (FEVER-Style Paradigm)</h3>
        </div>
        <p style="margin:4px 0 12px;color:var(--body-text-color);">
          Keputusan tidak dirata-ratakan agar satu bukti kuat tidak "diencerkan" oleh bukti netral (*anti-dilution*). Keputusan akhir diambil berdasarkan 4 aturan matematis berikut:
        </p>

        <div class="fc-rule-container">
          <div class="fc-rule-card gate">
            <div class="fc-rule-head" style="color:#2563eb;">🛡️ Gerbang Relevansi (Relevance Gate)</div>
            <div class="fc-rule-body">
              Hanya passage bukti dengan kemiripan:<br>
              <code style="background:rgba(59,130,246,0.15);padding:3px 8px;border-radius:6px;color:#1d4ed8;font-weight:700;display:inline-block;margin:4px 0;">Relevance ≥ τ<sub>rel</sub> = 0.35</code><br>
              yang dipertimbangkan masuk ke tahap agregasi keputusan.
            </div>
          </div>

          <div class="fc-rule-card supported">
            <div class="fc-rule-head" style="color:#10b981;">✔ SUPPORTED (Didukung Fakta)</div>
            <div class="fc-rule-body">
              Diputuskan jika:<br>
              <code style="background:rgba(16,185,129,0.15);padding:3px 8px;border-radius:6px;color:#065f46;font-weight:700;display:inline-block;margin:4px 0;">max P(entailment) ≥ 0.60</code><br>
              dan <code style="font-weight:700;">≥ max P(contradiction)</code>
            </div>
          </div>

          <div class="fc-rule-card refuted">
            <div class="fc-rule-head" style="color:#ef4444;">✘ REFUTED (Bertentangan / Hoaks)</div>
            <div class="fc-rule-body">
              Diputuskan jika:<br>
              <code style="background:rgba(239,68,68,0.15);padding:3px 8px;border-radius:6px;color:#991b1b;font-weight:700;display:inline-block;margin:4px 0;">max P(contradiction) ≥ 0.60</code><br>
              (Satu bantahan kuat cukup membuktikan hoaks)
            </div>
          </div>

          <div class="fc-rule-card nei">
            <div class="fc-rule-head" style="color:#d97706;">? NOT ENOUGH INFO (Bukti Tidak Cukup)</div>
            <div class="fc-rule-body">
              Diputuskan jika:<br>
              <code style="background:rgba(245,158,11,0.15);padding:3px 8px;border-radius:6px;color:#92400e;font-weight:700;display:inline-block;margin:4px 0;">Kondisi selain di atas</code><br>
              (Tidak ada bukti yang mencapai ambang keyakinan ≥ 0.60)
            </div>
          </div>
        </div>
      </div>

      <!-- Tahap 6 -->
      <div class="fc-method-card">
        <div class="fc-method-header">
          <span class="fc-method-badge">TAHAP 6</span>
          <h3 class="fc-method-title">🔬 Explainable AI (SHAP Saliency Attribution)</h3>
        </div>
        <p style="margin:4px 0 10px;color:var(--body-text-color);">
          Menggunakan algoritma <b>SHAP Partition Explainer</b> untuk menghitung nilai kontribusi marjinal setiap kata pada <i>Evidence Penentu</i> terhadap probabilitas keputusan model.
        </p>
        <div class="fc-math-box" style="justify-content:flex-start;">
          <span style="font-weight:600;">Formula Nilai Shapley per kata i:</span>
          <code style="color:#4338ca;font-weight:700;padding:2px 8px;border-radius:6px;background:rgba(67,56,202,0.08);">ϕ<sub>i</sub> = ∑ [ |S|!(|N|-|S|-1)! / |N|! ] · [ f(S ∪ {i}) - f(S) ]</code>
        </div>
        <p style="margin:6px 0 0;font-size:0.9em;opacity:0.85;">Kata dengan kontribusi positif (&gt; 0) disorot dengan warna interaktif dan tooltip nilai pada tab Explainability.</p>
      </div>

      <!-- Keterbatasan & Etika -->
      <div class="fc-alert-box">
        <h4>⚠️ Keterbatasan &amp; Etika Penggunaan Sistem</h4>
        <ul style="margin:6px 0 0;padding-left:20px;line-height:1.65;font-size:0.95em;">
          <li><b>Batasan Temporal Korpus</b>: Korpus berita mencakup rentang waktu <b>Maret – April 2023</b> (~32.000 artikel). Klaim peristiwa di luar rentang waktu tersebut secara alami akan menghasilkan <b>BUKTI TIDAK CUKUP</b>.</li>
          <li><b>Makna Epistemologis</b>: Status <i>"Didukung Berita"</i> mencerminkan konsistensi logis klaim terhadap fakta di korpus media massa nasional, bukan kebenaran absolut filosofis.</li>
          <li><b>Penalaran Kompleks</b>: NLI masih memiliki tantangan pada negasi ganda, perbandingan angka yang rumit, dan penalaran temporal yang mendalam.</li>
        </ul>
      </div>
    </div>
    """


def render_kpi_cards() -> str:
    """Kartu metrik performa formal hasil benchmark sistem."""
    return """
    <div class="fc-kpi-grid">
      <div class="fc-kpi-card" style="border-top:4px solid #10b981;">
        <div class="fc-kpi-sub">Akurasi NLI (Evidence Emas)</div>
        <div class="fc-kpi-val" style="color:#10b981;">88.9%</div>
        <div class="fc-kpi-sub">vs 42.2% baseline English-only</div>
      </div>
      <div class="fc-kpi-card" style="border-top:4px solid #3b82f6;">
        <div class="fc-kpi-sub">Macro-F1 Score NLI</div>
        <div class="fc-kpi-val" style="color:#3b82f6;">0.890</div>
        <div class="fc-kpi-sub">Seimbang di 3 kelas verifikasi</div>
      </div>
      <div class="fc-kpi-card" style="border-top:4px solid #8b5cf6;">
        <div class="fc-kpi-sub">Hybrid Recall@5 Artikel</div>
        <div class="fc-kpi-val" style="color:#8b5cf6;">53.3%</div>
        <div class="fc-kpi-sub">MRR 0.398 (BM25 + Dense + RRF)</div>
      </div>
      <div class="fc-kpi-card" style="border-top:4px solid #f59e0b;">
        <div class="fc-kpi-sub">End-to-End Accuracy</div>
        <div class="fc-kpi-val" style="color:#f59e0b;">66.7%</div>
        <div class="fc-kpi-sub">Agregasi Max (FEVER Paradigm)</div>
      </div>
    </div>
    """


def build_app(checker: FactChecker):
    import gradio as gr

    cfg = checker.config
    n_articles = len(checker.articles)
    sources = checker.articles["source"].nunique()

    def run(claim: str, top_k: int, method: str, strategy: str, explain: bool):
        claim = (claim or "").strip()
        if len(claim.split()) < 3:
            raise gr.Error("Klaim terlalu pendek — tulis minimal 3 kata.")
        result = checker.check(
            claim,
            explain=explain,
            top_passages=int(top_k),
            retrieval_method=method,
            aggregation=strategy,
        )
        fig = (
            plot_token_importance(result.explanation, result.explanation_target)
            if result.explanation
            else None
        )
        return (
            render_verdict(result),
            render_evidences(result),
            render_explanation(result),
            fig,
            result.to_dict(),
        )

    eval_md = PROJECT_ROOT / "reports" / "evaluation_summary.md"
    eval_png = PROJECT_ROOT / "reports" / "confusion_matrices.png"

    with gr.Blocks(title="Indonesian News Fact-Checker", css=CSS) as demo:
        gr.HTML(f"""
        <div class="fc-header">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <h1>🔎 Indonesian News Fact-Checker</h1>
            <span class="fc-chip" style="background:rgba(59,130,246,0.3);border-color:#60a5fa;color:#eff6ff;">
              ⚡ v2.0 · Multilingual NLI
            </span>
          </div>
          <p>Sistem verifikasi klaim berita berbahasa Indonesia terhadap <b>{n_articles:,} artikel berita terverifikasi</b>
             dari <b>{sources} portal berita nasional</b> (Maret–April 2023) menggunakan Retrieval-Augmented NLI dan Explainable AI.</p>
          <div class="fc-chips">
            <span class="fc-chip">① Hybrid Retrieval (BM25 + SBERT + RRF)</span>
            <span class="fc-chip">② Sentence-Window Passage Selection</span>
            <span class="fc-chip">③ Multilingual NLI (mDeBERTa-v3)</span>
            <span class="fc-chip">④ Multi-Evidence FEVER Aggregation</span>
            <span class="fc-chip">⑤ Explainable AI (SHAP Saliency)</span>
          </div>
        </div>""")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                claim = gr.Textbox(
                    label="📝 Klaim yang Ingin Diverifikasi",
                    lines=3,
                    placeholder="Contoh: Pemerintah menetapkan 1 Ramadan 1444 H jatuh pada 23 Maret 2023.",
                )
                with gr.Accordion("⚙️ Pengaturan Lanjutan (Pipeline Hyperparameters)", open=False):
                    top_k = gr.Slider(
                        1, 10, value=cfg.top_passages, step=1,
                        label="Jumlah Bukti (Top-K Passage Evidence)"
                    )
                    method = gr.Radio(
                        [("Hybrid (BM25 + SBERT)", "hybrid"), ("BM25 Saja (Leksikal)", "bm25"), ("Dense Saja (Semantik)", "dense")],
                        value="hybrid", label="Metode Retrieval Artikel"
                    )
                    strategy = gr.Radio(
                        [("Max (FEVER Standard)", "max"), ("Weighted Average", "weighted"), ("Mean (Rata-rata)", "mean")],
                        value=cfg.aggregation, label="Strategi Agregasi Multi-Evidence"
                    )
                    explain = gr.Checkbox(
                        value=True,
                        label="Explainability (SHAP) — Analisis kontribusi kata (menambah ±10–30 detik di CPU)"
                    )
                btn = gr.Button("🚀 Periksa Fakta Sekarang", variant="primary", size="lg")
                gr.Examples(
                    EXAMPLES,
                    inputs=claim,
                    label="💡 Contoh Klaim Populer (Klik untuk Menguji):",
                )
            with gr.Column(scale=6):
                verdict_html = gr.HTML("""
                <div class="fc-card fc-muted" style="text-align:center;padding:38px 24px;">
                  <div style="font-size:2.4em;margin-bottom:8px;">🔍</div>
                  <h3 style="margin:0 0 6px;">Siap Memeriksa Fakta</h3>
                  <p style="margin:0;font-size:0.95em;">Masukkan pernyataan klaim di sisi kiri atau pilih salah satu contoh preset untuk melihat keputusan dan analisis bukti.</p>
                </div>
                """)

        with gr.Tabs():
            with gr.Tab("📰 Bukti Berita (Evidence)"):
                evidence_html = gr.HTML()
            with gr.Tab("🔬 Explainability (SHAP)"):
                expl_html = gr.HTML()
                expl_plot = gr.Plot(label="Diagram Kontribusi Kata Paling Berpengaruh")
            with gr.Tab("🎯 Evaluasi & Benchmark"):
                gr.HTML(render_kpi_cards())
                if eval_png.exists():
                    gr.Image(str(eval_png), label="Confusion Matrix 3x3 (Benchmark 48 Kasus)", show_label=True)
                if eval_md.exists():
                    gr.Markdown(eval_md.read_text(encoding="utf-8"))
                else:
                    gr.Markdown("Laporan evaluasi belum digenerate. Jalankan `python -m fact_checker evaluate` untuk membuat laporan.")
            with gr.Tab("🧾 Raw JSON"):
                raw_json = gr.JSON(label="Struktur Data Hasil FactCheckResult")
            with gr.Tab("📑 Metodologi & Arsitektur"):
                gr.HTML(render_methodology_html())

        outputs = [verdict_html, evidence_html, expl_html, expl_plot, raw_json]
        btn.click(run, [claim, top_k, method, strategy, explain], outputs)
        claim.submit(run, [claim, top_k, method, strategy, explain], outputs)
    return demo


METHODOLOGY_MD = r"""
### 🏗️ Arsitektur Pipeline Sistem
1. **Preprocessing Dokumen**
   - Pembersihan boilerplate 7 portal berita Indonesia (dateline, *Baca juga*, *Pilihan Editor*, embed video `[Gambas]`, email tersamar).
   - Segmentasi kalimat berbasis regex sadar singkatan Indonesia (*Dr., Drs., Prof., Ir., Bpk., dll.*).
   - Deduplikasi dan filter teks pendek (< 30 kata).
2. **Hybrid Retrieval Artikel**
   - **Okapi BM25** leksikal atas teks judul + isi artikel.
   - **Dense Semantic Embedding** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d) atas judul + ringkasan.
   - **Reciprocal Rank Fusion (RRF)**:
     $$RRF(d) = \sum_{r \in \{\text{BM25},\, \text{dense}\}} \frac{1}{k + \text{rank}_r(d)}, \quad k = 60$$
3. **Sentence-Window Passage Selection**
   - Memecah artikel top kandidat menjadi passage berukuran 3 kalimat dengan pergeseran (stride) 2 kalimat.
   - Mengurutkan passage kandidat berdasarkan cosine similarity terhadap klaim.
4. **Natural Language Inference (NLI)**
   - Model multilingual SOTA: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
   - Menilai pasangan *(Premise = Passage, Hypothesis = Klaim)* menjadi distribusi probabilitas:
     $$P(\text{entailment}), \quad P(\text{neutral}), \quad P(\text{contradiction})$$
5. **Multi-Evidence Aggregation (FEVER-Style)**
   - Hanya mempertimbangkan passage dengan kemiripan relevansi $\ge \tau_{\text{rel}} = 0.35$.
   - **SUPPORTED**: Jika $\max P(\text{entailment}) \ge 0.60$ dan $\ge \max P(\text{contradiction})$.
   - **REFUTED**: Jika $\max P(\text{contradiction}) \ge 0.60$.
   - **NOT_ENOUGH_INFO**: Selain kondisi di atas.
6. **Explainable AI (SHAP Saliency)**
   - Menghitung Shapley values per kata pada passage penentu untuk memberikan transparansi audit logis.

### ⚠️ Keterbatasan & Etika
* Korpus berita mencakup rentang waktu **Maret – April 2023**. Klaim di luar rentang waktu tersebut akan cenderung menghasilkan *BUKTI TIDAK CUKUP*.
* Status "Didukung Berita" mencerminkan konsistensi klaim terhadap fakta di korpus media massa nasional, bukan kebenaran absolut filosofis.
"""


def launch(
    config: Optional[Config] = None,
    share: bool = False,
    port: Optional[int] = None,
    inbrowser: bool = False,
    inline: bool = False,
):
    import gradio as gr

    checker = FactChecker(config or Config())
    demo = build_app(checker)
    demo.queue().launch(
        share=share,
        server_port=port,
        inbrowser=inbrowser,
        inline=inline,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    )
