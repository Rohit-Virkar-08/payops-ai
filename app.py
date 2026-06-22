"""
AP Autopilot — Agent Command Center (Streamlit).

    streamlit run app.py

Dark-neon + Light themes: watch the agent pipeline light up node-by-node, stream each
agent's conclusion, and land on a glowing routed verdict + AI summary.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

import fitz  # PyMuPDF — PDF page previews
import pandas as pd
import streamlit as st

from extractors import extract
from extraction_agent import ExtractionAgent
from verification_agent import VerificationAgent
from po_matching_agent import POMatchingAgent
from decision_agent import DecisionAgent
from summary_agent import SummaryAgent
from po_store import POStore

PO_CSV = Path("data/po_dataset.csv")
LEDGER_CSV = Path("data/invoice_ledger.csv")

st.set_page_config(page_title="PayOPS-AI", page_icon="🧾", layout="wide")

# --------------------------------------------------------------------------- #
# Theme CSS
# --------------------------------------------------------------------------- #

_DARK_CSS = """
  .stApp { background: radial-gradient(1200px 600px at 20% -10%, #1b1245 0%, #0a0a14 55%) fixed; color:#e5e7eb; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] { background: #0e1020; border-right: 1px solid #23263a; }
  h1,h2,h3,h4 { color:#f3f4f6 !important; }

  .stApp p, .stApp li, .stApp span, .stMarkdown, .stCaption,
  [data-testid="stWidgetLabel"] p, label, [data-testid="stMarkdownContainer"] p { color:#cbd0e0 !important; }
  [data-testid="stSidebar"] * { color:#c3c8da; }
  [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color:#7f859c !important; }
  [data-testid="stMetricLabel"] { color:#9aa0b4 !important; }
  [data-testid="stMetricValue"] { color:#f3f4f6 !important; }

  @keyframes fadeUp { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform: none; } }
  .fade { animation: fadeUp .55s cubic-bezier(.2,.7,.2,1) both; }
  section.main .block-container { animation: fadeUp .45s ease both; }

  .hero { padding: 8px 2px 2px; }
  .hero h1 { font-size: 2.1rem; font-weight: 900; letter-spacing:-.5px;
             background: linear-gradient(90deg,#a78bfa,#22d3ee,#ec4899);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .hero p { color:#9aa0b4; margin-top:-4px; }

  .flow { display:flex; align-items:center; justify-content:space-between;
          gap:4px; padding:18px 10px; margin:6px 0 4px; overflow:visible;
          background:#0e1124; border:1px solid #23263a; border-radius:18px; }
  .node { position:relative; display:flex; flex-direction:column; align-items:center; gap:8px;
          width:92px; text-align:center; cursor:help; }

  .card { position:absolute; top:118%; width:270px; padding:16px 18px; border-radius:14px;
          background:rgba(16,19,38,.98); border:1px solid #3a3f63; text-align:left;
          box-shadow:0 18px 42px rgba(0,0,0,.55), 0 0 26px rgba(139,92,246,.30);
          opacity:0; pointer-events:none; z-index:60;
          transition:opacity .18s ease, transform .18s ease; }
  .card .ct { font-weight:800; color:#a78bfa; font-size:1rem; margin-bottom:6px; }
  .card .cb { font-size:.88rem; color:#cbd0e0; line-height:1.5; }
  .card::after { content:""; position:absolute; bottom:100%; width:12px; height:12px;
          background:rgba(16,19,38,.98); border-left:1px solid #3a3f63; border-top:1px solid #3a3f63;
          transform:rotate(45deg); margin-bottom:-6px; }
  .card-center { left:50%; transform:translateX(-50%) translateY(8px); }
  .card-center::after { left:50%; margin-left:-6px; }
  .card-left { left:0; transform:translateY(8px); }
  .card-left::after { left:34px; }
  .card-right { right:0; transform:translateY(8px); }
  .card-right::after { right:34px; }
  .node:hover .card-center { opacity:1; transform:translateX(-50%) translateY(0); }
  .node:hover .card-left   { opacity:1; transform:translateY(0); }
  .node:hover .card-right  { opacity:1; transform:translateY(0); }
  .dot  { width:54px; height:54px; border-radius:50%; display:flex; align-items:center; justify-content:center;
          font-size:22px; border:2px solid #2a2e45; background:#141833; color:#8b90a8;
          transition: all .35s ease; }
  .nlabel { font-size:.72rem; color:#8b90a8; font-weight:600; }
  .node.active .dot { border-color:#22d3ee; color:#a5f3fc; background:#0b2330;
          box-shadow:0 0 0 4px rgba(34,211,238,.15), 0 0 26px rgba(34,211,238,.65);
          transform:scale(1.10); }
  .node.active .nlabel { color:#67e8f9; }
  .node.done .dot { border-color:#7c3aed; color:#ddd6fe; background:#1a1336;
          box-shadow:0 0 18px rgba(139,92,246,.5); }
  .node.done .nlabel { color:#c4b5fd; }
  .conn { flex:1; height:3px; border-radius:3px; background:#23263a; }
  .conn.on { background:linear-gradient(90deg,#7c3aed,#22d3ee); box-shadow:0 0 10px rgba(124,58,237,.6); }

  .trace { background:#0e1124; border:1px solid #23263a; border-radius:16px; padding:14px 16px; min-height:210px; }
  .trace .row { display:flex; gap:10px; padding:7px 0; border-bottom:1px dashed #20233a; font-size:.92rem; }
  .trace .row:last-child { border-bottom:none; }
  .trace .ag { color:#67e8f9; font-weight:700; min-width:88px; }
  .trace .ms { color:#cbd0e0; }

  .vbox { display:flex; align-items:center; gap:18px; padding:26px 28px; border-radius:18px;
          color:#0a0a0a; box-shadow:0 12px 34px rgba(0,0,0,.40); position:relative; overflow:visible; }
  .vbox .vicon  { font-size:2.6rem; }
  .vbox .vlabel { font-size:1.95rem; font-weight:900; color:#0a0a0a !important; line-height:1.05; }
  .vbox .vsub   { font-size:.82rem; font-weight:800; letter-spacing:.6px;
                  color:#0a0a0a !important; opacity:.65; margin-top:2px; }
  .vbox .vexp   { font-size:.92rem; font-weight:600; color:#0a0a0a !important;
                  opacity:.82; margin-top:9px; line-height:1.38; }

  /* verdict info button + tooltip */
  .vinfo-wrap { position:absolute; top:14px; right:14px; }
  .vinfo-btn { width:24px; height:24px; border-radius:50%;
               background:rgba(0,0,0,.15); border:1.5px solid rgba(0,0,0,.22);
               color:rgba(0,0,0,.65); font-size:11px; font-weight:900; font-style:normal;
               display:flex; align-items:center; justify-content:center;
               cursor:default; user-select:none; line-height:1;
               transition:background .15s, transform .15s; }
  .vinfo-wrap:hover .vinfo-btn { background:rgba(0,0,0,.28); transform:scale(1.1); }
  .vinfo-card { position:absolute; bottom:calc(100% + 14px); right:-8px; width:310px;
                background:#0d1023; border:1px solid #7c3aed; border-radius:16px;
                padding:18px 20px; text-align:left; z-index:200;
                box-shadow:0 18px 44px rgba(0,0,0,.65), 0 0 30px rgba(124,58,237,.25);
                opacity:0; pointer-events:none; transform:translateY(8px);
                transition:opacity .2s ease, transform .2s ease; }
  .vinfo-wrap:hover .vinfo-card { opacity:1; pointer-events:auto; transform:translateY(0); }
  .vinfo-card::after { content:""; position:absolute; top:100%; right:12px;
                       width:11px; height:11px; background:#0d1023;
                       border-right:1px solid #7c3aed; border-bottom:1px solid #7c3aed;
                       transform:rotate(45deg); margin-top:-5px; }
  .vic-title { font-size:.95rem; font-weight:800; color:#a78bfa; margin-bottom:12px; }
  .vic-sect { margin-bottom:10px; }
  .vic-sect:last-child { margin-bottom:0; }
  .vic-lbl { font-size:.66rem; font-weight:800; letter-spacing:.9px;
             text-transform:uppercase; color:#22d3ee; margin-bottom:5px; }
  .vinfo-card ul { margin:0; padding-left:16px; }
  .vinfo-card ul li { font-size:.8rem; color:#c4c9de !important; line-height:1.5; margin-bottom:3px; }
  .vinfo-card ul li:last-child { margin-bottom:0; }

  .stage { background:#0e1124; border:1px solid #23263a; border-radius:16px; padding:6px 8px; }
  .stage .srow { display:flex; align-items:center; gap:13px; padding:12px 12px;
                 border-bottom:1px dashed #20233a; border-left:3px solid transparent; border-radius:8px; }
  .stage .srow:last-child { border-bottom:none; }
  .stage .srow.r-bad  { background:rgba(248,113,113,.09); border-left-color:#f87171; }
  .stage .srow.r-warn { background:rgba(251,191,36,.08); border-left-color:#fbbf24; }
  .stage .sico { width:36px; height:36px; border-radius:10px; display:flex; align-items:center;
                 justify-content:center; font-size:16px; background:#141833; border:1px solid #2a2e45; flex:0 0 auto; }
  .stage .smid { flex:1; min-width:0; }
  .stage .snm { font-weight:700; color:#e5e7eb !important; font-size:.92rem; }
  .stage .sdt { font-size:.82rem; color:#9aa0b4 !important; margin-top:2px; line-height:1.35; }
  .stage .spill { font-size:.72rem; font-weight:800; padding:4px 11px; border-radius:999px; white-space:nowrap; }
  .stage .spill.ok   { background:rgba(52,211,153,.14); color:#34d399 !important; border:1px solid rgba(52,211,153,.45); }
  .stage .spill.warn { background:rgba(251,191,36,.14); color:#fbbf24 !important; border:1px solid rgba(251,191,36,.45); }
  .stage .spill.bad  { background:rgba(248,113,113,.15); color:#f87171 !important; border:1px solid rgba(248,113,113,.5); }

  .chips { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 2px; }
  .chip { padding:7px 13px; border-radius:999px; font-size:.82rem; font-weight:700;
          background:#141833; border:1px solid #2a2e45; color:#cbd0e0; }

  [data-testid="stFileUploaderDropzone"] {
          background:#141833 !important; border:1px dashed #3a3f63 !important; }
  [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stFileUploaderDropzoneInstructions"] span,
  [data-testid="stFileUploaderDropzoneInstructions"] small { color:#cbd0e0 !important; }
  [data-testid="stFileUploaderDropzone"] button {
          background:#1a1f3a !important; color:#e5e7eb !important; border:1px solid #4b4f73 !important; }

  .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p { color:#cbd0e0 !important; font-weight:600; }
  .stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab"]:hover p { color:#e5e7eb !important; }
  .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p { color:#a78bfa !important; }
  .stTabs [data-baseweb="tab-highlight"] { background:#a78bfa !important; }

  .info-card { background:#0e1124; border:1px solid #23263a; border-radius:14px;
               padding:16px 20px; margin-top:14px; }
  .info-card .ic-head { font-size:.75rem; font-weight:800; letter-spacing:.8px;
                        text-transform:uppercase; margin-bottom:10px; }
  .info-card .ic-head.why  { color:#a78bfa; }
  .info-card .ic-head.act  { color:#22d3ee; }
  .info-card ul { margin:0; padding-left:18px; }
  .info-card ul li { color:#cbd0e0 !important; font-size:.9rem; line-height:1.6; margin-bottom:4px; }
  .info-card ul li:last-child { margin-bottom:0; }

  /* accordion detail sections (main panel only) */
  .detail-label { font-size:.72rem; font-weight:800; letter-spacing:.9px;
                  text-transform:uppercase; color:#7b7f9e; margin:28px 0 10px; }
  section.main [data-testid="stExpander"] {
    background:#0e1124 !important; border:1px solid #23263a !important;
    border-radius:14px !important; margin-bottom:8px; overflow:hidden;
    transition:border-color .2s ease, box-shadow .2s ease; }
  section.main [data-testid="stExpander"]:has(details[open]) {
    border-color:#7c3aed !important;
    box-shadow:0 0 0 1px rgba(124,58,237,.35), 0 0 22px rgba(124,58,237,.18) !important; }
  section.main [data-testid="stExpander"] summary { background:#0e1124 !important; }
  section.main [data-testid="stExpander"] summary p {
    color:#c4b5fd !important; font-weight:700 !important; font-size:.93rem; }
  section.main [data-testid="stExpander"] summary:hover p { color:#a78bfa !important; }
  section.main [data-testid="stExpander"] details[open] summary p { color:#a78bfa !important; }
  section.main [data-testid="stExpanderToggleIcon"] svg { stroke:#7c3aed !important; }
"""

_LIGHT_CSS = """
  .stApp { background: radial-gradient(1200px 600px at 20% -10%, #ede9fe 0%, #f8fafc 55%) fixed; color:#1e293b; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
  h1,h2,h3,h4 { color:#0f172a !important; }

  .stApp p, .stApp li, .stApp span, .stMarkdown, .stCaption,
  [data-testid="stWidgetLabel"] p, label, [data-testid="stMarkdownContainer"] p { color:#374151 !important; }
  [data-testid="stSidebar"] * { color:#374151; }
  [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color:#94a3b8 !important; }
  [data-testid="stMetricLabel"] { color:#64748b !important; }
  [data-testid="stMetricValue"] { color:#0f172a !important; }

  @keyframes fadeUp { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform: none; } }
  .fade { animation: fadeUp .55s cubic-bezier(.2,.7,.2,1) both; }
  section.main .block-container { animation: fadeUp .45s ease both; }

  .hero { padding: 8px 2px 2px; }
  .hero h1 { font-size: 2.1rem; font-weight: 900; letter-spacing:-.5px;
             background: linear-gradient(90deg,#7c3aed,#0891b2,#db2777);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .hero p { color:#64748b; margin-top:-4px; }

  .flow { display:flex; align-items:center; justify-content:space-between;
          gap:4px; padding:18px 10px; margin:6px 0 4px; overflow:visible;
          background:#ffffff; border:1px solid #e2e8f0; border-radius:18px;
          box-shadow:0 1px 4px rgba(0,0,0,.06); }
  .node { position:relative; display:flex; flex-direction:column; align-items:center; gap:8px;
          width:92px; text-align:center; cursor:help; }

  .card { position:absolute; top:118%; width:270px; padding:16px 18px; border-radius:14px;
          background:rgba(255,255,255,.98); border:1px solid #e2e8f0; text-align:left;
          box-shadow:0 18px 42px rgba(0,0,0,.12), 0 0 26px rgba(124,58,237,.10);
          opacity:0; pointer-events:none; z-index:60;
          transition:opacity .18s ease, transform .18s ease; }
  .card .ct { font-weight:800; color:#7c3aed; font-size:1rem; margin-bottom:6px; }
  .card .cb { font-size:.88rem; color:#374151; line-height:1.5; }
  .card::after { content:""; position:absolute; bottom:100%; width:12px; height:12px;
          background:rgba(255,255,255,.98); border-left:1px solid #e2e8f0; border-top:1px solid #e2e8f0;
          transform:rotate(45deg); margin-bottom:-6px; }
  .card-center { left:50%; transform:translateX(-50%) translateY(8px); }
  .card-center::after { left:50%; margin-left:-6px; }
  .card-left { left:0; transform:translateY(8px); }
  .card-left::after { left:34px; }
  .card-right { right:0; transform:translateY(8px); }
  .card-right::after { right:34px; }
  .node:hover .card-center { opacity:1; transform:translateX(-50%) translateY(0); }
  .node:hover .card-left   { opacity:1; transform:translateY(0); }
  .node:hover .card-right  { opacity:1; transform:translateY(0); }
  .dot  { width:54px; height:54px; border-radius:50%; display:flex; align-items:center; justify-content:center;
          font-size:22px; border:2px solid #cbd5e1; background:#f8fafc; color:#94a3b8;
          transition: all .35s ease; }
  .nlabel { font-size:.72rem; color:#64748b; font-weight:600; }
  .node.active .dot { border-color:#0891b2; color:#0e7490; background:#e0f2fe;
          box-shadow:0 0 0 4px rgba(8,145,178,.12), 0 0 20px rgba(8,145,178,.35);
          transform:scale(1.10); }
  .node.active .nlabel { color:#0891b2; }
  .node.done .dot { border-color:#7c3aed; color:#7c3aed; background:#f3f0ff;
          box-shadow:0 0 14px rgba(124,58,237,.20); }
  .node.done .nlabel { color:#7c3aed; }
  .conn { flex:1; height:3px; border-radius:3px; background:#e2e8f0; }
  .conn.on { background:linear-gradient(90deg,#7c3aed,#0891b2); box-shadow:0 0 8px rgba(124,58,237,.25); }

  .trace { background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:14px 16px; min-height:210px;
           box-shadow:0 1px 4px rgba(0,0,0,.05); }
  .trace .row { display:flex; gap:10px; padding:7px 0; border-bottom:1px dashed #e2e8f0; font-size:.92rem; }
  .trace .row:last-child { border-bottom:none; }
  .trace .ag { color:#0891b2; font-weight:700; min-width:88px; }
  .trace .ms { color:#374151; }

  .vbox { display:flex; align-items:center; gap:18px; padding:26px 28px; border-radius:18px;
          color:#0a0a0a; box-shadow:0 12px 34px rgba(0,0,0,.15); position:relative; overflow:visible; }
  .vbox .vicon  { font-size:2.6rem; }
  .vbox .vlabel { font-size:1.95rem; font-weight:900; color:#0a0a0a !important; line-height:1.05; }
  .vbox .vsub   { font-size:.82rem; font-weight:800; letter-spacing:.6px;
                  color:#0a0a0a !important; opacity:.65; margin-top:2px; }
  .vbox .vexp   { font-size:.92rem; font-weight:600; color:#0a0a0a !important;
                  opacity:.82; margin-top:9px; line-height:1.38; }

  /* verdict info button + tooltip */
  .vinfo-wrap { position:absolute; top:14px; right:14px; }
  .vinfo-btn { width:24px; height:24px; border-radius:50%;
               background:rgba(0,0,0,.12); border:1.5px solid rgba(0,0,0,.18);
               color:rgba(0,0,0,.55); font-size:11px; font-weight:900; font-style:normal;
               display:flex; align-items:center; justify-content:center;
               cursor:default; user-select:none; line-height:1;
               transition:background .15s, transform .15s; }
  .vinfo-wrap:hover .vinfo-btn { background:rgba(0,0,0,.22); transform:scale(1.1); }
  .vinfo-card { position:absolute; bottom:calc(100% + 14px); right:-8px; width:310px;
                background:#ffffff; border:1px solid #7c3aed; border-radius:16px;
                padding:18px 20px; text-align:left; z-index:200;
                box-shadow:0 18px 44px rgba(0,0,0,.12), 0 0 24px rgba(124,58,237,.12);
                opacity:0; pointer-events:none; transform:translateY(8px);
                transition:opacity .2s ease, transform .2s ease; }
  .vinfo-wrap:hover .vinfo-card { opacity:1; pointer-events:auto; transform:translateY(0); }
  .vinfo-card::after { content:""; position:absolute; top:100%; right:12px;
                       width:11px; height:11px; background:#ffffff;
                       border-right:1px solid #7c3aed; border-bottom:1px solid #7c3aed;
                       transform:rotate(45deg); margin-top:-5px; }
  .vic-title { font-size:.95rem; font-weight:800; color:#7c3aed; margin-bottom:12px; }
  .vic-sect { margin-bottom:10px; }
  .vic-sect:last-child { margin-bottom:0; }
  .vic-lbl { font-size:.66rem; font-weight:800; letter-spacing:.9px;
             text-transform:uppercase; color:#0891b2; margin-bottom:5px; }
  .vinfo-card ul { margin:0; padding-left:16px; }
  .vinfo-card ul li { font-size:.8rem; color:#374151 !important; line-height:1.5; margin-bottom:3px; }
  .vinfo-card ul li:last-child { margin-bottom:0; }

  .stage { background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:6px 8px;
           box-shadow:0 1px 4px rgba(0,0,0,.05); }
  .stage .srow { display:flex; align-items:center; gap:13px; padding:12px 12px;
                 border-bottom:1px dashed #e2e8f0; border-left:3px solid transparent; border-radius:8px; }
  .stage .srow:last-child { border-bottom:none; }
  .stage .srow.r-bad  { background:rgba(248,113,113,.07); border-left-color:#f87171; }
  .stage .srow.r-warn { background:rgba(251,191,36,.06); border-left-color:#fbbf24; }
  .stage .sico { width:36px; height:36px; border-radius:10px; display:flex; align-items:center;
                 justify-content:center; font-size:16px; background:#f8fafc; border:1px solid #e2e8f0; flex:0 0 auto; }
  .stage .smid { flex:1; min-width:0; }
  .stage .snm { font-weight:700; color:#0f172a !important; font-size:.92rem; }
  .stage .sdt { font-size:.82rem; color:#64748b !important; margin-top:2px; line-height:1.35; }
  .stage .spill { font-size:.72rem; font-weight:800; padding:4px 11px; border-radius:999px; white-space:nowrap; }
  .stage .spill.ok   { background:rgba(5,150,105,.10); color:#047857 !important; border:1px solid rgba(5,150,105,.35); }
  .stage .spill.warn { background:rgba(180,130,0,.10); color:#b45309 !important; border:1px solid rgba(180,130,0,.35); }
  .stage .spill.bad  { background:rgba(220,38,38,.08); color:#dc2626 !important; border:1px solid rgba(220,38,38,.35); }

  .chips { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 2px; }
  .chip { padding:7px 13px; border-radius:999px; font-size:.82rem; font-weight:700;
          background:#f1f5f9; border:1px solid #e2e8f0; color:#374151; }

  [data-testid="stFileUploaderDropzone"] {
          background:#f8fafc !important; border:1px dashed #cbd5e1 !important; }
  [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stFileUploaderDropzoneInstructions"] span,
  [data-testid="stFileUploaderDropzoneInstructions"] small { color:#374151 !important; }
  [data-testid="stFileUploaderDropzone"] button {
          background:#ffffff !important; color:#374151 !important; border:1px solid #cbd5e1 !important; }

  .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p { color:#374151 !important; font-weight:600; }
  .stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab"]:hover p { color:#0f172a !important; }
  .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p { color:#7c3aed !important; }
  .stTabs [data-baseweb="tab-highlight"] { background:#7c3aed !important; }

  .info-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
               padding:16px 20px; margin-top:14px; box-shadow:0 1px 4px rgba(0,0,0,.05); }
  .info-card .ic-head { font-size:.75rem; font-weight:800; letter-spacing:.8px;
                        text-transform:uppercase; margin-bottom:10px; }
  .info-card .ic-head.why  { color:#7c3aed; }
  .info-card .ic-head.act  { color:#0891b2; }
  .info-card ul { margin:0; padding-left:18px; }
  .info-card ul li { color:#374151 !important; font-size:.9rem; line-height:1.6; margin-bottom:4px; }
  .info-card ul li:last-child { margin-bottom:0; }

  /* accordion detail sections (main panel only) */
  .detail-label { font-size:.72rem; font-weight:800; letter-spacing:.9px;
                  text-transform:uppercase; color:#94a3b8; margin:28px 0 10px; }
  section.main [data-testid="stExpander"] {
    background:#ffffff !important; border:1px solid #e2e8f0 !important;
    border-radius:14px !important; margin-bottom:8px; overflow:hidden;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
    transition:border-color .2s ease, box-shadow .2s ease; }
  section.main [data-testid="stExpander"]:has(details[open]) {
    border-color:#7c3aed !important;
    box-shadow:0 0 0 1px rgba(124,58,237,.25), 0 0 16px rgba(124,58,237,.10) !important; }
  section.main [data-testid="stExpander"] summary { background:#ffffff !important; }
  section.main [data-testid="stExpander"] summary p {
    color:#374151 !important; font-weight:700 !important; font-size:.93rem; }
  section.main [data-testid="stExpander"] summary:hover p { color:#7c3aed !important; }
  section.main [data-testid="stExpander"] details[open] summary p { color:#7c3aed !important; }
  section.main [data-testid="stExpanderToggleIcon"] svg { stroke:#7c3aed !important; }
"""


def build_styles(theme: str) -> str:
    css = _LIGHT_CSS if theme == "Light" else _DARK_CSS
    return f"<style>{css}</style>"


# --------------------------------------------------------------------------- #
# Apply theme (reads session_state set by the sidebar radio on previous run)
# --------------------------------------------------------------------------- #
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

st.markdown(build_styles(st.session_state["theme"]), unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🧾 PayOPS-AI</h1>'
    '<p>Agentic invoice processing — extract · read · verify · dedupe · match · decide</p></div>',
    unsafe_allow_html=True,
)

NODES = [
    ("Extract", "🧩",
     "Extracts raw content from a PDF or scanned image. Uses the native text layer for clean PDFs "
     "and falls back to OCR for scans. Returns the full extracted text together with an overall "
     "confidence score (0–1) that reflects the quality and legibility of the extraction."),
    ("Read", "🧠",
     "Sends the extracted text to an LLM which parses it into structured invoice fields — vendor, "
     "invoice number, PO numbers, dates, currency, subtotal, tax, total, and line items. "
     "Every field is tagged with its own confidence score (0–1) so downstream agents know "
     "exactly how certain the model is about each extracted value."),
    ("Verify", "✅",
     "Checks that the three mandatory fields — PO number, invoice number, and total amount — "
     "are all present and each scores at or above the confidence gate you have configured. "
     "Any field below the threshold causes the invoice to be routed for human review. "
     "Also flags overdue invoices as a non-blocking warning."),
    ("Match", "🔗", "Finds the PO by number, confirms the vendor, and checks the amount against the PO balance."),
    ("Decide", "⚖️", "Applies the rules — duplicate, tolerance, approval — and routes the invoice."),
    ("Summary", "📝", "Writes a one-paragraph, plain-English explanation of the call."),
]

OUTCOME = {
    "AUTO_APPROVE":           ("#22c55e", "✅", "Auto-approved",
        "All checks passed. This invoice is cleared for payment — no human action needed."),
    "APPROVE_WITH_EXCEPTION": ("#14b8a6", "✅", "Approved with exception",
        "Cleared for payment, but a small overage was absorbed within tolerance and logged for audit."),
    "ROUTE_FOR_REVIEW":       ("#f59e0b", "🔎", "Routed for review",
        "A human needs to check this before payment. See the reasons and required actions below."),
    "HOLD":                   ("#fb923c", "⏸", "On hold",
        "Payment is paused until a blocking issue (e.g. a missing or unapproved PO) is resolved."),
    "REJECT":                 ("#ef4444", "⛔", "Rejected",
        "This invoice will not be paid. See the reasons below — typically a confirmed duplicate."),
}

OUTCOME_INFO = {
    "AUTO_APPROVE": {
        "criteria": [
            "No duplicate found — invoice # + vendor not in the ledger",
            "All mandatory fields pass the confidence gate",
            "PO found in the master table and status is Approved",
            "Vendor on invoice matches PO vendor above the similarity threshold",
            "Invoice amount is within the PO's remaining balance",
        ],
        "next": [
            "Invoice is cleared for payment immediately",
            "Ledger is updated when 'Commit on approval' is enabled",
        ],
    },
    "APPROVE_WITH_EXCEPTION": {
        "criteria": [
            "All core checks pass (no duplicate, fields OK, PO approved, vendor match)",
            "Invoice amount slightly exceeds PO remaining balance",
            "Overage is within the configured tolerance band — max(% slider, flat floor ₹)",
        ],
        "next": [
            "Invoice proceeds to payment",
            "Overage is logged as a tolerance exception for audit trail",
        ],
    },
    "ROUTE_FOR_REVIEW": {
        "criteria": [
            "Mandatory fields below the confidence threshold, OR",
            "Vendor name doesn't match PO vendor (below similarity threshold), OR",
            "Invoice # found in ledger under a different vendor (suspected duplicate), OR",
            "Invoice amount exceeds PO balance beyond the tolerance band",
        ],
        "next": [
            "Held — no payment until a human resolves the flagged issue",
            "Correct or confirm the flagged fields, then re-run",
        ],
    },
    "HOLD": {
        "criteria": [
            "PO number from the invoice not found in the master table, OR",
            "PO found but its status is not 'Approved'",
        ],
        "next": [
            "Payment blocked until the PO issue is resolved",
            "Raise or approve the PO, then re-submit the invoice",
        ],
    },
    "REJECT": {
        "criteria": [
            "Exact duplicate confirmed — same vendor + same invoice # already in ledger",
            "Hard stop: no further checks are performed once a duplicate is confirmed",
        ],
        "next": [
            "Do not pay this invoice",
            "Contact the vendor — this is likely a re-billing or system error",
        ],
    },
}


def flow_html(states: list[str]) -> str:
    n = len(NODES)
    parts = ['<div class="flow">']
    for i, (name, ico, tip) in enumerate(NODES):
        align = "left" if i <= 1 else ("right" if i >= n - 2 else "center")
        card = (f'<div class="card card-{align}">'
                f'<div class="ct">{ico} {name}</div><div class="cb">{tip}</div></div>')
        parts.append(f'<div class="node {states[i]}"><div class="dot">{ico}</div>'
                     f'<div class="nlabel">{name}</div>{card}</div>')
        if i < n - 1:
            parts.append(f'<div class="conn {"on" if states[i] == "done" else ""}"></div>')
    parts.append("</div>")
    return "".join(parts)


def trace_html(rows: list[tuple[str, str]], fade: bool = False) -> str:
    body = "".join(f'<div class="row"><span class="ag">{a}</span>'
                   f'<span class="ms">{m}</span></div>' for a, m in rows)
    return f'<div class="trace{" fade" if fade else ""}">{body or "&nbsp;"}</div>'


def verdict_html(outcome: str) -> str:
    color, icon, label, desc = OUTCOME.get(outcome, ("#94a3b8", "•", outcome, ""))
    info = OUTCOME_INFO.get(outcome, {})
    crit_li = "".join(f"<li>{c}</li>" for c in info.get("criteria", []))
    next_li = "".join(f"<li>{n}</li>" for n in info.get("next", []))
    tooltip = (
        f'<div class="vinfo-wrap">'
        f'  <div class="vinfo-btn">ℹ</div>'
        f'  <div class="vinfo-card">'
        f'    <div class="vic-title">{label}</div>'
        f'    <div class="vic-sect"><div class="vic-lbl">Triggered when</div><ul>{crit_li}</ul></div>'
        f'    <div class="vic-sect"><div class="vic-lbl">What happens next</div><ul>{next_li}</ul></div>'
        f'  </div>'
        f'</div>'
    )
    return (
        f'<div class="vbox fade" style="background:{color}">'
        f'  <div class="vicon">{icon}</div>'
        f'  <div style="flex:1"><div class="vlabel">{label}</div>'
        f'  <div class="vsub">{outcome}</div>'
        f'  <div class="vexp">{desc}</div></div>'
        f'  {tooltip}'
        f'</div>'
    )


def fmt_amount(currency, value) -> str:
    """'INR' + '59000' → 'INR 59,000'; avoids 'INR INR' if the value already has it."""
    if value in (None, ""):
        return "—"
    cur = (str(currency) if currency else "").strip()
    raw = str(value).strip()
    body = raw
    try:
        num = float(raw.replace(",", "").replace(cur, "").strip())
        body = f"{num:,.0f}" if num == int(num) else f"{num:,.2f}"
    except (ValueError, TypeError):
        pass
    if cur and raw.upper().startswith(cur.upper()):
        cur = ""
    return f"{cur} {body}".strip()


_PILL = {"ok": "✓ Passed", "warn": "⚠ Flagged", "bad": "✗ Failed"}


def stage_table_html(doc, result, verdict, match, decision) -> str:
    """Per-agent status table: green = passed, amber = flagged, red = failed/blocked."""
    f = result.fields
    cur = f.currency.value

    ext_status = "ok" if doc.confidence >= 0.85 else "warn"
    ext = (ext_status, f"{doc.fmt.value.upper()} · {len(doc.text)} chars · "
                       f"{doc.confidence:.0%} text confidence")

    n_fields = sum(1 for ef in (f.vendor_name, f.invoice_number, f.total_amount) if ef.value)
    read = ("ok", f"{n_fields}/3 key fields read · invoice total {fmt_amount(cur, f.total_amount.value)}")

    if verdict.proceed:
        ver = ("ok", "All mandatory fields present and above the confidence gate")
    else:
        ver = ("bad", "Mandatory-field gate failed — " + "; ".join(verdict.issues))

    if not match.po_found:
        mat = ("bad", f"PO '{match.po_number_input}' not found in the master table")
    elif match.vendor_mismatch:
        mat = ("bad", f"Vendor mismatch — invoice '{match.vendor_input}' vs PO "
                      f"'{match.po_vendor}' (similarity {match.vendor_score:.0f}/100)")
    elif match.over_billing:
        mat = ("warn", f"PO matched, but billing exceeds tolerance · pending "
                       f"{fmt_amount(cur, match.pending_amount)}")
    else:
        mat = ("ok", f"PO {match.po_number_input} matched · vendor {match.vendor_score:.0f}/100 · "
                     f"pending {fmt_amount(cur, match.pending_amount)}")

    dec_cls = {"AUTO_APPROVE": "ok", "APPROVE_WITH_EXCEPTION": "ok",
               "ROUTE_FOR_REVIEW": "warn", "HOLD": "warn", "REJECT": "bad"}.get(decision.outcome, "warn")
    dup = decision.duplicate
    dec_detail = decision.outcome.replace("_", " ").title()
    if decision.reasons:
        dec_detail += " — " + decision.reasons[0]
    if not dup.is_clear:
        dec_detail += f" · {dup.status.replace('_', ' ').lower()}"
    dec = (dec_cls, dec_detail)

    summ = ("ok", "Plain-English recap generated for the approver")

    stages = [
        ("🧩", "Extract", ext),
        ("🧠", "Read", read),
        ("✅", "Verify", ver),
        ("🔗", "Match", mat),
        ("⚖️", "Decide", dec),
        ("📝", "Summary", summ),
    ]

    rows = []
    for ico, name, (status, detail) in stages:
        row_cls = "r-bad" if status == "bad" else ("r-warn" if status == "warn" else "r-ok")
        rows.append(
            f'<div class="srow {row_cls}">'
            f'  <div class="sico">{ico}</div>'
            f'  <div class="smid"><div class="snm">{name}</div><div class="sdt">{detail}</div></div>'
            f'  <div class="spill {status}">{_PILL[status]}</div>'
            f'</div>'
        )
    return '<div class="stage fade">' + "".join(rows) + "</div>"


def preview(path: Path):
    if path.suffix.lower() == ".pdf":
        d = fitz.open(str(path)); pix = d[0].get_pixmap(matrix=fitz.Matrix(2, 2)); d.close()
        st.image(pix.tobytes("png"), use_container_width=True)
    else:
        st.image(str(path), use_container_width=True)


def field_df(fields):
    rows = []
    def add(l, ef): rows.append({"Field": l, "Value": ef.value, "Confidence": float(ef.confidence) * 100})
    add("Vendor", fields.vendor_name); add("Invoice #", fields.invoice_number)
    for i, po in enumerate(fields.po_numbers): add(f"PO #{i+1}", po)
    add("Invoice date", fields.invoice_date); add("Due date", fields.due_date)
    add("Period start", fields.billing_period_start); add("Period end", fields.billing_period_end)
    add("Currency", fields.currency); add("Subtotal", fields.subtotal)
    add("Tax", fields.tax_total); add("Total", fields.total_amount)
    return pd.DataFrame(rows)


def reasons_html(reasons: list[str], actions: list[str]) -> str:
    why_items  = "".join(f"<li>{r}</li>" for r in reasons) if reasons else "<li>—</li>"
    act_items  = "".join(f"<li>{a}</li>" for a in actions) if actions else "<li>—</li>"
    return (
        '<div class="info-card fade">'
        '  <div class="ic-head why">🔍 Why this decision</div>'
        f' <ul>{why_items}</ul>'
        '</div>'
        '<div class="info-card fade">'
        '  <div class="ic-head act">⚡ Required actions</div>'
        f' <ul>{act_items}</ul>'
        '</div>'
    )


def edit_csv(label, path: Path, key: str):
    """Editable grid backed by a CSV: edit cells, add/delete rows, then Save."""
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True,
                            hide_index=True, key=key)
    if st.button(f"💾 Save {label.lower()}", key=key + "_save"):
        edited.to_csv(path, index=False)
        st.success(f"Saved {len(edited)} row(s) to {path.name}")
    return edited


def data_manager(scope: str):
    """Render both tables as editable grids. `scope` keeps widget keys unique."""
    st.markdown("##### 📦 PO master table")
    edit_csv("PO table", PO_CSV, key=f"po_{scope}")
    st.markdown("##### 🧾 Invoice ledger  ·  source of truth")
    edit_csv("ledger", LEDGER_CSV, key=f"ledger_{scope}")
    if st.button("🔄 Rebuild PO table from ledger", key=f"rebuild_{scope}",
                 help="Recompute billed / pending / billing-status columns from the ledger entries"):
        POStore().rebuild()
        st.success("PO table recomputed from the ledger.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Console")

    # # --- Theme toggle ---
    # st.radio("🎨 Theme", ["Dark", "Light"], key="theme", horizontal=True)
    # st.divider()

    # --- Upload / sample ---
    uploaded = st.file_uploader("Invoice", type=["pdf", "png", "jpg", "jpeg", "tiff", "webp"])
    sample = st.selectbox("Sample", ["—"] + sorted(str(p) for p in Path("data").glob("*.pdf")))

    # --- Document preview (shown as soon as a file is chosen) ---
    _preview_path = None
    if uploaded is not None:
        _preview_path = Path(tempfile.gettempdir()) / uploaded.name
        _preview_path.write_bytes(uploaded.getbuffer())
    elif sample and sample != "—":
        _preview_path = Path(sample)
    if _preview_path is not None:
        with st.expander("👁️ Preview document", expanded=False):
            try:
                if _preview_path.suffix.lower() == ".pdf":
                    _d = fitz.open(str(_preview_path))
                    _pix = _d[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    _d.close()
                    st.image(_pix.tobytes("png"), use_container_width=True)
                else:
                    st.image(str(_preview_path), use_container_width=True)
            except Exception as _e:
                st.caption(f"Preview unavailable: {_e}")

    st.divider()

    with st.expander("🎯 Field confidence gate", expanded=True):
        conf_threshold = st.slider(
            "Minimum confidence", 0.50, 1.0, 0.90, 0.01, format="%.2f",
            help=(
                "After the AI reads the invoice every extracted field gets a confidence "
                "score (0–1). Mandatory fields — invoice number, PO number, total amount — "
                "must ALL score at or above this threshold to pass the verification gate and "
                "move forward. Lower = more permissive; higher = stricter."
            ),
        )
    with st.expander("🔗 Vendor match threshold", expanded=True):
        vendor_threshold = st.slider(
            "Similarity score", 50, 100, 85, 1,
            help=(
                "The vendor name on the invoice is compared to the vendor on the matched PO "
                "using fuzzy text similarity (0–100). At or above this score → same vendor, "
                "no issue. Below it → vendor mismatch flagged and the invoice is routed for "
                "human review before payment."
            ),
        )
    with st.expander("💰 AP billing tolerance", expanded=True):
        tol_pct = st.slider(
            "Tolerance %", 0.0, 10.0, 2.0, 0.5, format="%.1f%%",
            help=(
                "Invoices sometimes exceed the approved PO amount by a small margin due to "
                "rounding, freight, or minor price drift. This sets the % band above the "
                "approved amount that is still acceptable. The actual allowed overage is the "
                "GREATER of this % or the flat floor below — so small POs still get a "
                "meaningful buffer. Within band → Approved with exception; beyond → Review."
            ),
        )
        tol_abs = st.number_input(
            "Tolerance floor (₹)", value=500, step=100,
            help=(
                "The minimum flat rupee buffer allowed above a PO's approved amount, "
                "regardless of the % tolerance. For example with 2% tolerance and a ₹10,000 "
                "PO, 2% = ₹200 which is very tight — so the floor kicks in and allows up to "
                "₹500 instead. Whichever is larger (% amount or this floor) is used."
            ),
        )

    st.divider()
    commit = st.toggle("Commit on approval", value=False)
    run_btn = st.button("🚀 Run pipeline", type="primary", use_container_width=True)


def resolve_path():
    if uploaded is not None:
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.getbuffer()); return tmp
    if sample and sample != "—":
        return Path(sample)
    return None


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if not run_btn:
    st.markdown(flow_html(["pending"] * len(NODES)), unsafe_allow_html=True)
    st.caption("Pick an invoice on the left and hit **Run pipeline** to watch the agents work.")
    with st.expander("🗂️ Manage data tables — PO master & invoice ledger", expanded=False):
        st.caption("Edit cells directly, add or delete rows, then **Save**. "
                   "Saved changes are used on the next pipeline run.")
        data_manager("home")
    st.stop()

path = resolve_path()
if path is None:
    st.warning("Upload an invoice or choose a sample first.")
    st.stop()

states = ["pending"] * len(NODES)
flow_ph = st.empty()
trace_ph = st.empty()
trace: list[tuple[str, str]] = []
flow_ph.markdown(flow_html(states), unsafe_allow_html=True)
trace_ph.markdown(trace_html(trace), unsafe_allow_html=True)

store = POStore(tolerance=tol_pct / 100.0, tolerance_abs=float(tol_abs))


def step(i, agent, msg_fn, fn):
    states[i] = "active"; flow_ph.markdown(flow_html(states), unsafe_allow_html=True)
    trace.append((agent, "…")); trace_ph.markdown(trace_html(trace), unsafe_allow_html=True)
    time.sleep(0.15)
    out = fn()
    trace[-1] = (agent, msg_fn(out))
    states[i] = "done"; flow_ph.markdown(flow_html(states), unsafe_allow_html=True)
    trace_ph.markdown(trace_html(trace), unsafe_allow_html=True)
    time.sleep(0.12)
    return out

doc = step(0, "Extract", lambda d: f"{d.fmt.value} · {d.confidence:.0%} conf · {len(d.text)} chars",
           lambda: extract(path))
result = step(1, "Read", lambda r: f"{sum(1 for _ in [r.fields.vendor_name,r.fields.invoice_number,r.fields.total_amount])} key fields · total {r.fields.total_amount.value}",
              lambda: ExtractionAgent().run(doc))
verdict = step(2, "Verify", lambda v: "passed gate" if v.proceed else f"blocked · {len(v.issues)} issue(s)",
               lambda: VerificationAgent(threshold=conf_threshold).run(result))
match = step(3, "Match", lambda m: (f"{m.po_number_input} · " + ("vendor ✓" if not m.vendor_mismatch else "vendor ✗")
             + (" · over-bill" if m.over_billing else "")) if m.po_found else "PO not found",
             lambda: POMatchingAgent(store=store, vendor_threshold=float(vendor_threshold)).run(result))
decision = step(4, "Decide", lambda d: d.outcome + ("" if d.duplicate.is_clear else f" · {d.duplicate.status.replace('_', ' ').lower()}"),
                lambda: DecisionAgent(store=store).decide(result, verdict, match))
dup = decision.duplicate
summary = step(5, "Summary", lambda s: "recap ready",
               lambda: SummaryAgent().run(result, verdict, match, decision))

# ---- Verdict (+ why / actions) on the left, per-agent stage table on the right ----
trace_ph.empty()
cL, cR = st.columns([1, 1], gap="large")
with cL:
    st.markdown(verdict_html(decision.outcome), unsafe_allow_html=True)
    st.markdown(reasons_html(decision.reasons, decision.required_actions), unsafe_allow_html=True)
with cR:
    st.markdown(stage_table_html(doc, result, verdict, match, decision), unsafe_allow_html=True)

# ---- Commit (before tables so they reflect new state) ----
if commit and decision.approved and match.po_found:
    applied = store.apply_invoice(
        invoice_number=result.fields.invoice_number.value, po_number=match.po_number_input,
        vendor=match.vendor_matched, amount=match.invoice_amount, status="Approved")
    (st.warning if applied.condition == "DUPLICATE" else st.success)(
        f"{applied.condition}: {applied.message}")
elif commit:
    st.caption("Commit skipped — decision was not an approval.")

# ---- AI summary ----
st.subheader("📝 AI summary")
st.info(summary)

# ---- Details ----
st.markdown('<div class="detail-label">📊 Details</div>', unsafe_allow_html=True)

with st.expander("📋 Extracted fields", expanded=True):
    st.dataframe(field_df(result.fields), hide_index=True, use_container_width=True,
                 column_config={"Confidence": st.column_config.ProgressColumn(
                     "Confidence", min_value=0, max_value=100, format="%.0f%%")})
    if result.fields.line_items:
        st.markdown("**Line items**")
        st.dataframe(pd.DataFrame([{
            "Description": x.description.value, "Qty": x.quantity.value,
            "Unit price": x.unit_price.value, "Line total": x.line_total.value}
            for x in result.fields.line_items]), hide_index=True, use_container_width=True)

with st.expander("🔗 PO match & verification"):
    st.code(match.summary())
    if verdict.issues:
        st.markdown("**Verification issues**")
        for i in verdict.issues: st.markdown(f"- {i}")
    if decision.past_due:
        st.markdown("**Flags**")
        st.markdown(f"- ⚠ Invoice is **past due** by {decision.days_overdue} day(s)")

with st.expander("🗂️ Data tables"):
    st.caption("Edit cells, add or delete rows, then **Save**. Saved changes are used on the next run.")
    data_manager("tab")

with st.expander("📄 Document preview"):
    preview(path)

with st.expander("🔤 Raw extracted text"):
    st.text(doc.text)
