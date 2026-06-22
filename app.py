"""
AP Autopilot — Agent Command Center (Streamlit).

    streamlit run app.py

Dark-neon UI: watch the agent pipeline light up node-by-node, stream each agent's
conclusion, and land on a glowing routed verdict + AI summary.
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

st.set_page_config(page_title="AP Autopilot", page_icon="🧾", layout="wide")

# --------------------------------------------------------------------------- #
# Dark-neon theme
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(1200px 600px at 20% -10%, #1b1245 0%, #0a0a14 55%) fixed; color:#e5e7eb; }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] { background: #0e1020; border-right: 1px solid #23263a; }
      h1,h2,h3,h4 { color:#f3f4f6 !important; }

      /* readable text on dark everywhere */
      .stApp p, .stApp li, .stApp span, .stMarkdown, .stCaption,
      [data-testid="stWidgetLabel"] p, label, [data-testid="stMarkdownContainer"] p { color:#cbd0e0 !important; }
      [data-testid="stSidebar"] * { color:#c3c8da; }
      [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color:#7f859c !important; }
      [data-testid="stMetricLabel"] { color:#9aa0b4 !important; }
      [data-testid="stMetricValue"] { color:#f3f4f6 !important; }

      /* fade-in on results */
      @keyframes fadeUp { from { opacity:0; transform: translateY(10px); }
                          to   { opacity:1; transform: none; } }
      .fade { animation: fadeUp .55s cubic-bezier(.2,.7,.2,1) both; }
      section.main .block-container { animation: fadeUp .45s ease both; }

      .hero { padding: 8px 2px 2px; }
      .hero h1 { font-size: 2.1rem; font-weight: 900; letter-spacing:-.5px;
                 background: linear-gradient(90deg,#a78bfa,#22d3ee,#ec4899);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
      .hero p { color:#9aa0b4; margin-top:-4px; }

      /* agent flow */
      .flow { display:flex; align-items:center; justify-content:space-between;
              gap:4px; padding:18px 10px; margin:6px 0 4px; overflow:visible;
              background:#0e1124; border:1px solid #23263a; border-radius:18px; }
      .node { position:relative; display:flex; flex-direction:column; align-items:center; gap:8px;
              width:92px; text-align:center; cursor:help; }

      /* hover flashcard */
      .card { position:absolute; top:118%; width:236px; padding:14px 16px; border-radius:14px;
              background:rgba(16,19,38,.98); border:1px solid #3a3f63; text-align:left;
              box-shadow:0 18px 42px rgba(0,0,0,.55), 0 0 26px rgba(139,92,246,.30);
              opacity:0; pointer-events:none; z-index:60;
              transition:opacity .18s ease, transform .18s ease; }
      .card .ct { font-weight:800; color:#a78bfa; font-size:.92rem; margin-bottom:5px; }
      .card .cb { font-size:.82rem; color:#cbd0e0; line-height:1.4; }
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

      /* trace */
      .trace { background:#0e1124; border:1px solid #23263a; border-radius:16px; padding:14px 16px; min-height:210px; }
      .trace .row { display:flex; gap:10px; padding:7px 0; border-bottom:1px dashed #20233a; font-size:.92rem; }
      .trace .row:last-child { border-bottom:none; }
      .trace .ag { color:#67e8f9; font-weight:700; min-width:88px; }
      .trace .ms { color:#cbd0e0; }

      /* verdict — solid colour by outcome, black text */
      .vbox { display:flex; align-items:center; gap:18px; padding:26px 28px; border-radius:18px;
              color:#0a0a0a; box-shadow:0 12px 34px rgba(0,0,0,.40); }
      .vbox .vicon  { font-size:2.6rem; }
      .vbox .vlabel { font-size:1.95rem; font-weight:900; color:#0a0a0a !important; line-height:1.05; }
      .vbox .vsub   { font-size:.82rem; font-weight:800; letter-spacing:.6px;
                      color:#0a0a0a !important; opacity:.65; margin-top:2px; }
      .vbox .vexp   { font-size:.92rem; font-weight:600; color:#0a0a0a !important;
                      opacity:.82; margin-top:9px; line-height:1.38; }

      /* agent stage table — per-stage status, readable */
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

      /* file uploader — dark dropzone with readable text */
      [data-testid="stFileUploaderDropzone"] {
              background:#141833 !important; border:1px dashed #3a3f63 !important; }
      [data-testid="stFileUploaderDropzone"] *,
      [data-testid="stFileUploaderDropzoneInstructions"] span,
      [data-testid="stFileUploaderDropzoneInstructions"] small { color:#cbd0e0 !important; }
      [data-testid="stFileUploaderDropzone"] button {
              background:#1a1f3a !important; color:#e5e7eb !important; border:1px solid #4b4f73 !important; }

      /* tabs — readable labels */
      .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] p { color:#cbd0e0 !important; font-weight:600; }
      .stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab"]:hover p { color:#e5e7eb !important; }
      .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p { color:#a78bfa !important; }
      .stTabs [data-baseweb="tab-highlight"] { background:#a78bfa !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🧾 AP Autopilot</h1>'
    '<p>Agentic invoice processing — extract · read · verify · dedupe · match · decide</p></div>',
    unsafe_allow_html=True,
)

NODES = [
    ("Extract", "🧩", "PDF or image? Pulls the text — text layer for clean PDFs, OCR for scans — with a confidence score."),
    ("Read", "🧠", "An LLM turns the raw text into structured fields, each tagged with its own confidence."),
    ("Verify", "✅", "Checks the must-have fields are present and trustworthy. Flags overdue invoices."),
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
    return (
        f'<div class="vbox fade" style="background:{color}">'
        f'  <div class="vicon">{icon}</div>'
        f'  <div><div class="vlabel">{label}</div>'
        f'  <div class="vsub">{outcome}</div>'
        f'  <div class="vexp">{desc}</div></div>'
        f'</div>'
    )


def fmt_amount(currency, value) -> str:
    """'INR' + '59000' → 'INR 59,000'; avoids 'INR INR' if the value already has it."""
    if value in (None, ""):
        return "—"
    cur = (str(currency) if currency else "").strip()
    raw = str(value).strip()
    body = raw
    try:                                        # add thousands separators when numeric
        num = float(raw.replace(",", "").replace(cur, "").strip())
        body = f"{num:,.0f}" if num == int(num) else f"{num:,.2f}"
    except (ValueError, TypeError):
        pass
    if cur and raw.upper().startswith(cur.upper()):   # value already carries the currency
        cur = ""
    return f"{cur} {body}".strip()


_PILL = {"ok": "✓ Passed", "warn": "⚠ Flagged", "bad": "✗ Failed"}


def stage_table_html(doc, result, verdict, match, decision) -> str:
    """Per-agent status table: green = passed, amber = flagged, red = failed/blocked."""
    f = result.fields
    cur = f.currency.value

    # --- compute (status, detail) for each stage ---
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
    uploaded = st.file_uploader("Invoice", type=["pdf", "png", "jpg", "jpeg", "tiff", "webp"])
    sample = st.selectbox("…or a sample", ["—"] + sorted(str(p) for p in Path("data").glob("*.pdf")))
    st.divider()
    conf_threshold = st.slider("Field confidence gate", 0.50, 1.0, 0.90, 0.01)
    vendor_threshold = st.slider("Vendor match threshold", 50, 100, 85, 1)
    tol_pct = st.slider("AP tolerance %", 0.0, 0.10, 0.02, 0.005)
    tol_abs = st.number_input("Tolerance floor", value=500, step=100)
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

store = POStore(tolerance=tol_pct, tolerance_abs=float(tol_abs))


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
    st.markdown("**Why**")
    for r in decision.reasons:
        st.markdown(f"- {r}")
    st.markdown("**Required actions**")
    for a in (decision.required_actions or ["—"]):
        st.markdown(f"- {a}")
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
t1, t2, t3, t4, t5 = st.tabs(["📋 Fields", "🔗 PO match", "🗂️ Tables", "📄 Document", "🔤 Raw text"])
with t1:
    st.dataframe(field_df(result.fields), hide_index=True, use_container_width=True,
                 column_config={"Confidence": st.column_config.ProgressColumn(
                     "Confidence", min_value=0, max_value=100, format="%.0f%%")})
    if result.fields.line_items:
        st.markdown("**Line items**")
        st.dataframe(pd.DataFrame([{
            "Description": x.description.value, "Qty": x.quantity.value,
            "Unit price": x.unit_price.value, "Line total": x.line_total.value}
            for x in result.fields.line_items]), hide_index=True, use_container_width=True)
with t2:
    st.code(match.summary())
    if verdict.issues:
        st.markdown("**Verification issues**")
        for i in verdict.issues: st.markdown(f"- {i}")
    if decision.past_due:
        st.markdown("**Flags**")
        st.markdown(f"- ⚠ Invoice is **past due** by {decision.days_overdue} day(s)")
with t3:
    st.caption("Edit cells, add or delete rows, then **Save**. Saved changes are used on the next run.")
    data_manager("tab")
with t4:
    preview(path)
with t5:
    st.text(doc.text)
