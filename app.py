"""
Crude → India impact terminal (Streamlit)
Live global crude, USD/INR, Indian indices and sector stocks, run through the
crude → rupee → inflation → RBI → equities transmission logic.

Data is fetched server-side (yfinance, Google Finance fallback), so there are no
browser CORS limits. Auto-refreshes every 60 seconds. No inputs needed.
"""
import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
st.set_page_config(page_title="Crude to India impact", page_icon="🛢️", layout="wide")

# ───────────────────────── Config ─────────────────────────
MACRO = dict(
    import_mbpd=4.8,        # crude imports, million barrels/day
    gdp_usd_bn=4100,        # nominal GDP, USD bn
    cpi_bps_per_10pct=25,   # CPI bps per 10% rise in rupee crude
    baseline_days=60,       # baseline window for "vs normal"
)
BBL_BN = MACRO["import_mbpd"] * 365 / 1000  # bn barrels per year
REFRESH = "60s"

# id, name, yahoo, unit, inverse(up = bad for market), google finance fallback
GLOBAL = [
    ("brent", "Brent crude", "BZ=F", "$", True, "BZW00:NYMEX"),
    ("wti", "WTI crude", "CL=F", "$", True, "CLW00:NYMEX"),
    ("usdinr", "USD / INR", "INR=X", "₹", True, "USD-INR"),
    ("dxy", "Dollar index", "DX-Y.NYB", "", True, None),
]
INDICES = [
    ("nifty", "Nifty 50", "^NSEI", False, "NIFTY_50:INDEXNSE"),
    ("banknifty", "Nifty Bank", "^NSEBANK", False, "NIFTY_BANK:INDEXNSE"),
    ("sensex", "Sensex", "^BSESN", False, "SENSEX:INDEXBOM"),
    ("niftyauto", "Nifty Auto", "^CNXAUTO", False, None),
    ("niftyenergy", "Nifty Energy", "^CNXENERGY", False, None),
    ("indiavix", "India VIX", "^INDIAVIX", True, None),
]
SECTORS = [
    ("Aviation", -1.0, False, "Jet fuel is the single largest cost line for airlines", ["INDIGO"]),
    ("Paints", -0.8, False, "Crude derivatives are a large share of raw material cost", ["ASIANPAINT", "BERGEPAINT"]),
    ("Logistics", -0.5, False, "Diesel is the biggest variable cost", ["CONCOR", "BLUEDART"]),
    ("Automobiles", -0.6, False, "Higher running costs dent demand; plastics and rubber inputs rise", ["MARUTI", "M&M", "HEROMOTOCO"]),
    ("Upstream E&P", 0.9, False, "Better realisation per barrel lifts revenue", ["ONGC", "OIL"]),
    ("Oil marketing (OMCs)", -0.6, True, "Retail price caps squeeze margins; inventory gains can offset", ["BPCL", "HINDPETRO", "IOC"]),
]
STOCKS = [s for sec in SECTORS for s in sec[4]]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}

# ───────────────────────── Data ─────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(tickers: tuple) -> dict:
    """Daily closes (4 months) for every ticker in one batched yfinance call."""
    out = {}
    try:
        df = yf.download(list(tickers), period="4mo", interval="1d", group_by="ticker",
                         auto_adjust=False, progress=False, threads=True)
    except Exception:
        return out
    for t in tickers:
        try:
            s = df[t]["Close"] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
            s = s.dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[~s.index.duplicated(keep="last")]
            if len(s):
                out[t] = s.astype(float)
        except Exception:
            pass
    return out


@st.cache_data(ttl=60, show_spinner=False)
def google_quote(gsym: str):
    """Fallback: price + previous close scraped from the Google Finance quote page."""
    try:
        html = requests.get(f"https://www.google.com/finance/quote/{gsym}?hl=en", headers=UA, timeout=10).text
        lp = re.search(r'data-last-price="([\d.]+)"', html)
        if not lp:
            return None
        pc = None
        tail = html.split("Previous close", 1)
        if len(tail) > 1:
            m = re.search(r">\s*[^\d<>]{0,4}([\d,]+(?:\.\d+)?)\s*<", tail[1][:600])
            pc = float(m.group(1).replace(",", "")) if m else None
        return float(lp.group(1)), pc
    except Exception:
        return None


def from_series(s: pd.Series, src="Yahoo Finance"):
    price = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) > 1 else math.nan
    ago = lambda n: float(s.iloc[-1 - n]) if len(s) > n else float(s.iloc[0])
    return dict(price=price, prev=prev, pct=(price / prev - 1) * 100 if prev else math.nan,
                pct5=(price / ago(5) - 1) * 100, pct1m=(price / ago(21) - 1) * 100,
                series=s, asof=s.index[-1].strftime("%d %b"), src=src)


def from_google(gsym):
    q = google_quote(gsym) if gsym else None
    if not q:
        return None
    p, pc = q
    return dict(price=p, prev=pc if pc else math.nan,
                pct=(p / pc - 1) * 100 if pc else math.nan, pct5=math.nan, pct1m=math.nan,
                series=None, asof="live", src="Google Finance")


def load_all():
    tickers = tuple([g[2] for g in GLOBAL] + [i[2] for i in INDICES] + [s + ".NS" for s in STOCKS])
    H = fetch_history(tickers)
    g = {gid: (from_series(H[y]) if y in H else from_google(gf)) for gid, _, y, _, _, gf in GLOBAL}
    idx = {iid: (from_series(H[y]) if y in H else from_google(gf)) for iid, _, y, _, gf in INDICES}
    stk = {s: (from_series(H[s + ".NS"]) if s + ".NS" in H else from_google(f"{s}:NSE")) for s in STOCKS}
    return g, idx, stk


# ───────────────────────── Analytics ─────────────────────────
def ok(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def beta_corr(a: pd.Series, b: pd.Series):
    if a is None or b is None:
        return math.nan, math.nan
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(df) < 15:
        return math.nan, math.nan
    r = np.log(df / df.shift(1)).dropna()
    cov = np.cov(r.iloc[:, 0], r.iloc[:, 1])
    return cov[0, 1] / cov[1, 1], cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])


def baselines(g):
    b, f = g["brent"], g["usdinr"]
    if b.get("series") is not None and len(b["series"]) > 2:
        bs = b["series"].iloc[:-1].tail(MACRO["baseline_days"])
        base_brent = float(bs.mean())
        if f.get("series") is not None:
            j = pd.concat([bs, f["series"]], axis=1, join="inner").dropna()
            base_inr = float((j.iloc[:, 0] * j.iloc[:, 1]).mean()) if len(j) else base_brent * f["price"]
        else:
            base_inr = base_brent * f["price"]
    else:
        base_brent = b["prev"] if ok(b["prev"]) else b["price"]
        base_inr = base_brent * f["price"]
    return base_brent, base_inr


def macro(brent, fx, base_brent, base_inr):
    extra = (brent - base_brent) * BBL_BN
    inr_crude = brent * fx
    vs_base = (inr_crude / base_inr - 1) * 100
    cpi = vs_base / 10 * MACRO["cpi_bps_per_10pct"]
    rbi = ("Less room to cut", "bad") if cpi > 15 else ("More room to cut", "good") if cpi < -15 else ("Little change", "flat")
    return dict(extra=extra, cad=extra / MACRO["gdp_usd_bn"] * 100, inr_crude=inr_crude,
                vs_base=vs_base, cpi=cpi, rbi=rbi, per_l=inr_crude / 159)


def verdict(s):
    if s <= -35: return "Strong headwind", "bad", "Bearish"
    if s <= -12: return "Headwind", "bad", "Mildly bearish"
    if s < 12: return "Neutral", "flat", "Neutral"
    if s < 35: return "Tailwind", "good", "Mildly bullish"
    return "Strong tailwind", "good", "Bullish"


clamp = lambda v, a, b: max(a, min(b, v))
n0 = lambda v: v if ok(v) else 0.0


def tone(v, inv=False, th=0.01):
    if not ok(v) or abs(v) < th:
        return "flat"
    return "good" if (v > 0) != inv else "bad"


def fmt(v, d=2):
    return "—" if not ok(v) else f"{v:,.{d}f}"


def sgn(v, d=2):
    return "—" if not ok(v) else ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):,.{d}f}"


def pct(v, d=2):
    return "—" if not ok(v) else sgn(v, d) + "%"


# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
.block-container{padding-top:1.4rem;max-width:1400px}
.chain{display:grid;grid-template-columns:repeat(6,1fr);border:1px solid #24505B;border-radius:10px;overflow:hidden;background:#14343D}
.node{position:relative;padding:14px 16px 16px;border-right:1px solid #24505B}
.node:last-child{border-right:0;background:#183D47}
.node .bar{position:absolute;left:0;top:0;height:3px;width:100%;background:#9FB3B8}
.node.bad .bar{background:#E5604F}.node.good .bar{background:#3DBE8E}
.node .st{font-size:12px;color:#8FAAB0}.node .tt{font-weight:600;margin:2px 0 8px}
.node .v{font-size:26px;font-weight:700;line-height:1.1}
.node .sb{font-size:12.5px;color:#8FAAB0;margin-top:6px;line-height:1.35}
.bad{color:#E5604F}.good{color:#3DBE8E}.flat{color:#9FB3B8}
.comp{display:grid;grid-template-columns:230px 1fr 56px;gap:10px;align-items:center;margin:6px 0;font-size:14px}
.track{position:relative;height:10px;background:#0E252D;border-radius:5px}
.track:before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#8FAAB0}
.fill{position:absolute;top:0;bottom:0;border-radius:5px}
@media(max-width:900px){.chain{grid-template-columns:repeat(2,1fr)}.node{border-bottom:1px solid #24505B}.comp{grid-template-columns:1fr 50px}.comp .track{grid-column:1/-1}}
</style>
""", unsafe_allow_html=True)

st.title("Crude to India impact terminal")
st.caption("How today's oil move flows through the rupee, inflation and RBI into Indian indices and sectors. "
           "Auto-refreshes every 60 seconds.")


# ───────────────────────── Dashboard ─────────────────────────
@st.fragment(run_every=REFRESH)
def dashboard():
    with st.spinner("Fetching live prices…"):
        g, idx, stk = load_all()
    now = datetime.now(IST).strftime("%d %b %Y, %H:%M:%S IST")
    if not g.get("brent") or not g.get("usdinr"):
        st.error(f"Couldn't load Brent or USD/INR from Yahoo or Google Finance ({now}). Retrying automatically in 60 seconds.")
        return

    b, fx = g["brent"], g["usdinr"]
    b_prev = b["prev"] if ok(b["prev"]) else b["price"]
    f_prev = fx["prev"] if ok(fx["prev"]) else fx["price"]
    inr_now, inr_prev = b["price"] * fx["price"], b_prev * f_prev
    day_inr = (inr_now / inr_prev - 1) * 100
    base_brent, base_inr = baselines(g)
    M = macro(b["price"], fx["price"], base_brent, base_inr)
    nifty = idx.get("nifty")
    for iid, v in idx.items():
        if v and v.get("series") is not None and b.get("series") is not None:
            v["beta"], v["corr"] = beta_corr(v["series"], b["series"])

    comps = [
        ("Crude move today, in rupees", -clamp(day_inr * 12, -35, 35)),
        ("Brent trend, last 5 sessions", -clamp(n0(b["pct5"]) * 3, -25, 25)),
        (f"Rupee crude vs {MACRO['baseline_days']}-day average", -clamp(M["vs_base"], -20, 20)),
        ("Rupee move today", -clamp(n0(fx["pct"]) * 40, -20, 20)),
    ]
    score = clamp(sum(c[1] for c in comps), -100, 100)
    vt, vtone, vdir = verdict(score)

    st.caption(f"Last update {now}")

    # Transmission chain
    fx_t = tone(fx["pct"], True, 0.02)
    fx_note = {"bad": "Weaker rupee, FII outflow risk", "good": "Stronger rupee, supports FII flows"}.get(fx_t, "Stable")
    nodes = [
        (1, "Crude in rupees", f"₹{fmt(inr_now, 0)}", f"{pct(day_inr)} today<br>{pct(M['vs_base'], 1)} vs {MACRO['baseline_days']}-day avg", tone(day_inr, True, 0.1)),
        (2, "Import bill", f"{'+' if M['extra'] >= 0 else '−'}${fmt(abs(M['extra']), 1)}bn", f"a year vs average Brent<br>CAD {sgn(M['cad'])}% of GDP", tone(M["extra"], True, 1)),
        (3, "Rupee", f"₹{fmt(fx['price'])}", f"{pct(fx['pct'])} today<br>{fx_note}", fx_t),
        (4, "Inflation", f"{sgn(M['cpi'], 0)} bps", "est. CPI pass-through<br>from fuel and logistics", tone(M["cpi"], True, 5)),
        (5, "RBI room", M["rbi"][0], "rate-cut headroom<br>from the crude channel", M["rbi"][1]),
        (6, "Nifty and Sensex", vdir, f"pressure score {sgn(score, 0)}<br>Nifty {pct(nifty['pct'] if nifty else None)} today", vtone),
    ]
    st.subheader("Transmission chain")
    st.markdown('<div class="chain">' + "".join(
        f'<div class="node {t}"><div class="bar"></div><div class="st">{n}</div><div class="tt">{ttl}</div>'
        f'<div class="v {t}" style="{"font-size:20px" if n == 5 else ""}">{v}</div><div class="sb">{sb}</div></div>'
        for n, ttl, v, sb, t in nodes) + "</div>", unsafe_allow_html=True)

    def metric(col, name, q, unit="", inv=False, extra=""):
        with col:
            if not q:
                st.metric(name, "—", help="Not available right now")
                return
            st.metric(name, f"{unit}{fmt(q['price'])}",
                      delta=None if not ok(q["pct"]) else f"{sgn(q['price'] - q['prev'])} ({pct(q['pct'])})",
                      delta_color="inverse" if inv else "normal",
                      help=f"5D {pct(q['pct5'], 1)} · 1M {pct(q['pct1m'], 1)}{extra} · {q['src']}, {q['asof']}")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Global prices")
        cols = st.columns(4)
        for c, (gid, name, _, unit, inv, _) in zip(cols, GLOBAL):
            metric(c, name, g.get(gid), unit, inv)
        st.subheader("Indian prices")
        w = g.get("wti")
        cols = st.columns(3)
        if w:
            wp = w["prev"] if ok(w["prev"]) else w["price"]
            mcx = dict(price=w["price"] * fx["price"], prev=wp * f_prev, pct=(w["price"] * fx["price"] / (wp * f_prev) - 1) * 100,
                       pct5=math.nan, pct1m=math.nan, src="Estimated as WTI × USD/INR", asof="live")
        else:
            mcx = None
        metric(cols[0], "MCX crude est. (₹/bbl)", mcx, "₹", True)
        metric(cols[1], "Brent in rupees (₹/bbl)", dict(price=inr_now, prev=inr_prev, pct=day_inr, pct5=math.nan, pct1m=math.nan, src="Brent × USD/INR", asof="live"), "₹", True)
        metric(cols[2], "Crude per litre", dict(price=M["per_l"], prev=inr_prev / 159, pct=day_inr, pct5=math.nan, pct1m=math.nan, src="before refining and taxes", asof="live"), "₹", True)
    with right:
        st.subheader("Indices")
        cols = st.columns(3)
        for k, (iid, name, _, inv, _) in enumerate(INDICES):
            q = idx.get(iid)
            extra = f" · β {fmt(q.get('beta'))} ρ {fmt(q.get('corr'))} vs Brent (3M)" if q and ok(q.get("beta")) else ""
            metric(cols[k % 3], name, q, "", inv, extra)
        st.caption("Hover or tap ⓘ on a card for 5-day and 1-month change, and each index's 3-month beta and correlation to Brent.")

    # Score
    st.subheader("Crude pressure on Indian equities")
    reasons = [
        f"Crude in rupee terms is {'up' if day_inr >= 0 else 'down'} {fmt(abs(day_inr))}% today "
        f"(Brent {pct(b['pct'])}, rupee {'weaker' if n0(fx['pct']) >= 0 else 'stronger'} by {fmt(abs(n0(fx['pct'])))}%).",
        f"At ${fmt(b['price'])} Brent, India's annual oil import bill runs about ${fmt(abs(M['extra']), 1)}bn "
        f"{'higher' if M['extra'] >= 0 else 'lower'} than at the {MACRO['baseline_days']}-day average of ${fmt(base_brent)}.",
        (f"That adds roughly {fmt(M['cpi'], 0)} bps to CPI, leaving the RBI less room to cut rates." if M["cpi"] > 15 else
         f"That takes roughly {fmt(-M['cpi'], 0)} bps off CPI, giving the RBI more room to cut rates." if M["cpi"] < -15 else
         "The inflation effect is small, so the RBI outlook is largely unchanged by crude."),
    ]
    if nifty and ok(nifty["pct"]) and abs(score) >= 12:
        agree = (nifty["pct"] > 0) == (score > 0)
        reasons.append(f"Nifty is {'up' if nifty['pct'] > 0 else 'down'} {fmt(abs(nifty['pct']))}%, "
                       + ("in line with the crude signal." if agree else "against the crude signal, so other drivers are dominating today."))
    if nifty and ok(nifty.get("corr")):
        c = nifty["corr"]
        reasons.append(f"Over 3 months, Nifty's daily correlation with Brent is {fmt(c)} ("
                       + ("weak; crude has not been a major driver lately)." if abs(c) < 0.2 else
                          "crude up has tended to mean Nifty down)." if c < 0 else
                          "positive; both have been moving on global risk appetite)."))
    bars = "".join(
        f'<div class="comp"><span>{n}</span><div class="track"><div class="fill" style="left:{50 - abs(v) / 40 * 50 if v < 0 else 50}%;'
        f'width:{abs(v) / 40 * 50}%;background:{"#E5604F" if v < 0 else "#3DBE8E"}"></div></div>'
        f'<span class="{tone(v)}" style="text-align:right">{sgn(v, 0)}</span></div>' for n, v in comps)
    st.markdown(
        f'<div style="display:flex;gap:14px;align-items:baseline;flex-wrap:wrap"><span class="{vtone}" style="font-size:40px;font-weight:700">{sgn(score, 0)}</span>'
        f'<span class="{vtone}" style="font-size:18px;font-weight:600">{vt}</span><span style="color:#8FAAB0;font-size:13px">'
        f'Scale −100 (strong headwind) to +100 (strong tailwind). Direction: {vdir}</span></div>{bars}'
        + "<ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>", unsafe_allow_html=True)

    # Sectors
    st.subheader("Sector impact")
    n_pct = nifty["pct"] if nifty and ok(nifty["pct"]) else 0.0
    rows = []
    for name, sens, mixed, why, syms in SECTORS:
        vals = [(s, stk.get(s)["pct"] if stk.get(s) else math.nan) for s in syms]
        got = [v for _, v in vals if ok(v)]
        avg = sum(got) / len(got) if got else math.nan
        rel = avg - n_pct if ok(avg) else math.nan
        flat = abs(day_inr) < 0.3
        exp = sens * day_inr
        reading = "Crude flat, no signal" if flat or not ok(rel) else ("Following crude" if (rel > 0) == (exp > 0) else "Ignoring crude")
        rows.append({"Sector": name, "Why": why, "Stocks today": ", ".join(f"{s} {pct(v)}" for s, v in vals),
                     "Expected from crude": "Neutral" if flat else ("Negative" if exp < 0 else "Positive") + (" (mixed)" if mixed else ""),
                     "Sector avg %": avg, "vs Nifty %": rel, "Reading": reading})
    df = pd.DataFrame(rows)
    colr = lambda v: "" if not ok(v) or abs(v) < 0.01 else f"color:{'#3DBE8E' if v > 0 else '#E5604F'}"
    txt = lambda v: {"Negative": "color:#E5604F", "Positive": "color:#3DBE8E", "Following crude": "color:#3DBE8E",
                     "Ignoring crude": "color:#E5604F"}.get(str(v).replace(" (mixed)", ""), "color:#9FB3B8")
    st.dataframe(df.style.map(colr, subset=["Sector avg %", "vs Nifty %"]).map(txt, subset=["Expected from crude", "Reading"])
                 .format({"Sector avg %": lambda v: pct(v), "vs Nifty %": lambda v: pct(v)}),
                 hide_index=True, width="stretch")

    # Chart
    if b.get("series") is not None and nifty and nifty.get("series") is not None:
        st.subheader("Brent vs Nifty, last 3 months (rebased to 100)")
        j = pd.concat([b["series"].rename("Brent"), nifty["series"].rename("Nifty 50")], axis=1).dropna().tail(66)
        st.line_chart(j / j.iloc[0] * 100, color=["#E0A526", "#6FB7C8"])

    # Scenario
    st.subheader("What if Brent moves to…")
    c1, c2 = st.columns([3, 1])
    br = c1.slider("Brent ($/bbl)", 40, 150, int(round(b["price"])), key="sc_brent")
    fxv = c2.number_input("USD/INR", value=round(fx["price"], 2), step=0.1, key="sc_fx")
    S = macro(br, fxv, base_brent, base_inr)
    vs_now = (br * fxv / inr_now - 1) * 100
    beta = nifty.get("beta") if nifty else None
    s_v = verdict(clamp(-vs_now * 2.2 - clamp(S["vs_base"], -20, 20), -100, 100))
    cells = [
        (f"₹{fmt(br * fxv, 0)}", f"Crude per barrel ({pct(vs_now, 1)} vs now)", tone(vs_now, True, 0.5)),
        (f"{'+' if S['extra'] >= 0 else '−'}${fmt(abs(S['extra']), 1)}bn", "Annual import bill vs average", tone(S["extra"], True, 1)),
        (f"{sgn(S['cad'])}%", "Current account deficit, % of GDP", tone(S["cad"], True, 0.02)),
        (f"{sgn(S['cpi'], 0)} bps", "CPI inflation impact", tone(S["cpi"], True, 5)),
        (S["rbi"][0], "RBI rate-cut room", S["rbi"][1]),
        (pct(beta * (br / b["price"] - 1) * 100, 1) if ok(beta) else "—", "Nifty move implied by 3M beta (low reliability)", "flat"),
        (s_v[2], "Crude pressure on indices", s_v[1]),
    ]
    cols = st.columns(4)
    for k, (v, lbl, t) in enumerate(cells):
        cols[k % 4].markdown(f'<div style="margin-bottom:12px"><div class="{t}" style="font-size:22px;font-weight:700">{v}</div>'
                             f'<div style="color:#8FAAB0;font-size:13px">{lbl}</div></div>', unsafe_allow_html=True)

    st.caption(f"Sources: Yahoo Finance via yfinance (Indian data can lag up to 15 minutes), Google Finance as fallback. "
               f"MCX crude is estimated as WTI × USD/INR. Assumptions: imports {MACRO['import_mbpd']} mb/d, GDP ${MACRO['gdp_usd_bn']}bn, "
               f"{MACRO['cpi_bps_per_10pct']} bps CPI per 10% rise in rupee crude, {MACRO['baseline_days']}-day baselines. "
               "Rules of thumb for direction, not forecasts. Not investment advice.")


dashboard()
