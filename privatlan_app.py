import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import json

# ── Sidkonfiguration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kreditrisk — Privatlån",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f7f5f2; }

section[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #a8b2d8 !important; }
section[data-testid="stSidebar"] h2 {
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #64748b !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
    font-family: 'DM Mono', monospace !important;
}

h1 { color: #0f172a !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h2 { color: #334155 !important; font-weight: 500 !important; font-size: 0.85rem !important;
     letter-spacing: 0.08em; text-transform: uppercase; }
h3 { color: #475569 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-radius: 7px !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #0f172a !important;
    color: white !important;
}
hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── MPL-tema ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#e2e8f0",
    "axes.labelcolor": "#475569",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#f1f5f9",
    "grid.linewidth": 0.8,
    "text.color": "#334155",
    "font.family": "sans-serif",
})

# ── Ladda modell ──────────────────────────────────────────────────────────────
@st.cache_resource
def ladda_modell():
    with open('privatlan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('privatlan_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('privatlan_features.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

@st.cache_data
def ladda_data():
    return pd.read_csv('privatlan_data.csv')

try:
    model, scaler, features = ladda_modell()
    df = ladda_data()
    modell_laddad = True
except FileNotFoundError:
    modell_laddad = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Kreditrisk\nPrivatlån")
    st.markdown("---")
    st.markdown("## Låntagaruppgifter")

    anstallning = st.selectbox(
        "Anställningsform",
        ["Fast anställd", "Vikariat", "Egenföretagare", "Pensionär", "Arbetslös"]
    )

    syfte = st.selectbox(
        "Syfte med lånet",
        ["Renovering", "Bil", "Konsumtion", "Övrigt"]
    )

    inkomst = st.slider("Årsinkomst (kr)", 120000, 1200000, 420000, step=10000,
                        format="%d")
    lanebelopp = st.slider("Lånebelopp (kr)", 10000, 500000, 150000, step=5000,
                           format="%d")
    alder = st.slider("Ålder", 18, 85, 35)
    ranta = st.slider("Ränta (%)", 4.0, 25.0, 9.5, step=0.5)
    amorteringstid = st.select_slider(
        "Amorteringstid (månader)",
        options=[12, 24, 36, 48, 60, 72, 84],
        value=36
    )
    tidigare_anmarkning = st.selectbox("Tidigare betalningsanmärkning", ["Nej", "Ja"])

    st.markdown("---")
    st.markdown(
        "<span style='color:#2d3748;font-size:0.68rem;'>Modell: Logistisk regression<br>"
        "Data: Syntetisk · 10 000 lån<br>Gini: ~0.55</span>",
        unsafe_allow_html=True
    )

# ── Beräkningar ───────────────────────────────────────────────────────────────
skuldsattningsgrad = lanebelopp / inkomst
manlig_ranta = ranta / 100 / 12
if manlig_ranta > 0:
    manadskostnad = int(lanebelopp * (manlig_ranta * (1 + manlig_ranta)**amorteringstid) /
                        ((1 + manlig_ranta)**amorteringstid - 1))
else:
    manadskostnad = int(lanebelopp / amorteringstid)
betalningsborda = manadskostnad / (inkomst / 12)

# ── Förbered features för modellen ───────────────────────────────────────────
def berakna_risk(model, scaler, features, anstallning, syfte, inkomst,
                 lanebelopp, alder, ranta, amorteringstid, tidigare_anmarkning):
    row = {f: 0 for f in features}
    row['lanebelopp']             = lanebelopp
    row['inkomst']                = inkomst
    row['alder']                  = alder
    row['ranta']                  = ranta
    row['amorteringstid_manader'] = amorteringstid
    row['skuldsattningsgrad']     = lanebelopp / inkomst
    manlig_r = ranta / 100 / 12
    mk = int(lanebelopp * (manlig_r * (1 + manlig_r)**amorteringstid) /
             ((1 + manlig_r)**amorteringstid - 1)) if manlig_r > 0 else lanebelopp // amorteringstid
    row['manadskostnad']   = mk
    row['betalningsborda'] = mk / (inkomst / 12)
    row['tidigare_anmarkning'] = 1 if tidigare_anmarkning == "Ja" else 0

    # One-hot encoding
    for a in ['Vikariat', 'Egenföretagare', 'Pensionär', 'Arbetslös']:
        key = f'anstallningsform_{a}'
        if key in row:
            row[key] = 1 if anstallning == a else 0
    for s in ['Bil', 'Konsumtion', 'Övrigt']:
        key = f'syfte_{s}'
        if key in row:
            row[key] = 1 if syfte == s else 0

    X = pd.DataFrame([row])[features]
    X_scaled = scaler.transform(X)
    risk = model.predict_proba(X_scaled)[0][1] * 100
    return risk, row

if modell_laddad:
    risk, row = berakna_risk(model, scaler, features, anstallning, syfte,
                              inkomst, lanebelopp, alder, ranta,
                              amorteringstid, tidigare_anmarkning)
else:
    risk = 25.0

# ── Rubrik ────────────────────────────────────────────────────────────────────
st.markdown("# Kreditriskbedömning — Privatlån")
st.markdown(
    "<span style='color:#64748b;font-size:0.85rem;'>"
    "Logistisk regressionsmodell tränad på 10 000 syntetiska privatlån · "
    "Gini-koefficient ~0.55 · IFRS 9-dokumenterad</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# ── Riskindikator ─────────────────────────────────────────────────────────────
if risk < 5:
    farg = "#22c55e"
    status = "🟢 Låg risk — Rekommenderas för godkännande"
    bg = "#f0fdf4"
elif risk < 20:
    farg = "#f59e0b"
    status = "🟡 Medel risk — Kräver manuell granskning"
    bg = "#fffbeb"
else:
    farg = "#ef4444"
    status = "🔴 Hög risk — Rekommenderas för avslag"
    bg = "#fef2f2"

st.markdown(
    f"<div style='background:{bg};border-left:4px solid {farg};"
    f"padding:16px 20px;border-radius:8px;margin-bottom:20px;'>"
    f"<span style='font-size:1.1rem;font-weight:600;color:{farg};'>{status}</span><br>"
    f"<span style='font-size:0.8rem;color:#64748b;margin-top:4px;display:block;'>"
    f"Betalningsrisk: <strong style='font-family:DM Mono,monospace;color:#0f172a;'>{risk:.1f}%</strong> — "
    f"{'Av 100 liknande låntagare beräknas ' + str(int(risk)) + ' sluta betala'}</span>"
    f"</div>",
    unsafe_allow_html=True
)

# ── KPI-mätvärden ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Betalningsrisk",      f"{risk:.1f}%")
c2.metric("Månadskostnad",       f"{manadskostnad:,} kr")
c3.metric("Skuldsättningsgrad",  f"{skuldsattningsgrad:.2f}×")
c4.metric("Betalningsbörda",     f"{betalningsborda:.0%}")
c5.metric("Lånebelopp / Inkomst",f"{lanebelopp/inkomst:.2f}×")

st.markdown("---")

# ── Flikar ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Riskanalys", "🔍 Varför denna risk?", "📈 Jämför med portfölj"])

# ════════════════════════════════════════════════════
# FLIK 1: Riskanalys
# ════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    # Riskgauge
    with col1:
        st.markdown("## Riskmätare")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="white")
        ax.set_facecolor("white")

        # Bakgrundssegment
        for start, end, color in [(0, 5, "#bbf7d0"), (5, 20, "#fef08a"), (20, 100, "#fecaca")]:
            theta1 = 180 - (start / 100 * 180)
            theta2 = 180 - (end / 100 * 180)
            wedge = mpatches.Wedge((0.5, 0), 0.4, theta2, theta1,
                                    width=0.12, facecolor=color, edgecolor="white", linewidth=2)
            ax.add_patch(wedge)

        # Nål
        angle = np.radians(180 - (min(risk, 100) / 100 * 180))
        ax.annotate("", xy=(0.5 + 0.35 * np.cos(angle), 0 + 0.35 * np.sin(angle)),
                    xytext=(0.5, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.5))

        ax.text(0.5, -0.08, f"{risk:.1f}%", ha='center', va='center',
                fontsize=22, fontweight='bold', color='#0f172a',
                fontfamily='monospace')
        ax.text(0.1, -0.12, "Låg", fontsize=9, color="#22c55e", fontweight='500')
        ax.text(0.45, -0.12, "Medel", fontsize=9, color="#f59e0b", fontweight='500')
        ax.text(0.78, -0.12, "Hög", fontsize=9, color="#ef4444", fontweight='500')

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 0.5)
        ax.axis('off')
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Riskfaktorer
    with col2:
        st.markdown("## Riskfaktorer")

        faktorer = {
            "Betalningsbörda":    (betalningsborda, 0.30, "hög om >30% av inkomsten"),
            "Skuldsättningsgrad": (skuldsattningsgrad, 1.0, "hög om >1× årsinkomst"),
            "Ränta":              (ranta / 25, 0.5, f"{ranta:.1f}% av max 25%"),
        }

        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="white")
        ax.set_facecolor("white")

        y_pos = range(len(faktorer))
        for i, (namn, (varde, grans, beskr)) in enumerate(faktorer.items()):
            pct = min(varde, 1.0)
            color = "#ef4444" if varde > grans else "#22c55e"
            ax.barh(i, pct, color=color, alpha=0.85, height=0.5)
            ax.axvline(grans, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(1.02, i, beskr, va='center', fontsize=8, color="#64748b")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(list(faktorer.keys()), fontsize=10)
        ax.set_xlim(0, 1.5)
        ax.set_xlabel("Nivå (streckad linje = gränsvärde)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Risktabell
    st.markdown("## Risknivåer")
    st.markdown("""
| Risknivå | Betalningsrisk | Rekommendation |
|---|---|---|
| 🟢 Låg | < 5% | Godkänn |
| 🟡 Medel | 5–20% | Manuell granskning |
| 🔴 Hög | > 20% | Avslå |
""")

# ════════════════════════════════════════════════════
# FLIK 2: Varför denna risk?
# ════════════════════════════════════════════════════
with tab2:
    st.markdown("## Vad driver risken för denna låntagare?")

    if modell_laddad:
        # Feature importance för denna låntagare
        koeff = model.coef_[0]
        X_row = pd.DataFrame([row])[features]
        X_scaled = scaler.transform(X_row)
        bidrag = koeff * X_scaled[0]

        importance_df = pd.DataFrame({
            'feature': features,
            'bidrag': bidrag
        }).sort_values('bidrag', ascending=True)

        # Visa bara de 8 mest påverkande
        top = pd.concat([
            importance_df.head(4),
            importance_df.tail(4)
        ]).sort_values('bidrag')

        col1, col2 = st.columns([1.5, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
            ax.set_facecolor("white")
            colors = ["#ef4444" if v > 0 else "#22c55e" for v in top['bidrag']]
            bars = ax.barh(top['feature'], top['bidrag'], color=colors,
                           alpha=0.85, edgecolor='none', height=0.6)
            ax.axvline(0, color="#334155", linewidth=1.2)
            ax.set_xlabel("Påverkan på betalningsrisken", fontsize=10)
            ax.set_title("Röd = ökar risken   Grön = minskar risken",
                         fontsize=9, color="#64748b", pad=10)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("### Sammanfattning")

            risk_faktorer = []
            skydd_faktorer = []

            if betalningsborda > 0.30:
                risk_faktorer.append(f"⚠️ Betalningsbörda {betalningsborda:.0%} — över 30%-gränsen")
            if skuldsattningsgrad > 1.0:
                risk_faktorer.append(f"⚠️ Skuldsättning {skuldsattningsgrad:.2f}× — över 1× inkomsten")
            if tidigare_anmarkning == "Ja":
                risk_faktorer.append("⚠️ Tidigare betalningsanmärkning")
            if anstallning in ["Arbetslös", "Vikariat"]:
                risk_faktorer.append(f"⚠️ Anställningsform: {anstallning}")
            if ranta > 15:
                risk_faktorer.append(f"⚠️ Hög ränta: {ranta:.1f}%")

            if betalningsborda <= 0.20:
                skydd_faktorer.append(f"✅ Låg betalningsbörda {betalningsborda:.0%}")
            if skuldsattningsgrad <= 0.5:
                skydd_faktorer.append(f"✅ Låg skuldsättning {skuldsattningsgrad:.2f}×")
            if tidigare_anmarkning == "Nej":
                skydd_faktorer.append("✅ Ingen betalningsanmärkning")
            if anstallning == "Fast anställd":
                skydd_faktorer.append("✅ Fast anställning")
            if syfte == "Renovering":
                skydd_faktorer.append("✅ Lågriskändamål: Renovering")

            if risk_faktorer:
                st.markdown("**Riskfaktorer:**")
                for f in risk_faktorer:
                    st.markdown(f)
            if skydd_faktorer:
                st.markdown("**Skyddsfaktorer:**")
                for f in skydd_faktorer:
                    st.markdown(f)

            if not risk_faktorer and not skydd_faktorer:
                st.markdown("Låntagaren har en genomsnittlig riskprofil.")

# ════════════════════════════════════════════════════
# FLIK 3: Jämför med portfölj
# ════════════════════════════════════════════════════
with tab3:
    st.markdown("## Hur ser denna låntagare ut jämfört med portföljen?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Skuldsättningsgrad")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        ax.set_facecolor("#fafafa")
        sns.kdeplot(df['skuldsattningsgrad'], ax=ax, color="#3b82f6",
                    fill=True, alpha=0.2, linewidth=2, label="Portföljen")
        ax.axvline(skuldsattningsgrad, color="#ef4444", linewidth=2,
                   linestyle="--", label=f"Denna låntagare ({skuldsattningsgrad:.2f}×)")
        ax.set_xlabel("Skuldsättningsgrad")
        ax.set_ylabel("Densitet")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("## Betalningsbörda")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        ax.set_facecolor("#fafafa")
        sns.kdeplot(df['betalningsborda'], ax=ax, color="#3b82f6",
                    fill=True, alpha=0.2, linewidth=2, label="Portföljen")
        ax.axvline(betalningsborda, color="#ef4444", linewidth=2,
                   linestyle="--", label=f"Denna låntagare ({betalningsborda:.0%})")
        ax.axvline(0.30, color="#f59e0b", linewidth=1.2,
                   linestyle=":", label="Gränsvärde 30%")
        ax.set_xlabel("Betalningsbörda (andel av månadsinkomst)")
        ax.set_ylabel("Densitet")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Percentil
    st.markdown("## Var befinner sig låntagaren i portföljen?")
    pct_skuld = (df['skuldsattningsgrad'] < skuldsattningsgrad).mean() * 100
    pct_borda = (df['betalningsborda'] < betalningsborda).mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Skuldsättning — percentil",
              f"{pct_skuld:.0f}%",
              f"Högre än {pct_skuld:.0f}% av portföljen")
    c2.metric("Betalningsbörda — percentil",
              f"{pct_borda:.0f}%",
              f"Högre än {pct_borda:.0f}% av portföljen")
    c3.metric("Portföljens snitt-betalningsrisk",
              f"{df['default'].mean()*100:.1f}%")
