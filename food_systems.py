# app_streamlit_boardroom_sensors.py
# Streamlit Suite — Sensory Econometrics for Food Systems
# Boardroom-Grade UI with Sensor-Driven Econometrics
# Designed & Developed by Jit

import os, sys, json, hashlib, time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statistics import NormalDist

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, lasso_path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan

# ----------------- Config -----------------
APP_VERSION = "streamlit-v4.0-boardroom-sensors"
FEATURES = [
    "dE","brix","salt_ppm","pH","spectral_centroid","peak_rate",
    "odor_pc1","peak_force","work_chew","fractures","rms"
]

st.set_page_config(page_title="Food-System Sensory Econometrics", layout="wide")

# ----------------- Boardroom CSS -----------------
st.markdown("""<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg,#1a2a3a 0%,#0d1b2a 100%);color:#f5f7fa;}
[data-testid="stSidebar"] {background: rgba(20,30,50,0.95);backdrop-filter:blur(8px);border-right: 1px solid rgba(255,255,255,0.1);}
div[data-testid="stVerticalBlock"]>div {background:rgba(255,255,255,0.05);padding:1rem;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,0.4);margin-bottom:1rem;}
h1,h2,h3,h4 {font-family:"Segoe UI","Roboto",sans-serif;color:#e6edf3;border-bottom:2px solid #0066cc20;padding-bottom:4px;}
.stTabs [role="tablist"] {border-bottom:2px solid rgba(255,255,255,0.15);} .stTabs [role="tab"] {font-weight:600;padding:8px 20px;}
.stTabs [aria-selected="true"] {border-bottom:3px solid #00c6ff;color:#00c6ff;}
.stButton>button {background:linear-gradient(135deg,#0066cc,#00c6ff);color:white;border-radius:12px;font-weight:600;border:none;padding:0.5rem 1.2rem;box-shadow:0 3px 12px rgba(0,0,0,0.3);} .stButton>button:hover {background:linear-gradient(135deg,#005bb5,#00aaff);} 
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.3);} 
.hero{background:linear-gradient(135deg,rgba(0,198,255,0.08),rgba(0,102,204,0.08));border:1px solid rgba(255,255,255,0.08);padding:18px 22px;border-radius:20px;box-shadow:inset 0 6px 24px rgba(0,0,0,0.28),0 4px 18px rgba(0,0,0,0.35);margin-bottom:14px}
.hero .title{font-size:28px;font-weight:800;color:#e6edf3;letter-spacing:.2px}
.hero .sub{font-size:14px;color:#cbd5e1;margin-top:4px}
.hero .ver{font-size:13px;color:#99e9ff;margin-top:2px;font-weight:600}
</style>""", unsafe_allow_html=True)

# ----------------- App Header -----------------
st.markdown(f"""
<div class="hero">
  <div class="title">Sensor‑Centric Econometric Suite for Agri‑Food Systems</div>
  <div class="sub">Agri‑food sensor‑centric panel econometrics: SPC & drift, uncertainty, trade‑offs, ordered logit, regularization, and agent‑assisted predictions.</div>
  <div class="ver">Version: {APP_VERSION}</div>
</div>
""", unsafe_allow_html=True)

# ----------------- Helpers -----------------
def _crosshair(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=30, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def traffic(ok: bool) -> str:
    return "✅" if ok else "⚠️"

def psi_score(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    r, _ = np.histogram(ref, bins=bins, density=True)
    c, _ = np.histogram(cur, bins=bins, density=True)
    r = np.where(r == 0, 1e-6, r)
    c = np.where(c == 0, 1e-6, c)
    return float(np.sum((r - c) * np.log(r / c)))

# Diagnostics helpers

def calibration_by_decile(y: pd.Series, yhat: pd.Series, q: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "yhat": yhat}).sort_values("yhat")
    df["bin"] = pd.qcut(df["yhat"], q=q, duplicates="drop")
    g = df.groupby("bin").agg(y_mean=("y","mean"), yhat_mean=("yhat","mean"), n=("y","size"))
    return g.reset_index()

def residual_var_heatmap(zdf: pd.DataFrame) -> pd.DataFrame:
    return (
        zdf.groupby(["batch","session"])['resid'].var().reset_index().rename(columns={"resid":"resid_var"})
    )

# Sensitivity / contributions

def local_elasticities(model, row: pd.DataFrame, features_z):
    yhat = float(model.predict(row)[0])
    rows = []
    for t in features_z:
        up = float(model.predict(row.assign(**{t: row[t] + 1.0}))[0]) - yhat
        rows.append({"feature": t.replace("_z",""), "impact_+1SD": up, "elasticity_proxy": up / max(yhat,1e-6)})
    return pd.DataFrame(rows).sort_values("impact_+1SD", ascending=False)

def tornado_effects(model, row: pd.DataFrame, features_z):
    base = float(model.predict(row)[0])
    rows = []
    for t in features_z:
        up = float(model.predict(row.assign(**{t: row[t] + 1.0}))[0]) - base
        dn = float(model.predict(row.assign(**{t: row[t] - 1.0}))[0]) - base
        rows.append({"feature": t.replace("_z",""), "down": min(up,dn), "up": max(up,dn)})
    return pd.DataFrame(rows).sort_values("up", ascending=True)

def contribution_waterfall(model, row: pd.DataFrame, features_z):
    base0 = float(model.predict(row.assign(**{t:0.0 for t in features_z}))[0])
    parts = []
    for t in features_z:
        y1 = float(model.predict(row)[0])
        y0 = float(model.predict(row.assign(**{t:0.0}))[0])
        parts.append({"term": t.replace("_z",""), "contrib": y1 - y0})
    dfc = pd.DataFrame(parts).sort_values("contrib")
    return dfc, base0

# Trade-offs

def targeter(model, row: pd.DataFrame, features_z, target_y: float, bounds: dict, steps: int = 150, lr: float = 0.15):
    x = row[features_z].iloc[0].copy()
    for _ in range(steps):
        y = float(model.predict(pd.DataFrame([x]).assign(panelist=row["panelist"].iloc[0], batch=row["batch"].iloc[0], session=row["session"].iloc[0]))[0])
        err = target_y - y
        if abs(err) < 1e-3:
            break
        grads = {t: model.params.get(t, 0.0) for t in features_z}
        best = max(grads.items(), key=lambda kv: abs(kv[1]))[0]
        x[best] += lr * np.sign(err) * np.sign(grads[best])
        lo, hi = bounds.get(best.replace("_z",""), (-2.0, 2.0))
        x[best] = float(np.clip(x[best], lo, hi))
    final_pred = float(model.predict(pd.DataFrame([x]).assign(panelist=row["panelist"].iloc[0], batch=row["batch"].iloc[0], session=row["session"].iloc[0]))[0])
    out = pd.DataFrame({"feature_z": x.index, "z_value": x.values})
    out["feature"] = out["feature_z"].str.replace("_z","", regex=False)
    return out[["feature","z_value"]], final_pred

def cost_proxy(row: pd.Series) -> float:
    return 1.0 + 0.02*row["brix"] + 0.0003*row["salt_ppm"] + 0.0005*max(row["work_chew"],0) + 0.02*abs(row["odor_pc1"])

def pareto_front(df: pd.DataFrame, liking_col="y_hat", sodium_col="salt_ppm", cost_col="cost") -> pd.DataFrame:
    # keep points that are not dominated by any other (higher liking, lower sodium, lower cost dominate)
    pts = df[[liking_col, sodium_col, cost_col]].to_numpy()
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        li, si, ci = pts[i]
        dominated = (df[liking_col] >= li) & (df[sodium_col] <= si) & (df[cost_col] <= ci)
        dominated.iloc[i] = False
        if dominated.any():
            keep[i] = False
    return df.loc[keep]

# ----------------- Data simulation -----------------
@st.cache_data
def simulate_sensory_data(n_products=8, n_batches_per_product=3, n_panelists=24,
                          n_sessions=4, tastings_per_combo=3, random_state=42, noise_sd=0.8) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    products = [f"P{i+1}" for i in range(n_products)]
    sessions = [f"S{i+1}" for i in range(n_sessions)]
    panelists = [f"R{i+1:02d}" for i in range(n_panelists)]
    rows = []
    # Latent product traits
    prod_latent = {
        p: {
            "color_L": rng.normal(65,5), "color_a": rng.normal(5,3), "color_b": rng.normal(20,4),
            "crunchiness": rng.normal(0.0,1.0), "sweetness_brix": rng.normal(9,2),
            "salt_ppm": rng.normal(650,120), "pH": rng.normal(6.6,0.6), "aroma_intensity": rng.normal(0.0,1.0),
            "texture_firm": rng.normal(0.0,1.0)
        }
        for p in products
    }
    panel_len = {r: rng.normal(0.0, 0.3) for r in panelists}
    sess_eff = {s: rng.normal(0.0, 0.15) for s in sessions}

    for p in products:
        for b in range(1, n_batches_per_product+1):
            batch_id = f"{p}_B{b}"
            drift = {"dE": abs(rng.normal(1.5,0.6)), "cr": rng.normal(0.0,0.25),
                     "ar": rng.normal(0.0,0.35), "fm": rng.normal(0.0,0.25),
                     "bx": rng.normal(0.0,0.5), "sa": rng.normal(0.0,40), "ph": rng.normal(0.0,0.15)}
            for s in sessions:
                for r in panelists:
                    for _ in range(tastings_per_combo):
                        storage_days = max(0, rng.normal(7,3))
                        dE = drift["dE"] + 0.05*storage_days + abs(rng.normal(0,0.4))
                        brix = prod_latent[p]["sweetness_brix"] + drift["bx"] + rng.normal(0,0.6)
                        salt_ppm = prod_latent[p]["salt_ppm"] + drift["sa"] + rng.normal(0,35)
                        pH = prod_latent[p]["pH"] + drift["ph"] + rng.normal(0,0.1)
                        odor_pc1 = prod_latent[p]["aroma_intensity"] + drift["ar"] - 0.03*storage_days + rng.normal(0,0.4)
                        spectral_centroid = 3500 + 600*(prod_latent[p]["crunchiness"] + drift["cr"]) + rng.normal(0,200)
                        peak_rate = max(0, rng.normal(12 + 4*(prod_latent[p]["crunchiness"] + drift["cr"]), 3))
                        peak_force = 35 + 8*prod_latent[p]["texture_firm"] + rng.normal(0,3)
                        work_chew = 120 + 30*prod_latent[p]["texture_firm"] + rng.normal(0,10)
                        fractures = max(0, int(rng.normal(6 + 2*prod_latent[p]["texture_firm"], 2)))
                        rms = max(0, rng.normal(0.08 + 0.03*prod_latent[p]["crunchiness"], 0.02))
                        liking = (
                            6.0 - 0.18*dE + 0.10*(brix-10) + 0.12*odor_pc1 + 0.3*((spectral_centroid-3500)/500.0)
                            + panel_len[r] + sess_eff[s] + rng.normal(0, noise_sd)
                        )
                        liking = float(np.clip(liking, 1.0, 9.0))
                        rows.append(dict(
                            product=p, batch=batch_id, session=s, panelist=r, storage_days=storage_days,
                            dE=dE, brix=brix, salt_ppm=salt_ppm, pH=pH, spectral_centroid=spectral_centroid,
                            peak_rate=peak_rate, peak_force=peak_force, work_chew=work_chew, fractures=fractures,
                            odor_pc1=odor_pc1, rms=rms, overall_liking=liking
                        ))
    return pd.DataFrame(rows)

@st.cache_data
def fit_ols_fe(df: pd.DataFrame):
    z = df.copy()
    for c in FEATURES:
        mu, sd = z[c].mean(), z[c].std(ddof=0)
        z[c+"_z"] = 0.0 if sd == 0 else (z[c]-mu)/sd
    z_features = [f+"_z" for f in FEATURES]
    formula = "overall_liking ~ " + " + ".join(z_features) + " + C(panelist) + C(batch) + C(session)"
    mdl = smf.ols(formula=formula, data=z).fit(cov_type="HC3")
    z["y_hat"] = mdl.predict(z)
    z["resid"] = z["overall_liking"] - z["y_hat"]
    return mdl, z, z_features, formula

# ----------------- Agent V -----------------
def agent_v_predict(row: pd.Series) -> dict:
    y0 = (
        6.2 - 0.12*row["dE"] + 0.05*(row["brix"]-10)
        + 0.22*((row["spectral_centroid"]-3500)/500.0)
        + 0.10*((row["peak_rate"]-12)/4.0) + 0.12*row["odor_pc1"]
    )
    y = float(np.clip(y0 + np.random.normal(0,0.45), 1, 9))
    conf = float(np.clip(0.55 + 0.2*np.random.rand(), 0, 1))
    return {"y_v": y, "confidence": conf, "qc": "lighting_ok"}

# ----------------- Sidebar -----------------
st.sidebar.header("Controls")
n_products = st.sidebar.slider("Products", 4, 20, 8)
n_batches = st.sidebar.slider("Batches/product", 2, 6, 3)
n_panel = st.sidebar.slider("Panelists", 8, 40, 24)
n_sessions = st.sidebar.slider("Sessions", 2, 8, 4)
tpc = st.sidebar.slider("Tastings/combo", 1, 5, 3)
noise = st.sidebar.slider("Noise SD", 0.2, 1.5, 0.8)
seed = st.sidebar.number_input("Seed", value=42)

df = simulate_sensory_data(n_products, n_batches, n_panel, n_sessions, tpc, seed, noise)
mdl, zdf, z_features, formula = fit_ols_fe(df)

# ----------------- KPI BAR -----------------
st.markdown("## Executive KPIs")
psi_val = psi_score(zdf["dE"].iloc[: max(1, len(zdf)//5) ], zdf["dE"].iloc[ max(1, len(zdf)//5): ])
bp = het_breuschpagan(mdl.resid, mdl.model.exog)
vif_df = pd.DataFrame({
    "feature": z_features,
    "VIF": [variance_inflation_factor(sm.add_constant(zdf[z_features]).to_numpy(), i)
             for i in range(1, len(z_features)+1)]
})
cal = sm.OLS(zdf["overall_liking"], sm.add_constant(zdf["y_hat"])) .fit()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("R²", f"{mdl.rsquared:.3f}")
kpi2.metric("Adj R²", f"{mdl.rsquared_adj:.3f}")
kpi3.metric("Observations", f"{len(df):,}")
kpi4.metric("PSI ΔE", f"{psi_val:.3f}")

kpi5, kpi6, kpi7 = st.columns(3)
kpi5.metric("VIF<5", traffic((vif_df["VIF"] < 5).all()))
kpi6.metric("Heteroskedasticity", traffic(bp[1] > 0.05))
kpi7.metric("Calibration≈1", traffic(0.9 <= cal.params.get("y_hat", 0) <= 1.1))

st.markdown("---")

# ----------------- Tabs -----------------
tabs = st.tabs([
    "Overview","Uncertainty","Diagnostics","Sensitivity",
    "SPC & Drift","Trade-offs","Ordered Logit","Regularization",
    "Agents","Data","Governance"
])

# ---- Overview ----
with tabs[0]:
    st.subheader("Model Overview")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(zdf, x="y_hat", y="overall_liking", hover_data=["product","batch","panelist","session"]) 
        fig.add_trace(go.Scatter(x=[1,9], y=[1,9], mode="lines", name="Ideal"))
        st.plotly_chart(_crosshair(fig), use_container_width=True)
    with col2:
        params = mdl.params.drop([p for p in mdl.params.index if p.startswith("C(") or p=="Intercept"]) 
        conf = mdl.conf_int().loc[params.index]
        coef_df = pd.DataFrame({"term": params.index, "coef": params.values,
                                "ci_low": conf[0].values, "ci_high": conf[1].values}).sort_values("coef")
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=coef_df["coef"], y=coef_df["term"],
                                  error_x=dict(type="data", array=coef_df["ci_high"]-coef_df["coef"],
                                               arrayminus=coef_df["coef"]-coef_df["ci_low"]),
                                  mode="markers"))
        st.plotly_chart(_crosshair(figc), use_container_width=True)
    st.caption("Designed & Developed by Jit")

# ---- Uncertainty ----
with tabs[1]:
    st.subheader("Uncertainty (Bootstrap)")
    feat = st.selectbox("Slice feature", FEATURES, index=0)
    q05, q95 = df[feat].quantile(0.05), df[feat].quantile(0.95)
    grid = np.linspace(q05, q95, 60)
    med = df[FEATURES].median()
    med_df = pd.DataFrame([med]*len(grid)); med_df[feat] = grid
    for c in FEATURES:
        mu, sd = df[c].mean(), df[c].std(ddof=0)
        med_df[c+"_z"] = 0.0 if sd==0 else (med_df[c]-mu)/sd
    med_df["panelist"] = df["panelist"].iloc[0]; med_df["batch"] = df["batch"].iloc[0]; med_df["session"] = df["session"].iloc[0]

    y0 = mdl.predict(med_df)
    boot_preds = []
    rng = np.random.default_rng(123)
    for _ in range(200):
        idx = rng.integers(0, len(df), len(df))
        try:
            boot_model = smf.ols(formula=formula, data=df.iloc[idx]).fit()
            boot_preds.append(boot_model.predict(med_df))
        except Exception:
            pass
    if len(boot_preds) >= 5:
        bands = np.percentile(np.vstack(boot_preds), [5,20,50,80,95], axis=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=y0, name="ŷ", mode="lines"))
        fig.add_trace(go.Scatter(x=grid, y=bands[4], line=dict(width=0)))
        fig.add_trace(go.Scatter(x=grid, y=bands[0], fill="tonexty", opacity=0.18, name="95%"))
        fig.add_trace(go.Scatter(x=grid, y=bands[3], line=dict(width=0)))
        fig.add_trace(go.Scatter(x=grid, y=bands[1], fill="tonexty", opacity=0.28, name="80%"))
        st.plotly_chart(_crosshair(fig), use_container_width=True)
    else:
        st.warning("Bootstrap failed to converge; showing point estimate only.")
        st.plotly_chart(_crosshair(px.line(x=grid, y=y0, labels={"x":feat, "y":"ŷ"})), use_container_width=True)

# ---- Diagnostics ----
with tabs[2]:
    st.subheader("Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(zdf, x="y_hat", y="resid", labels={"y_hat":"Fitted (ŷ)", "resid":"Residual"})
        fig.add_hline(y=0)
        st.plotly_chart(_crosshair(fig), use_container_width=True)
    with col2:
        # Calibration curve by deciles
        cal_df = calibration_by_decile(zdf["overall_liking"], zdf["y_hat"], q=10)
        fig_cal = px.line(cal_df, x="yhat_mean", y="y_mean", markers=True, labels={"yhat_mean":"Mean ŷ (bin)","y_mean":"Mean y"})
        fig_cal.add_trace(go.Scatter(x=[1,9], y=[1,9], mode="lines", name="Ideal"))
        st.plotly_chart(_crosshair(fig_cal), use_container_width=True)

    # QQ plot
    r = np.sort(zdf["resid"].values)
    n = len(r)
    probs = (np.arange(1, n+1)-0.5)/n
    q_theor = np.array([NormalDist().inv_cdf(p) for p in probs])
    figqq = go.Figure()
    figqq.add_trace(go.Scatter(x=q_theor, y=r, mode="markers", name="Residuals"))
    lim = max(abs(q_theor).max(), abs(r).max())
    figqq.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="Ideal"))
    st.plotly_chart(_crosshair(figqq), use_container_width=True)

    # Residual variance heatmap (batch x session)
    hv = residual_var_heatmap(zdf)
    pivot = hv.pivot(index="session", columns="batch", values="resid_var")
    fig_hm = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", origin="lower", labels=dict(color="Var(resid)"))
    fig_hm.update_layout(title="Residual Variance Heatmap — batch × session")
    st.plotly_chart(_crosshair(fig_hm), use_container_width=True)

    cooks = pd.DataFrame({"row": np.arange(len(df)), "cooks_d": OLSInfluence(mdl).cooks_distance[0]})
    st.dataframe(cooks.sort_values("cooks_d", ascending=False).head(15))

# ---- Sensitivity ----
with tabs[3]:
    st.subheader("Sensitivity & Contributions")
    idx = st.number_input("Row index", min_value=0, max_value=len(zdf)-1, value=0)
    row = zdf.iloc[[int(idx)]][["panelist","batch","session"] + z_features]

    tor = tornado_effects(mdl, row, z_features)
    figt = go.Figure()
    figt.add_trace(go.Bar(x=tor["up"], y=tor["feature"], orientation="h", name="+1 SD"))
    figt.add_trace(go.Bar(x=tor["down"], y=tor["feature"], orientation="h", name="-1 SD"))
    figt.update_layout(barmode="overlay", title="Tornado Sensitivity (±1 SD)")
    st.plotly_chart(_crosshair(figt), use_container_width=True)

    wdf, base0 = contribution_waterfall(mdl, row, z_features)
    figw = go.Figure()
    figw.add_trace(go.Bar(x=wdf["term"], y=wdf["contrib"], name="Contribution"))
    figw.add_hline(y=0)
    figw.update_layout(title=f"Contributions (β·z) — Base (FE+Intercept): {base0:.2f}")
    st.plotly_chart(_crosshair(figw), use_container_width=True)

    elas = local_elasticities(mdl, row, z_features)
    st.dataframe(elas)

# ---- SPC & Drift ----
with tabs[4]:
    st.subheader("SPC & Drift")
    feat = st.selectbox("SPC feature (batch means)", ["dE","odor_pc1","spectral_centroid","brix","salt_ppm"], index=0)
    s = df.groupby("batch")[feat].mean()
    mu = s.mean(); sd = s.std(ddof=0)
    ucl = mu + 3*sd; lcl = mu - 3*sd
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index.astype(str), y=s.values, mode="lines+markers", name=feat))
    fig.add_hline(y=mu, line_dash="dash", annotation_text="Center")
    fig.add_hline(y=ucl, line_dash="dot", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dot", annotation_text="LCL")
    fig.update_layout(title=f"Control Chart — {feat} (batch means)")
    st.plotly_chart(_crosshair(fig), use_container_width=True)

    # PSI drift vs baseline (first 20%)
    cut = max(1, len(df)//5)
    psi_val_feat = psi_score(df[feat].iloc[:cut], df[feat].iloc[cut:])
    st.info(f"PSI ({feat}) using 20% split: {psi_val_feat:.3f}  (heuristic: <0.1 low • 0.1–0.25 moderate • >0.25 high)")

# ---- Trade-offs ----
with tabs[5]:
    st.subheader("Trade-offs: Targeting & Pareto")
    tcol1, tcol2 = st.columns([1,2])
    with tcol1:
        idx_t = st.number_input("Row index (targeter)", min_value=0, max_value=len(zdf)-1, value=0)
        target_y = st.slider("Target liking", 1.0, 9.0, 7.5, 0.1)
        bound_z = st.slider("Feature bound |z| ≤", 0.5, 3.0, 2.0, 0.1)
        row_t = zdf.iloc[[int(idx_t)]][["panelist","batch","session"] + z_features]
        bounds = {f: (-float(bound_z), float(bound_z)) for f in FEATURES}
        sol, yfinal = targeter(mdl, row_t, z_features, float(target_y), bounds)
        st.write(f"Predicted liking after targeting: **{yfinal:.2f}**")
        st.dataframe(sol)
    with tcol2:
        mdl2, zdf2, _, _ = fit_ols_fe(df)
        tmp = df.copy()
        tmp["y_hat"] = zdf2["y_hat"].values
        tmp["cost"] = tmp.apply(cost_proxy, axis=1)
        pf = pareto_front(tmp, "y_hat", "salt_ppm", "cost")
        figp = px.scatter(tmp, x="salt_ppm", y="y_hat", size="cost", color="cost", hover_data=["product","batch"])
        figp.add_trace(go.Scatter(x=pf["salt_ppm"], y=pf["y_hat"], mode="lines+markers", name="Pareto"))
        figp.update_layout(title="Liking vs Sodium vs Cost (Pareto front)")
        st.plotly_chart(_crosshair(figp), use_container_width=True)

# ---- Ordered Logit ----
with tabs[6]:
    st.subheader("Ordered Logit (1–9)")
    y_ord = zdf["overall_liking"].round().astype(int).clip(1,9)
    X_ord = zdf[[f+"_z" for f in FEATURES]]
    mod = OrderedModel(y_ord, X_ord, distr='logit')
    try:
        res = mod.fit(method='bfgs', disp=False)
        llf = res.llf
        llnull = OrderedModel(y_ord, np.ones((len(y_ord),1)), distr='logit').fit(method='bfgs', disp=False).llf
        pseudo_r2 = 1 - llf/llnull
        feat = st.selectbox("Slice feature", FEATURES, index=0)
        q05,q95 = df[feat].quantile(0.05), df[feat].quantile(0.95)
        grid = np.linspace(q05, q95, 60)
        med = df[FEATURES].median()
        Xslice = pd.DataFrame([med]*len(grid)); Xslice[feat] = grid
        for c in FEATURES:
            mu, sd = df[c].mean(), df[c].std(ddof=0)
            Xslice[c+"_z"] = 0.0 if sd==0 else (Xslice[c]-mu)/sd
        probs = res.model.predict(res.params, exog=Xslice[[f+"_z" for f in FEATURES]])
        prob_df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(1, probs.shape[1]+1)])
        prob_df["x"] = grid
        figprob = go.Figure()
        for i in range(1,10):
            col = f"class_{i}"
            if col in prob_df:
                figprob.add_trace(go.Scatter(x=prob_df["x"], y=prob_df[col], mode="lines", name=f"Pr(y={i})"))
        figprob.update_layout(title=f"Class Probabilities vs {feat}")
        st.plotly_chart(_crosshair(figprob), use_container_width=True)
        with st.expander("Model summary"):
            st.text(str(res.summary()))
        st.info(f"McFadden pseudo-R²: {pseudo_r2:.3f}")
    except Exception as e:
        st.warning(f"Ordered logit failed: {e}")

# ---- Regularization ----
with tabs[7]:
    st.subheader("Regularization: Ridge & LASSO vs OLS")
    X = df[FEATURES].to_numpy(); y = df["overall_liking"].to_numpy()
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    alphas = np.logspace(-3, 2, 50)
    ridge = RidgeCV(alphas=alphas).fit(Xs, y)
    lasso = LassoCV(alphas=alphas, cv=5, random_state=1).fit(Xs, y)
    l_alphas, coefs, _ = lasso_path(Xs, y, alphas=np.logspace(-3,1,30))

    ols = fit_ols_fe(df)[0]
    ols_coefs = pd.Series({f: ols.params.get(f+"_z", 0.0) for f in FEATURES}, name="OLS (β on z)")
    ridge_coefs = pd.Series(ridge.coef_, index=FEATURES, name="Ridge")
    lasso_coefs = pd.Series(np.nan_to_num(lasso.coef_), index=FEATURES, name="LASSO")
    coef_cmp = pd.concat([ols_coefs, ridge_coefs, lasso_coefs], axis=1).reset_index().melt(id_vars="index", var_name="Model", value_name="Coef")
    figcmp = px.bar(coef_cmp, x="Coef", y="index", color="Model", barmode="group", labels={"index":"Feature"})
    figcmp.update_layout(title="Coefficient Comparison (standardized features)")
    st.plotly_chart(_crosshair(figcmp), use_container_width=True)

    figpath = go.Figure()
    for j, f in enumerate(FEATURES):
        figpath.add_trace(go.Scatter(x=np.log10(l_alphas), y=coefs[j], mode="lines", name=f))
    figpath.update_layout(title="LASSO Path (log10 α vs coef)")
    st.plotly_chart(_crosshair(figpath), use_container_width=True)

    st.info(f"Ridge α*: {ridge.alpha_:.4f} | LASSO α*: {lasso.alpha_:.4f}")

# ---- Agents ----
with tabs[8]:
    st.subheader("Agents (QC Sensor)")
    idx = st.number_input("Row index (Agent)", min_value=0, max_value=len(zdf)-1, value=0)
    row = zdf.iloc[int(idx)]

    # OLS prediction (extract first element, then cast)
    pred_series = mdl.predict(zdf.iloc[[int(idx)]])
    ols_pred = float(pred_series.iloc[0])

    # Agent V prediction
    av = agent_v_predict(row)

    # Arbitration: weight Agent V if closer to observed truth
    alpha = 0.25 if abs(av["y_v"] - row["overall_liking"]) < abs(ols_pred - row["overall_liking"]) else 0.0
    y_ens = (1 - alpha) * ols_pred + alpha * av["y_v"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("OLS Prediction", f"{ols_pred:.2f}")
        st.metric("Agent V Prediction", f"{av['y_v']:.2f}")
        st.metric("Confidence", f"{av['confidence']:.2f}")
    with col2:
        st.metric("Ensemble ŷ*", f"{y_ens:.2f}")
        st.write(f"QC status: {av['qc']}")



# ---- Data ----
with tabs[9]:
    st.subheader("Data Preview")
    st.dataframe(df.head(200))
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "synthetic_sensory_dataset.csv", "text/csv")

# ---- Governance ----
with tabs[10]:
    st.subheader("Governance & Audit")
    env_hash = hashlib.sha256(json.dumps({
        "python": sys.version.split()[0],
        "statsmodels": sm.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "plotly": px.__version__
    }, sort_keys=True).encode()).hexdigest()[:12]
    audit_rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": APP_VERSION,
        "seed": int(seed),
        "n_products": int(n_products), "n_batches": int(n_batches), "n_panel": int(n_panel), "n_sessions": int(n_sessions), "tpc": int(tpc),
        "noise": float(noise), "env_hash": env_hash
    }
    st.json(audit_rec)
    st.caption("Designed & Developed by Jit")
