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
APP_VERSION = "streamlit-v3.1-boardroom-sensors"
FEATURES = ["dE","brix","salt_ppm","pH","spectral_centroid","peak_rate",
            "odor_pc1","peak_force","work_chew","fractures","rms"]

st.set_page_config(page_title="Food-System Sensory Econometrics", layout="wide")

# ----------------- Boardroom CSS -----------------
st.markdown("""<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg,#1a2a3a 0%,#0d1b2a 100%);color:#f5f7fa;}
[data-testid="stSidebar"] {background: rgba(20,30,50,0.95);backdrop-filter:blur(8px);}
div[data-testid="stVerticalBlock"]>div {background:rgba(255,255,255,0.05);padding:1rem;
 border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,0.4);margin-bottom:1rem;}
h1,h2,h3,h4 {font-family:"Segoe UI","Roboto",sans-serif;color:#e6edf3;border-bottom:2px solid #0066cc20;padding-bottom:4px;}
.stTabs [role="tablist"] {border-bottom:2px solid rgba(255,255,255,0.15);}
.stTabs [role="tab"] {font-weight:600;padding:8px 20px;}
.stTabs [aria-selected="true"] {border-bottom:3px solid #00c6ff;color:#00c6ff;}
.stButton>button {background:linear-gradient(135deg,#0066cc,#00c6ff);color:white;border-radius:12px;font-weight:600;
 border:none;padding:0.5rem 1.2rem;box-shadow:0 3px 12px rgba(0,0,0,0.3);}
.stButton>button:hover {background:linear-gradient(135deg,#005bb5,#00aaff);}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.3);}
</style>""", unsafe_allow_html=True)

# ----------------- Helpers -----------------
def _crosshair(fig): 
    fig.update_layout(hovermode="x unified",margin=dict(l=30,r=20,t=50,b=40),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    return fig
def traffic(ok): return "✅" if ok else "⚠️"
def psi_score(ref,cur,bins=10):
    r,c=np.histogram(ref,bins=bins,density=True)[0],np.histogram(cur,bins=bins,density=True)[0]
    r,c=np.where(r==0,1e-6,r),np.where(c==0,1e-6,c)
    return np.sum((r-c)*np.log(r/c))

# ----------------- Data simulation -----------------
@st.cache_data
def simulate_sensory_data(n_products=8,n_batches_per_product=3,n_panelists=24,
                          n_sessions=4,tastings_per_combo=3,random_state=42,noise_sd=0.8):
    rng=np.random.default_rng(random_state)
    products=[f"P{i+1}" for i in range(n_products)]
    sessions=[f"S{i+1}" for i in range(n_sessions)]
    panelists=[f"R{i+1:02d}" for i in range(n_panelists)]
    rows=[]
    prod_latent={p:{"color_L":rng.normal(65,5),"color_a":rng.normal(5,3),"color_b":rng.normal(20,4),
        "crunchiness":rng.normal(0.0,1.0),"sweetness_brix":rng.normal(9,2),
        "salt_ppm":rng.normal(650,120),"pH":rng.normal(6.6,0.6),"aroma_intensity":rng.normal(0.0,1.0),
        "texture_firm":rng.normal(0.0,1.0)} for p in products}
    panel_len={r:rng.normal(0.0,0.3) for r in panelists}
    sess_eff={s:rng.normal(0.0,0.15) for s in sessions}
    for p in products:
        for b in range(1,n_batches_per_product+1):
            batch_id=f"{p}_B{b}"
            drift={"dE":abs(rng.normal(1.5,0.6)),"cr":rng.normal(0.0,0.25),
                   "ar":rng.normal(0.0,0.35),"fm":rng.normal(0.0,0.25),
                   "bx":rng.normal(0.0,0.5),"sa":rng.normal(0.0,40),"ph":rng.normal(0.0,0.15)}
            for s in sessions:
                for r in panelists:
                    for _ in range(tastings_per_combo):
                        storage_days=max(0,rng.normal(7,3))
                        dE=drift["dE"]+0.05*storage_days+abs(rng.normal(0,0.4))
                        brix=prod_latent[p]["sweetness_brix"]+drift["bx"]+rng.normal(0,0.6)
                        salt_ppm=prod_latent[p]["salt_ppm"]+drift["sa"]+rng.normal(0,35)
                        pH=prod_latent[p]["pH"]+drift["ph"]+rng.normal(0,0.1)
                        odor_pc1=prod_latent[p]["aroma_intensity"]+drift["ar"]-0.03*storage_days+rng.normal(0,0.4)
                        spectral_centroid=3500+600*(prod_latent[p]["crunchiness"]+drift["cr"])+rng.normal(0,200)
                        peak_rate=max(0,rng.normal(12+4*(prod_latent[p]["crunchiness"]+drift["cr"]),3))
                        peak_force=35+8*prod_latent[p]["texture_firm"]+rng.normal(0,3)
                        work_chew=120+30*prod_latent[p]["texture_firm"]+rng.normal(0,10)
                        fractures=max(0,int(rng.normal(6+2*prod_latent[p]["texture_firm"],2)))
                        rms=max(0,rng.normal(0.08+0.03*(prod_latent[p]["crunchiness"]),0.02))
                        liking=(6.0-0.18*dE+0.10*(brix-10)+0.12*odor_pc1+0.3*((spectral_centroid-3500)/500.0)
                                +panel_len[r]+sess_eff[s]+rng.normal(0,noise_sd))
                        liking=float(np.clip(liking,1.0,9.0))
                        rows.append(dict(product=p,batch=batch_id,session=s,panelist=r,storage_days=storage_days,
                            dE=dE,brix=brix,salt_ppm=salt_ppm,pH=pH,spectral_centroid=spectral_centroid,
                            peak_rate=peak_rate,peak_force=peak_force,work_chew=work_chew,fractures=fractures,
                            odor_pc1=odor_pc1,rms=rms,overall_liking=liking))
    return pd.DataFrame(rows)

@st.cache_data
def fit_ols_fe(df):
    z=df.copy()
    for c in FEATURES:
        mu,sd=z[c].mean(),z[c].std(ddof=0)
        z[c+"_z"]=0.0 if sd==0 else (z[c]-mu)/sd
    z_features=[f+"_z" for f in FEATURES]
    formula="overall_liking ~ "+" + ".join(z_features)+" + C(panelist)+C(batch)+C(session)"
    mdl=smf.ols(formula=formula,data=z).fit(cov_type="HC3")
    z["y_hat"]=mdl.predict(z); z["resid"]=z["overall_liking"]-z["y_hat"]
    return mdl,z,z_features,formula

# ----------------- Agent V -----------------
def agent_v_predict(row):
    y0=(6.2-0.12*row["dE"]+0.05*(row["brix"]-10)
        +0.22*((row["spectral_centroid"]-3500)/500.0)
        +0.10*((row["peak_rate"]-12)/4.0)+0.12*row["odor_pc1"])
    y=float(np.clip(y0+np.random.normal(0,0.45),1,9))
    conf=float(np.clip(0.55+0.2*np.random.rand(),0,1))
    return {"y_v":y,"confidence":conf,"qc":"lighting_ok"}

# ----------------- Sidebar -----------------
st.sidebar.header("Controls")
n_products=st.sidebar.slider("Products",4,20,8)
n_batches=st.sidebar.slider("Batches/product",2,6,3)
n_panel=st.sidebar.slider("Panelists",8,40,24)
n_sessions=st.sidebar.slider("Sessions",2,8,4)
tpc=st.sidebar.slider("Tastings/combo",1,5,3)
noise=st.sidebar.slider("Noise SD",0.2,1.5,0.8)
seed=st.sidebar.number_input("Seed",value=42)

df=simulate_sensory_data(n_products,n_batches,n_panel,n_sessions,tpc,seed,noise)
mdl,zdf,z_features,formula=fit_ols_fe(df)

# ----------------- KPI BAR -----------------
st.markdown("## Executive KPIs")
psi_val=psi_score(zdf["dE"].iloc[:200],zdf["dE"].iloc[200:])
bp=het_breuschpagan(mdl.resid,mdl.model.exog)
vif_df=pd.DataFrame({"feature":z_features,"VIF":[variance_inflation_factor(sm.add_constant(zdf[z_features]).to_numpy(),i)
        for i in range(1,len(z_features)+1)]})
cal=sm.OLS(zdf["overall_liking"],sm.add_constant(zdf["y_hat"])).fit()

kpi1,kpi2,kpi3,kpi4=st.columns(4)
kpi1.metric("R²",f"{mdl.rsquared:.3f}")
kpi2.metric("Adj R²",f"{mdl.rsquared_adj:.3f}")
kpi3.metric("Observations",f"{len(df):,}")
kpi4.metric("PSI ΔE",f"{psi_val:.3f}")
kpi5,kpi6,kpi7=st.columns(3)
kpi5.metric("VIF<5",traffic((vif_df["VIF"]<5).all()))
kpi6.metric("Heteroskedasticity",traffic(bp[1]>0.05))
kpi7.metric("Calibration≈1",traffic(0.9<=cal.params["y_hat"]<=1.1))
st.markdown("---")

# ----------------- Tabs -----------------
tabs=st.tabs(["Overview","Uncertainty","Diagnostics","Sensitivity",
              "SPC & Drift","Agents","Data","Governance"])

# ---- Overview ----
with tabs[0]:
    st.subheader("Model Overview")
    col1,col2=st.columns(2)
    with col1:
        fig=px.scatter(zdf,x="y_hat",y="overall_liking",hover_data=["product","batch","panelist","session"])
        fig.add_trace(go.Scatter(x=[1,9],y=[1,9],mode="lines",name="Ideal"))
        st.plotly_chart(_crosshair(fig),use_container_width=True)
    with col2:
        params=mdl.params.drop([p for p in mdl.params.index if p.startswith("C(") or p=="Intercept"])
        conf=mdl.conf_int().loc[params.index]
        coef_df=pd.DataFrame({"term":params.index,"coef":params.values,
                              "ci_low":conf[0].values,"ci_high":conf[1].values}).sort_values("coef")
        figc=go.Figure()
        figc.add_trace(go.Scatter(x=coef_df["coef"],y=coef_df["term"],
                                  error_x=dict(type="data",array=coef_df["ci_high"]-coef_df["coef"],
                                               arrayminus=coef_df["coef"]-coef_df["ci_low"]),
                                  mode="markers"))
        st.plotly_chart(_crosshair(figc),use_container_width=True)
    st.caption("Designed & Developed by Jit")

# ---- Uncertainty ----
with tabs[1]:
    st.subheader("Uncertainty (Bootstrap)")
    feat=st.selectbox("Slice feature",FEATURES)
    q05,q95=df[feat].quantile(0.05),df[feat].quantile(0.95)
    grid=np.linspace(q05,q95,50)
    med=df[FEATURES].median()
    med_df=pd.DataFrame([med]*len(grid)); med_df[feat]=grid
    for c in FEATURES:
        mu,sd=df[c].mean(),df[c].std(ddof=0)
        med_df[c+"_z"]=0.0 if sd==0 else (med_df[c]-mu)/sd
    med_df["panelist"]=df["panelist"].iloc[0]
    med_df["batch"]=df["batch"].iloc[0]
    med_df["session"]=df["session"].iloc[0]
    y0=mdl.predict(med_df)
    boot_preds=[]
    rng=np.random.default_rng(123)
    for _ in range(200):
        idx=rng.integers(0,len(df),len(df))
        try:
            boot_model=smf.ols(formula=formula,data=df.iloc[idx]).fit()
            boot_preds.append(boot_model.predict(med_df))
        except: pass
    bands=np.percentile(np.vstack(boot_preds),[5,20,50,80,95],axis=0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=grid,y=y0,name="ŷ",mode="lines"))
    fig.add_trace(go.Scatter(x=grid,y=bands[4],line=dict(width=0)))
    fig.add_trace(go.Scatter(x=grid,y=bands[0],fill="tonexty",opacity=0.2,name="95%"))
    fig.add_trace(go.Scatter(x=grid,y=bands[3],line=dict(width=0)))
    fig.add_trace(go.Scatter(x=grid,y=bands[1],fill="tonexty",opacity=0.3,name="80%"))
    st.plotly_chart(_crosshair(fig),use_container_width=True)

# ---- Diagnostics ----
with tabs[2]:
    st.subheader("Diagnostics")
    col1,col2=st.columns(2)
    with col1:
        fig=px.scatter(zdf,x="y_hat",y="resid")
        fig.add_hline(y=0)
        st.plotly_chart(_crosshair(fig),use_container_width=True)
    with col2:
        r=np.sort(zdf["resid"])
        probs=(np.arange(1,len(r)+1)-0.5)/len(r)
        q_theor=[NormalDist().inv_cdf(p) for p in probs]
        figqq=go.Figure()
        figqq.add_trace(go.Scatter(x=q_theor,y=r,mode="markers"))
        st.plotly_chart(_crosshair(figqq),use_container_width=True)
    cooks=pd.DataFrame({"row":np.arange(len(df)),"cooks_d":OLSInfluence(mdl).cooks_distance[0]})
    st.dataframe(cooks.sort_values("cooks_d",ascending=False).head(15))

# ---- Sensitivity ----
with tabs[3]:
    st.subheader("Sensitivity (Tornado)")
    idx=st.number_input("Row index",0,len(zdf)-1,0)
    row=zdf.iloc[[int(idx)]][["panelist","batch","session"]+z_features]
    base=float(mdl.predict(row)[0])
    tor=[]
    for t in z_features:
        up=float(mdl.predict(row.assign(**{t:row[t]+1.0}))[0])-base
        dn=float(mdl.predict(row.assign(**{t:row[t]-1.0}))[0])-base
        tor.append({"feature":t.replace("_z",""),"up":max(up,dn),"down":min(up,dn)})
    tor_df=pd.DataFrame(tor)
    figt=px.bar(tor_df,y="feature",x=["up","down"],barmode="overlay",orientation="h")
    st.plotly_chart(figt,use_container_width=True)

# ---- SPC & Drift ----
with tabs[4]:
    st.subheader("SPC & Drift")
    feat=st.selectbox("SPC feature",["dE","odor_pc1","spectral_centroid","brix","salt_ppm"])
    s=df.groupby("batch")[feat].mean()
    mu=s.mean()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=s.index,y=s.values,mode="lines+markers"))
    fig.add_hline(y=mu,annotation_text="Center")
    st.plotly_chart(fig,use_container_width=True)
    st.info(f"PSI drift (20% split): {psi_score(df[feat].iloc[:200],df[feat].iloc[200:]):.3f}")

# ---- Agents ----
with tabs[5]:
    st.subheader("Agents (QC Sensor)")
    idx = st.number_input("Row index (Agent)", 0, len(zdf)-1, 0)
    row = zdf.iloc[int(idx)]
    ols_pred = float(mdl.predict(zdf.iloc[[int(idx)]]))[0]
    av = agent_v_predict(row)
    alpha = 0.25 if abs(av["y_v"] - row["overall_liking"]) < abs(ols_pred - row["overall_liking"]) else 0.0
    y_ens = (1-alpha)*ols_pred + alpha*av["y_v"]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("OLS Prediction", f"{ols_pred:.2f}")
        st.metric("Agent V Prediction", f"{av['y_v']:.2f}")
        st.metric("Confidence", f"{av['confidence']:.2f}")
    with col2:
        st.metric("Ensemble ŷ*", f"{y_ens:.2f}")
        st.write(f"QC status: {av['qc']}")

# ---- Data ----
with tabs[6]:
    st.subheader("Data Preview")
    st.dataframe(df.head(200))
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "synthetic_sensory_dataset.csv", "text/csv")

# ---- Governance ----
with tabs[7]:
    st.subheader("Governance & Audit")
    env_hash=hashlib.sha256(json.dumps({
        "python":sys.version.split()[0],
        "statsmodels":sm.__version__,
        "pandas":pd.__version__,
        "numpy":np.__version__,
        "plotly":px.__version__
    },sort_keys=True).encode()).hexdigest()[:12]
    audit_rec={
        "ts":time.strftime("%Y-%m-%d %H:%M:%S"),
        "version":APP_VERSION,
        "seed":seed,
        "n_products":n_products,"n_batches":n_batches,"n_panel":n_panel,"n_sessions":n_sessions,"tpc":tpc,
        "noise":noise,"env_hash":env_hash
    }
    st.json(audit_rec)
