import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="VALUEX // AI Real Estate Valuation & Investment Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enterprise UI styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #080c14;
    }

    /* Top Command Header */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .brand-title {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .system-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38bdf8;
    }

    /* Metric Deck */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .kpi-tag {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
    }
    .kpi-metric {
        font-size: 26px;
        font-weight: 700;
        margin-top: 6px;
        letter-spacing: -0.5px;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Content Cards */
    .card-panel {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .val-banner {
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 4px solid #38bdf8;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("models", "prop_val_xgb.pkl")

@st.cache_resource
def load_engine():
    if not os.path.exists(MODEL_PATH):
        import train_model
        train_model.train_and_export()
    return joblib.load(MODEL_PATH)

engine = load_engine()
model = engine["model"]
feature_cols = engine["features"]

# Top Header Bar
st.markdown("""
<div class="top-nav">
    <div class="brand-title">
        <span>🏢</span> VALUEX // Institutional Asset Valuation & Underwriting Desk
    </div>
    <div class="system-status">
        <span class="pulse-dot"></span> AVM CORE: ACTIVE (XGBOOST REGRESSOR)
    </div>
</div>
""", unsafe_allow_html=True)

tab_batch, tab_single = st.tabs(["📁 Batch Portfolio Valuation", "🎯 Individual Asset Appraiser"])

# -------------------------------------------------------------
# TAB 1: BATCH INGESTION & PORTFOLIO VALUATION
# -------------------------------------------------------------
with tab_batch:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown("##### Portfolio Telemetry Ingestion Hub")
    st.caption("Upload property inventory (.xlsx or .csv) to compute automated fair-market valuations and cap-rate metrics.")
    
    uploaded_file = st.file_uploader(
        "Upload Telemetry File",
        type=["xlsx", "xls", "csv"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    df_raw = None
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.toast(f"Successfully loaded {len(df_raw):,} asset records", icon="🏢")
        except Exception as e:
            st.error(f"File ingestion error: {e}")
    else:
        sample_path = os.path.join("data", "properties_1000.xlsx")
        if os.path.exists(sample_path):
            df_raw = pd.read_excel(sample_path)
            st.caption(f"Displaying verified baseline inventory: `{sample_path}` ({len(df_raw):,} records)")

    if df_raw is not None:
        # Encode categoricals using dummy variables
        df_encoded = pd.get_dummies(df_raw, columns=['Locality', 'Property_Type'], drop_first=True)
        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        preds = model.predict(df_encoded[feature_cols])
        df_raw['AI_Valuation_Lakhs'] = np.round(preds, 2)
        df_raw['Valuation_Per_Sqft'] = np.round((df_raw['AI_Valuation_Lakhs'] * 100000) / df_raw['Total_Sqft'], 0)
        
        # Portfolio KPIs
        total_portfolio_cr = df_raw['AI_Valuation_Lakhs'].sum() / 100.0
        avg_rate_sqft = df_raw['Valuation_Per_Sqft'].mean()
        avg_price_lakhs = df_raw['AI_Valuation_Lakhs'].mean()
        top_locality = df_raw['Locality'].mode()[0]

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-box">
                <div class="kpi-tag">Total Portfolio Asset Value</div>
                <div class="kpi-metric">₹{total_portfolio_cr:,.2f} Cr</div>
            </div>
            <div class="kpi-box" style="border-left: 3px solid #38bdf8;">
                <div class="kpi-tag">Mean Asset Pricing</div>
                <div class="kpi-metric" style="color: #38bdf8;">₹{avg_price_lakhs:,.1f} L</div>
            </div>
            <div class="kpi-box" style="border-left: 3px solid #10b981;">
                <div class="kpi-tag">Average Unit Rate</div>
                <div class="kpi-metric" style="color: #10b981;">₹{avg_rate_sqft:,.0f} / sq.ft</div>
            </div>
            <div class="kpi-box" style="border-left: 3px solid #f59e0b;">
                <div class="kpi-tag">Dominant Locality</div>
                <div class="kpi-metric" style="color: #f59e0b;">{top_locality}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown('<div class="card-panel">', unsafe_allow_html=True)
            st.markdown("##### Mean Valuation by Locality (₹ Lakhs)")
            loc_val = df_raw.groupby('Locality')['AI_Valuation_Lakhs'].mean().sort_values(ascending=False)
            st.bar_chart(loc_val, color=["#38bdf8"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c2:
            st.markdown('<div class="card-panel">', unsafe_allow_html=True)
            st.markdown("##### Property Format Valuation Comparison")
            type_val = df_raw.groupby('Property_Type')['AI_Valuation_Lakhs'].mean()
            st.bar_chart(type_val, color=["#10b981"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown("##### Real Estate Valuation Ledger")
        st.dataframe(
            df_raw[['Property_ID', 'Locality', 'Property_Type', 'BHK', 'Total_Sqft', 'Valuation_Per_Sqft', 'AI_Valuation_Lakhs']],
            width='stretch',
            height=340
        )
        csv_ledger = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Portfolio Valuation Ledger (CSV)",
            data=csv_ledger,
            file_name=f"property_valuation_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: INDIVIDUAL ASSET VALUATION & INVESTMENT CALCULATOR
# -------------------------------------------------------------
with tab_single:
    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown("##### Property Physical & Locational Attributes")
        loc_input = st.selectbox("Market Micro-Locality", ['Hitec City', 'Gachibowli', 'Kondapur', 'Madhapur', 'Banjara Hills', 'Jubilee Hills', 'Kukatpally', 'Miyapur'])
        type_input = st.selectbox("Structural Classification", ['Apartment', 'Gated Villa', 'Independent Floor'])
        sqft_input = st.number_input("Carpet & Built-Up Area (Sq.Ft)", min_value=400, max_value=12000, value=1850, step=50)
        bhk_input = st.slider("BHK Layout", 1, 5, 3)
        baths_input = st.slider("Bathroom Configuration", 1, 6, 3)
        floor_input = st.slider("Floor Level", 0, 35, 7)
        age_input = st.slider("Structural Age (Years)", 0, 30, 2)
        metro_input = st.slider("Proximity to Transit / Metro (KM)", 0.2, 10.0, 1.2, step=0.1)
        gated_input = st.radio("Access-Controlled Gated Society?", ["Yes", "No"], horizontal=True)
        amenity_input = st.slider("Amenity & Clubhouse Index (1-10)", 1, 10, 8)

        calc_btn = st.button("Generate Appraisal Valuation", type="primary", width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown("##### Automated Valuation Model (AVM) Output")

        if calc_btn:
            input_df = pd.DataFrame([{
                'BHK': bhk_input,
                'Total_Sqft': sqft_input,
                'Bathrooms': baths_input,
                'Floor_Level': floor_input,
                'Property_Age_Years': age_input,
                'Proximity_To_Metro_KM': metro_input,
                'Gated_Community': 1 if gated_input == "Yes" else 0,
                'Amenity_Rating': amenity_input,
                'Locality': loc_input,
                'Property_Type': type_input
            }])

            encoded_input = pd.get_dummies(input_df, columns=['Locality', 'Property_Type'], drop_first=True)
            for c in feature_cols:
                if c not in encoded_input.columns:
                    encoded_input[c] = 0

            est_price = float(model.predict(encoded_input[feature_cols])[0])
            price_sqft = round((est_price * 100000) / sqft_input, 0)
            ci_low = round(est_price * 0.94, 2)
            ci_high = round(est_price * 1.06, 2)
            est_rent = round((est_price * 100000 * 0.032) / 12, 0)

            st.markdown(f"""
            <div class="val-banner">
                <span style="font-size: 11px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;">ESTIMATED FAIR MARKET VALUE</span>
                <h2 style="color: #38bdf8; margin: 4px 0 0 0; font-family: 'JetBrains Mono';">₹{est_price:,.2f} Lakhs</h2>
                <p style="color: #94a3b8; font-size: 13px; margin-top: 4px;">Valuation Band (95% CI): <strong>₹{ci_low:,.2f} L – ₹{ci_high:,.2f} L</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            - **Unit Base Rate:** ₹{price_sqft:,.0f} / sq.ft
            - **Estimated Monthly Rental Yield:** ₹{est_rent:,.0f} / month (~3.2% gross cap rate)
            - **Market Zone:** {loc_input} prime IT / residential corridor
            - **Transit Accessibility:** {metro_input} km to nearest transit hub
            """)

            memo = {
                "dossier_id": f"AVM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "asset_specifications": input_df.to_dict(orient="records")[0],
                "fair_valuation_lakhs": est_price,
                "confidence_band": {"low_lakhs": ci_low, "high_lakhs": ci_high},
                "derived_metrics": {
                    "price_per_sqft": price_sqft,
                    "estimated_monthly_rent": est_rent
                }
            }

            st.download_button(
                label="📥 Download Valuation Memorandum (JSON)",
                data=json.dumps(memo, indent=4),
                file_name=f"{memo['dossier_id']}.json",
                mime="application/json",
                width='stretch'
            )
        else:
            st.info("Input asset specifications on the left and click **Generate Appraisal Valuation**.")
        st.markdown('</div>', unsafe_allow_html=True)
