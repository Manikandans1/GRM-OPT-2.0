import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from preprocessing import preprocess_data
from garch_model import calculate_volatility
from lstm_model import train_lstm
from grm_engine import calculate_future_value, calculate_grm

st.set_page_config(page_title="GRM-OPT 2.0", layout="wide")

# ---------- CUSTOM STYLING ----------
st.markdown("""
<style>
.big-title {
    font-size:34px !important;
    font-weight:700;
}
.section-title {
    font-size:22px !important;
    font-weight:600;
    margin-top:35px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="big-title">GRM-OPT 2.0</div>', unsafe_allow_html=True)
st.markdown("Hybrid GARCH–LSTM Risk-Calibrated Wealth Forecasting System")

# ---------- SIDEBAR ----------
st.sidebar.header("Investment Parameters")
investment = st.sidebar.number_input("Monthly Investment (₹)", value=5000)
years = st.sidebar.number_input("Investment Duration (Years)", value=7)

markets = ["nifty50", "sp500", "dax40"]

if st.sidebar.button("Run Forecast"):

    results = []

    with st.spinner("Running Hybrid Forecast Model... Please wait..."):

        for market in markets:

            # Preprocess Data
            df, mean, std = preprocess_data(f"data/{market}.csv")

            # GARCH Volatility
            volatility = calculate_volatility(df['Log_Return'])

            # LSTM Prediction (Rolling Window 120 months)
            # normalized_prediction = train_lstm(df['Normalized_Return'].values)


            _, normalized_predictions = train_lstm(df['Normalized_Return'].values)

            normalized_prediction = normalized_predictions[-1]

            # Denormalize prediction
            predicted_return = (normalized_prediction * std) + mean

            # Wealth Projection
            fv = calculate_future_value(investment, predicted_return, years)

            # GRM Score
            grm_score = calculate_grm(fv, volatility)

            annual_return = predicted_return * 12 * 100

            results.append([
                market.upper(),
                predicted_return * 100,
                volatility * 100,
                annual_return,
                fv,
                grm_score
            ])

    # ---------- CREATE TABLE ----------
    result_df = pd.DataFrame(results, columns=[
        "Market",
        "Monthly Return (%)",
        "Volatility (%)",
        "Annual Return (%)",
        "Future Wealth (₹)",
        "GRM Score"
    ])

    result_df.sort_values(by="GRM Score", ascending=False, inplace=True)

    # ---------- KEY HIGHLIGHTS ----------
    st.markdown('<div class="section-title">📈 Key Highlights</div>', unsafe_allow_html=True)

    top_market = result_df.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Market", top_market["Market"])
    col2.metric("💰 Highest Wealth", f"₹ {top_market['Future Wealth (₹)']:,.2f}")
    col3.metric("📊 Best GRM Score", f"{top_market['GRM Score']:,.2f}")

    st.markdown("---")

    # ---------- COMPARISON TABLE ----------
    st.markdown('<div class="section-title">📋 Market Comparison Table</div>', unsafe_allow_html=True)

    st.dataframe(result_df.style.format({
        "Monthly Return (%)": "{:.2f}",
        "Volatility (%)": "{:.2f}",
        "Annual Return (%)": "{:.2f}",
        "Future Wealth (₹)": "₹ {:,.2f}",
        "GRM Score": "{:,.2f}"
    }))

    st.markdown("---")

    # ---------- WEALTH BAR CHART ----------
    st.markdown('<div class="section-title">💰 Projected Wealth Comparison</div>', unsafe_allow_html=True)

    fig1 = px.bar(
        result_df,
        x="Market",
        y="Future Wealth (₹)",
        color="Market",
        text_auto=True,
        template="plotly_dark"
    )

    fig1.update_layout(height=450)

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    

    # ---------- PROFESSIONAL RISK-RETURN CHART ----------
    st.markdown('<div class="section-title">⚖ Risk vs Return Positioning</div>', unsafe_allow_html=True)

    fig2 = px.scatter(
        result_df,
        x="Volatility (%)",
        y="Monthly Return (%)",
        color="Market",
        text="Market",
        template="plotly_dark"
    )

    fig2.update_traces(
        textposition='top center',
        marker=dict(size=14, line=dict(width=2, color='white'))
    )

    # Add reference mean lines
    mean_vol = result_df["Volatility (%)"].mean()
    mean_ret = result_df["Monthly Return (%)"].mean()

    fig2.add_hline(y=mean_ret, line_dash="dash", line_color="gray")
    fig2.add_vline(x=mean_vol, line_dash="dash", line_color="gray")

    fig2.update_layout(
        xaxis_title="Volatility (%)",
        yaxis_title="Monthly Return (%)",
        height=520,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.info("""
Interpretation:

• Top-left quadrant → Higher return, lower risk (Preferred zone)  
• Bottom-right quadrant → Lower return, higher risk (Avoid zone)  
• Dashed lines represent average market positioning.
""")

    st.markdown("---")

    # ---------- EXPLANATION POPUP ----------
    with st.expander("📘 Model Explanation (Click to Expand)"):
        st.write("""
**GARCH (1,1)** captures time-varying conditional volatility and volatility clustering.

**LSTM Network** models nonlinear temporal dependencies in financial return sequences.

**Rolling Window (120 months)** ensures adaptation to dynamic market regimes.

**Future Wealth** is calculated using compound SIP projection formula.

**GRM Score** normalizes projected wealth using predicted volatility to enable
cross-market risk-calibrated comparison.
""")

    st.success("Forecast Completed Successfully ✅")