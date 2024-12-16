# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from BondPortfolioOptimizer import BondPortfolioOptimizer

# Set page configuration
st.set_page_config(
    page_title="Bond Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and Description
st.subheader("Mathematical Model")
st.write(
    """
    The optimization in this application is based on a **Mean-Variance Optimization (MVO)** framework, 
    which aims to balance returns and risks in a portfolio. The mathematical formulation used is as follows:
    """
)

# Objective Function
st.subheader("Objective Function")
st.latex(r"Maximize \ R_{p} - \lambda \cdot \sigma_{p}^2")
st.write("Where:")
st.write(
    """
    - **Rₚ**: Expected portfolio return, calculated as:
    """
)
st.latex(r"R_{p} = \mathbf{w}^\top \mathbf{r}")
st.write(
    """
    - **w**: Weight vector of portfolio allocation.
    - **r**: Vector of individual bond yields.
    - **λ**: Risk aversion coefficient, a user-defined parameter to control the trade-off between return and risk.
    - **σₚ²**: Portfolio risk (variance), computed as:
    """
)
st.latex(r"\sigma_{p}^2 = \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}")
st.write("Where **\u03A3** is the covariance matrix of bond price changes.")

# Constraints
st.subheader("Constraints")
st.latex(
    r"""
    \begin{aligned}
    1. & \quad \sum_{i} w_{i} = 1 \quad \text{(Total portfolio weight must equal 1)} \\
    2. & \quad w_{i} \geq 0 \quad \text{(No short selling allowed)} \\
    3. & \quad w_{i} \leq \text{Max Position Size} \quad \text{(Limit on maximum exposure to any single bond)} \\
    4. & \quad R_{p} \geq \text{Min Return} \quad \text{(Minimum portfolio return constraint)} \\
    5. & \quad \mathbf{w}^\top \mathbf{q} \geq \text{Min Rating Score} \quad \text{(Minimum average rating score for the portfolio)} \\
    \end{aligned}
    """
)

# Sidebar for Data Upload
st.sidebar.header("1. Upload Data Files")

uploaded_bonds = st.sidebar.file_uploader("Upload Bonds.xlsx", type=["xlsx"])
uploaded_prices = st.sidebar.file_uploader("Upload HP.xlsx", type=["xlsx"])

if uploaded_bonds and uploaded_prices:
    # Read the uploaded Excel files
    bonds_df = pd.read_excel(uploaded_bonds)
    prices_df = pd.read_excel(uploaded_prices, parse_dates=['Dates'])

    st.sidebar.success("Data files uploaded successfully!")
else:
    st.sidebar.warning("Please upload both Bonds.xlsx and HP.xlsx files to proceed.")
    st.stop()

# Sidebar for Optimization Parameters
st.sidebar.header("2. Set Optimization Parameters")

total_investment = st.sidebar.number_input(
    "Total Investment (Rp)", 
    min_value=1000.0, 
    max_value=1e9, 
    value=1000000.0, 
    step=1000.0
)

min_return = st.sidebar.number_input(
    "Minimum Expected Return (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=7.0, 
    step=0.1
) / 100  # Convert to decimal

risk_aversion = st.sidebar.slider(
    "Risk Aversion", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.0, 
    step=0.1
)

max_position_size = st.sidebar.slider(
    "Maximum Position Size (%)", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.3, 
    step=0.01
)

min_rating_score = st.sidebar.slider(
    "Minimum Rating Score", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.7, 
    step=0.05
)

# Button to Optimize
st.sidebar.header("3. Run Optimization")
optimize_button = st.sidebar.button("Optimize Portfolio")

if optimize_button:
    with st.spinner("Optimizing portfolio..."):
        # Initialize the optimizer
        optimizer = BondPortfolioOptimizer(
            bonds_df=bonds_df,
            prices_df=prices_df
        )

        # Optimize the portfolio
        weights, results = optimizer.optimize_portfolio(
            total_investment=total_investment,
            min_return=min_return,
            risk_aversion=risk_aversion,
            max_position_size=max_position_size,
            min_rating_score=min_rating_score
        )

        # Generate report
        portfolio_data, summary_data, figures = optimizer.generate_report(weights)

    # Display Optimization Results
    st.success("Portfolio optimized successfully!")

    # Display Summary Metrics
    st.subheader("📊 Portfolio Summary")
    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
    summary_df = summary_df[summary_df['Metric'] != 'Analysis Date']  # Exclude Analysis Date
    st.table(summary_df.set_index('Metric'))

    # Display Detailed Portfolio Allocation
    st.subheader("📈 Portfolio Allocation")
    # Filter out bonds with negligible weights
    allocation_df = portfolio_data[portfolio_data['Weight (%)'] > 0.01].copy()
    # Convert Investment to Rupiah (assuming 1 USD = 15,860 IDR)
    allocation_df['Investment (Rp)'] = allocation_df['Investment ($)']
    allocation_df = allocation_df[[
        'Issuer', 'Ticker', 'Weight (%)', 'Yield (%)',
        'Duration (years)', 'Rating', 'Investment (Rp)', 'Standard Deviation'
    ]]
    allocation_df.rename(columns={
        'Issuer': 'Issuer',
        'Ticker': 'Ticker',
        'Weight (%)': 'Weight (%)',
        'Yield (%)': 'Yield (%)',
        'Duration (years)': 'Duration (years)',
        'Rating': 'Rating',
        'Investment (Rp)': 'Investment (Rp)',
        'Standard Deviation': 'Standard Deviation'
    }, inplace=True)
    st.dataframe(allocation_df.style.format({
        'Weight (%)': "{:.2f}",
        'Yield (%)': "{:.2f}",
        'Duration (years)': "{:.2f}",
        'Investment (Rp)': "{:,.0f}",
        'Standard Deviation': "{:.2f}"
    }))

    # Display Interactive Plots
    st.subheader("📊 Interactive Visualizations")

    # Portfolio Composition Sunburst
    st.plotly_chart(figures['composition'], use_container_width=True)

    # Risk-Return Bubble Chart
    st.plotly_chart(figures['risk_return'], use_container_width=True)

    # Rating Distribution Treemap
    st.plotly_chart(figures['ratings'], use_container_width=True)

    # The following sections have been removed:
    # - 💾 Download Interactive Report
    # - ✅ Optimization Details

else:
    st.info("Please upload your data files and set the optimization parameters, then click 'Optimize Portfolio' to begin.")
