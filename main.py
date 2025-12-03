import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Black-Litterman Model", layout="wide")

st.title("Black-Litterman Portfolio Optimization")
st.markdown("Build optimal portfolios using the Black-Litterman model with your market views")

# Sidebar inputs
st.sidebar.header("Portfolio Configuration")

# Stock selection
default_tickers = "AAPL,MSFT,GOOGL,AMZN,TSLA"
tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)

col1, col2 = st.sidebar.columns(2)
with col1:
    start = st.date_input("Start Date", start_date)
with col2:
    end = st.date_input("End Date", end_date)

# Model parameters
st.sidebar.subheader("Model Parameters")
risk_aversion = st.sidebar.slider("Risk Aversion (δ)", 0.1, 5.0, 2.5, 0.1)
tau = st.sidebar.slider("Uncertainty Scale (τ)", 0.01, 0.1, 0.025, 0.005)

st.sidebar.markdown("""
**Parameter Guide:**
- **Risk Aversion**: Higher = more conservative
- **Uncertainty Scale**: How much to trust market equilibrium (lower = trust more)
""")

# Fetch data button
fetch_data = st.sidebar.button("🔄 Fetch Data", type="primary")

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_list, start_dt, end_dt):
    """Download historical price data with robust error handling"""
    all_prices = {}
    valid_tickers = []
    
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_dt, end=end_dt, auto_adjust=True)
            
            if not hist.empty and 'Close' in hist.columns:
                all_prices[ticker] = hist['Close']
                valid_tickers.append(ticker)
            else:
                st.warning(f"No data for {ticker}")
                
        except Exception as e:
            st.warning(f"Error fetching {ticker}: {str(e)}")
    
    if not all_prices:
        return None, []
    
    # Combine into DataFrame
    df = pd.DataFrame(all_prices)
    
    # Clean data
    df = df.ffill().bfill()
    df = df.dropna(axis=1, how='all')
    
    return df, list(df.columns)

def calculate_metrics(prices_df):
    """Calculate financial metrics"""
    returns = prices_df.pct_change().dropna()
    
    # Annualized metrics
    ann_returns = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_returns / ann_vol
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    metrics_df = pd.DataFrame({
        'Annual Return': ann_returns,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    })
    
    return returns, metrics_df

def calculate_market_weights(prices_df):
    """Calculate market cap weights (simplified using current prices)"""
    last_prices = prices_df.iloc[-1]
    weights = last_prices / last_prices.sum()
    return weights.values

def black_litterman_model(pi, sigma, P, Q, omega, tau=0.025):
    """Black-Litterman implementation with numerical stability"""
    n = len(pi)
    
    pi_vec = np.array(pi).reshape(-1, 1)
    sigma_mat = np.array(sigma)
    
    # Calculate prior precision
    try:
        sigma_inv = np.linalg.inv(tau * sigma_mat)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.inv(tau * sigma_mat + np.eye(n) * 1e-6)
    
    try:
        omega_inv = np.linalg.inv(omega)
        M_mat = np.linalg.inv(sigma_inv + P.T @ omega_inv @ P)
        posterior_mean = M_mat @ (sigma_inv @ pi_vec + P.T @ omega_inv @ Q)
        posterior_cov = sigma_mat + M_mat
        
        return posterior_mean.flatten(), posterior_cov
        
    except np.linalg.LinAlgError:
        omega_inv = np.linalg.inv(omega + np.eye(omega.shape[0]) * 1e-6)
        M_mat = np.linalg.inv(sigma_inv + P.T @ omega_inv @ P + np.eye(n) * 1e-8)
        posterior_mean = M_mat @ (sigma_inv @ pi_vec + P.T @ omega_inv @ Q)
        posterior_cov = sigma_mat + M_mat
        
        return posterior_mean.flatten(), posterior_cov

def optimize_portfolio_with_constraints(expected_returns, cov_matrix, risk_free_rate=0.02):
    """Portfolio optimization with constraints"""
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        weights = inv_cov @ expected_returns
        
        # Ensure non-negative weights (long-only)
        weights = np.maximum(weights, 0)
        
        # Normalize to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
        
        # Calculate portfolio metrics
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return weights, portfolio_return, portfolio_vol, sharpe
        
    except np.linalg.LinAlgError:
        n = len(expected_returns)
        weights = np.ones(n) / n
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        return weights, portfolio_return, portfolio_vol, sharpe

def create_efficient_frontier(expected_returns, cov_matrix, n_points=50):
    """Create efficient frontier"""
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(expected_returns))
        
        # Minimum variance portfolio
        w_minvar = inv_cov @ ones
        w_minvar = w_minvar / w_minvar.sum() if w_minvar.sum() != 0 else ones / len(ones)
        
        # Maximum return portfolio
        w_maxret = inv_cov @ expected_returns
        w_maxret = w_maxret / w_maxret.sum() if w_maxret.sum() != 0 else ones / len(ones)
        
        # Create frontier
        returns_frontier = []
        volatilities_frontier = []
        
        for alpha in np.linspace(-0.5, 1.5, n_points):
            w = alpha * w_maxret + (1 - alpha) * w_minvar
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
                ret = w @ expected_returns
                vol = np.sqrt(w @ cov_matrix @ w)
                returns_frontier.append(ret)
                volatilities_frontier.append(vol)
        
        return returns_frontier, volatilities_frontier
        
    except:
        return [], []

# Main app logic
if fetch_data or 'prices_df' not in st.session_state:
    if len(tickers) >= 2:
        with st.spinner(f"Downloading data for {len(tickers)} stocks..."):
            prices_data, valid_tickers = fetch_stock_data(tickers, start, end)
            
            if prices_data is not None and not prices_data.empty and len(valid_tickers) >= 2:
                st.session_state.prices_df = prices_data
                st.session_state.tickers = valid_tickers
                st.success(f"✅ Successfully downloaded data for {len(valid_tickers)} stocks")
                tickers = valid_tickers
            else:
                st.error("Failed to fetch sufficient data. Please check tickers and date range.")
                if 'prices_df' in st.session_state:
                    del st.session_state.prices_df
    else:
        st.error("Please enter at least 2 valid ticker symbols.")

if 'prices_df' in st.session_state and 'tickers' in st.session_state:
    prices_df = st.session_state.prices_df
    tickers = st.session_state.tickers
    n_assets = len(tickers)
    
    # Calculate returns and metrics
    returns_df, metrics_df = calculate_metrics(prices_df)
    cov_matrix = returns_df.cov() * 252
    
    # Calculate market weights and implied returns
    market_weights = calculate_market_weights(prices_df)
    pi = risk_aversion * cov_matrix.values @ market_weights
    
    # Display data section
    st.header("1️⃣ Historical Data & Statistics")
    
    tab1, tab2, tab3 = st.tabs(["Price History", "Returns Analysis", "Correlation Matrix"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            normalized_prices = prices_df / prices_df.iloc[0] * 100
            fig = px.line(normalized_prices, title="Normalized Price Performance (Base 100)")
            fig.update_layout(yaxis_title="Normalized Price", xaxis_title="Date", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Number of Assets", n_assets)
            st.metric("Time Period", f"{len(prices_df)} days")
            st.metric("Start Date", prices_df.index[0].strftime('%Y-%m-%d'))
            st.metric("End Date", prices_df.index[-1].strftime('%Y-%m-%d'))
    
    with tab2:
        formatted_metrics = metrics_df.copy()
        formatted_metrics['Annual Return'] = formatted_metrics['Annual Return'].apply(lambda x: f"{x:.2%}")
        formatted_metrics['Annual Volatility'] = formatted_metrics['Annual Volatility'].apply(lambda x: f"{x:.2%}")
        formatted_metrics['Sharpe Ratio'] = formatted_metrics['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        formatted_metrics['Max Drawdown'] = formatted_metrics['Max Drawdown'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(formatted_metrics, use_container_width=True)
        
        fig = px.box(returns_df, title="Daily Returns Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        corr_matrix = returns_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(title="Returns Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Market Equilibrium
    st.header("2️⃣ Market Equilibrium Returns")
    st.markdown("Returns implied by current market equilibrium (CAPM-based).")
    
    pi_df = pd.DataFrame({
        'Market Weight': [f"{w:.2%}" for w in market_weights],
        'Implied Return': [f"{r:.2%}" for r in pi]
    }, index=tickers)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(pi_df, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tickers, y=pi * 100, name='Implied Returns', marker_color='lightblue'))
        fig.add_trace(go.Scatter(x=tickers, y=market_weights * 100, name='Market Weights (%)', 
                                yaxis='y2', line=dict(color='red', width=2), mode='lines+markers'))
        fig.update_layout(
            title="Market Implied Returns & Weights",
            yaxis=dict(title="Implied Return (%)"),
            yaxis2=dict(title="Market Weight (%)", overlaying='y', side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User Views
    st.header("3️⃣ Your Market Views")
    
    view_type = st.selectbox(
        "Select view type:",
        ["Absolute Views", "Relative Views"],
        key="view_type_select"
    )
    
    if view_type == "Absolute Views":
        st.subheader("Absolute Expected Returns")
        
        n_cols = min(4, n_assets)
        views = np.zeros(n_assets)
        confidences = np.ones(n_assets) * 0.5
        
        cols = st.columns(n_cols)
        
        for idx, ticker in enumerate(tickers):
            col_idx = idx % n_cols
            with cols[col_idx]:
                st.markdown(f"**{ticker}**")
                default_view = float(pi[idx] * 100)
                
                view = st.number_input(
                    "Expected Return (%)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=default_view,
                    step=1.0,
                    key=f"view_abs_{ticker}_{idx}",
                    help=f"Current implied: {default_view:.1f}%"
                )
                views[idx] = view / 100
                
                confidence = st.slider(
                    "Confidence",
                    0, 100, 50,
                    key=f"conf_abs_{ticker}_{idx}",
                    label_visibility="collapsed"
                )
                confidences[idx] = confidence / 100
        
        P_matrix = np.eye(n_assets)
        Q_vector = views.reshape(-1, 1)
        
    else:
        st.subheader("Relative Views")
        st.info("Example: 'AAPL will outperform MSFT by 5%'")
        
        n_views = st.slider("Number of relative views", 1, min(5, n_assets), 2)
        
        P_matrix = np.zeros((n_views, n_assets))
        Q_vector = np.zeros((n_views, 1))
        confidences = []
        
        for i in range(n_views):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                asset_a = st.selectbox(f"Asset A (View {i+1})", tickers, key=f"asset_a_{i}")
                idx_a = tickers.index(asset_a)
            with col2:
                default_b_idx = (idx_a + 1) % n_assets
                asset_b = st.selectbox(f"Asset B (View {i+1})", tickers, index=default_b_idx, key=f"asset_b_{i}")
                idx_b = tickers.index(asset_b)
            with col3:
                outperformance = st.number_input(f"Outperformance (%)", -50.0, 50.0, 0.0, 0.5, key=f"outperf_{i}")
                Q_vector[i] = outperformance / 100
            with col4:
                confidence = st.slider("Conf", 0, 100, 50, key=f"rel_conf_{i}", label_visibility="collapsed")
                confidences.append(confidence / 100)
            
            P_matrix[i, idx_a] = 1
            P_matrix[i, idx_b] = -1
    
    # Calculate uncertainty matrix
    omega_diag = []
    for i, conf in enumerate(confidences):
        if len(confidences) == n_assets:
            view_variance = cov_matrix.iloc[i, i] if i < len(cov_matrix) else 0.04
        else:
            view_variance = P_matrix[i] @ cov_matrix.values @ P_matrix[i].T if i < P_matrix.shape[0] else 0.04
        
        uncertainty = (1 - conf) * tau * max(view_variance, 1e-6)
        omega_diag.append(max(uncertainty, 1e-8))
    
    omega_matrix = np.diag(omega_diag)
    
    # Calculate BL Portfolio
    if st.button("🚀 Calculate Black-Litterman Portfolio", type="primary", use_container_width=True):
        st.header("4️⃣ Black-Litterman Results")
        
        with st.spinner("Calculating optimal portfolio..."):
            # BL posterior
            bl_returns, bl_cov = black_litterman_model(pi, cov_matrix.values, P_matrix, Q_vector, omega_matrix, tau)
            
            # Optimize
            bl_weights, bl_return, bl_vol, bl_sharpe = optimize_portfolio_with_constraints(bl_returns, cov_matrix.values)
            
            # Benchmarks
            mkt_return = market_weights @ pi
            mkt_vol = np.sqrt(market_weights @ cov_matrix.values @ market_weights)
            mkt_sharpe = mkt_return / mkt_vol if mkt_vol > 0 else 0
            
            eq_weights = np.ones(n_assets) / n_assets
            eq_return = eq_weights @ pi
            eq_vol = np.sqrt(eq_weights @ cov_matrix.values @ eq_weights)
            eq_sharpe = eq_return / eq_vol if eq_vol > 0 else 0
        
        # Results tabs
        result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
            "Portfolio Allocation", "Returns Comparison", "Performance Metrics", "Efficient Frontier"
        ])
        
        with result_tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                allocation_df = pd.DataFrame({
                    'Ticker': tickers,
                    'BL Weight': bl_weights,
                    'Market Weight': market_weights,
                    'Difference': bl_weights - market_weights
                }).sort_values('BL Weight', ascending=False)
                
                fig = go.Figure(data=[go.Pie(
                    labels=allocation_df['Ticker'],
                    values=allocation_df['BL Weight'],
                    hole=0.4,
                    textinfo='label+percent'
                )])
                fig.update_layout(title="Black-Litterman Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Portfolio Comparison")
                formatted_alloc = allocation_df.copy()
                formatted_alloc['BL Weight'] = formatted_alloc['BL Weight'].apply(lambda x: f"{x:.2%}")
                formatted_alloc['Market Weight'] = formatted_alloc['Market Weight'].apply(lambda x: f"{x:.2%}")
                formatted_alloc['Difference'] = formatted_alloc['Difference'].apply(lambda x: f"{x:+.2%}")
                st.dataframe(formatted_alloc.set_index('Ticker'), use_container_width=True)
                
                active_share = np.sum(np.abs(allocation_df['Difference'])) / 2
                st.metric("Active Share", f"{active_share:.1%}")
        
        with result_tab2:
            returns_comparison = pd.DataFrame({
                'Market Implied': [f"{r:.2%}" for r in pi],
                'BL Posterior': [f"{r:.2%}" for r in bl_returns],
                'Difference': [f"{(bl_returns[i] - pi[i]):.2%}" for i in range(len(pi))]
            }, index=tickers)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(returns_comparison, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=tickers, y=pi * 100, name='Market', marker_color='lightblue'))
                fig.add_trace(go.Bar(x=tickers, y=bl_returns * 100, name='BL', marker_color='salmon'))
                fig.update_layout(title="Expected Returns Comparison", yaxis_title="Annual Return (%)", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        with result_tab3:
            metrics_data = {
                'Metric': ['Expected Return', 'Expected Volatility', 'Sharpe Ratio'],
                'Black-Litterman': [f"{bl_return:.2%}", f"{bl_vol:.2%}", f"{bl_sharpe:.2f}"],
                'Market Portfolio': [f"{mkt_return:.2%}", f"{mkt_vol:.2%}", f"{mkt_sharpe:.2f}"],
                'Equal Weight': [f"{eq_return:.2%}", f"{eq_vol:.2%}", f"{eq_sharpe:.2f}"]
            }
            
            st.dataframe(pd.DataFrame(metrics_data).set_index('Metric'), use_container_width=True)
        
        with result_tab4:
            frontier_returns, frontier_vols = create_efficient_frontier(bl_returns, bl_cov)
            
            if frontier_returns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frontier_vols, y=frontier_returns, mode='lines', 
                                        name='Efficient Frontier', line=dict(color='gray', width=2)))
                
                portfolios = [
                    ('BL Portfolio', bl_vol, bl_return, 'green'),
                    ('Market', mkt_vol, mkt_return, 'blue'),
                    ('Equal Weight', eq_vol, eq_return, 'orange')
                ]
                
                for name, vol, ret, color in portfolios:
                    fig.add_trace(go.Scatter(x=[vol], y=[ret], mode='markers+text', name=name,
                                           marker=dict(size=15, color=color), text=[name], textposition="top center"))
                
                fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Return")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Enter at least 2 stock tickers in the sidebar and click 'Fetch Data' to begin")
    
    with st.expander("📚 Instructions", expanded=True):
        st.markdown("""
        ### How to use:
        
        1. **Enter Stock Tickers** (comma-separated)
        2. **Select Date Range** (2+ years recommended)
        3. **Adjust Model Parameters**
        4. **Click 'Fetch Data'**
        5. **Enter Your Views**
        6. **Calculate Portfolio**
        
        ### Example Tickers:
        - Tech: AAPL, MSFT, GOOGL, AMZN, META
        - Banking: JPM, BAC, WFC, C, GS
        - Healthcare: JNJ, PFE, MRK, UNH
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Black-Litterman Portfolio Optimizer | Educational Tool | Not Financial Advice</p>
</div>
""", unsafe_allow_html=True)