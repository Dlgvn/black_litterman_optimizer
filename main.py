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

# Stock selection with multi-select
st.sidebar.subheader("Stock Selection")
popular_tickers = {
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABT", "TMO"],
    "Consumer": ["WMT", "KO", "PG", "PEP", "MCD", "DIS"]
}

all_popular = []
for category in popular_tickers.values():
    all_popular.extend(category)

# Default selection
default_selection = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Manual input option
input_method = st.sidebar.radio(
    "Stock selection method:",
    ["Select from popular stocks", "Enter manually"],
    index=0
)

if input_method == "Select from popular stocks":
    selected_tickers = st.sidebar.multiselect(
        "Select stocks:",
        all_popular,
        default=default_selection
    )
    # Allow adding custom tickers
    custom_tickers = st.sidebar.text_input(
        "Add custom tickers (comma-separated):",
        placeholder="e.g., BRK-B, V, MA"
    )
    if custom_tickers:
        custom_list = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        selected_tickers = list(set(selected_tickers + custom_list))
else:
    tickers_input = st.sidebar.text_input(
        "Enter stock tickers (comma-separated):",
        value=", ".join(default_selection),
        help="Enter tickers like: AAPL, MSFT, GOOGL"
    )
    selected_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Display selected tickers
if selected_tickers:
    st.sidebar.info(f"Selected: {', '.join(selected_tickers)}")

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
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
    help="Annual risk-free rate in percentage"
) / 100  # Convert to decimal

st.sidebar.markdown("""
**Parameter Guide:**
- **Risk Aversion**: Higher = more conservative
- **Uncertainty Scale**: How much to trust market equilibrium (lower = trust more)
- **Risk-Free Rate**: Used for Sharpe ratio calculation
""")

# Fetch data button
fetch_data = st.sidebar.button("Fetch Data", type="primary")

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker_list, start_dt, end_dt):
    """Download historical price data with market cap information"""
    all_prices = {}
    all_market_caps = {}
    valid_tickers = []
    
    # First pass: get basic data
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_dt, end=end_dt, auto_adjust=True)
            
            if not hist.empty and 'Close' in hist.columns:
                all_prices[ticker] = hist['Close']
                valid_tickers.append(ticker)
            else:
                st.warning(f"No price data for {ticker}")
                
        except Exception as e:
            st.warning(f"Error fetching {ticker}: {str(e)}")
    
    if not all_prices:
        return None, [], {}, {}
    
    # Combine into DataFrame
    prices_df = pd.DataFrame(all_prices)
    
    # Clean data
    prices_df = prices_df.ffill().bfill()
    prices_df = prices_df.dropna(axis=1, how='all')
    
    # Get market caps for valid tickers
    valid_tickers = list(prices_df.columns)
    market_caps = {}
    
    for ticker in valid_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try different market cap fields
            cap_fields = ['marketCap', 'totalAssets', 'enterpriseValue']
            market_cap = None
            
            for field in cap_fields:
                if field in info and info[field] is not None:
                    market_cap = info[field]
                    break
            
            if market_cap and market_cap > 0:
                market_caps[ticker] = market_cap
            else:
                # Fallback: use price * shares outstanding
                if 'sharesOutstanding' in info and info['sharesOutstanding']:
                    shares = info['sharesOutstanding']
                    current_price = prices_df[ticker].iloc[-1]
                    market_caps[ticker] = shares * current_price
                else:
                    # Use average if market cap not available
                    avg_price = prices_df[ticker].mean()
                    market_caps[ticker] = avg_price * 1e9  # Default estimate
                    
        except Exception as e:
            st.warning(f"Could not get market cap for {ticker}: {str(e)}")
            # Use price-based estimate as fallback
            avg_price = prices_df[ticker].mean()
            market_caps[ticker] = avg_price * 1e9
    
    return prices_df, valid_tickers, market_caps

def calculate_metrics(prices_df, risk_free_rate=0.02):
    """Calculate financial metrics"""
    returns = prices_df.pct_change().dropna()
    
    # Annualized metrics
    ann_returns = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_returns - risk_free_rate) / ann_vol
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # Calculate daily returns statistics
    daily_stats = returns.describe().T
    
    metrics_df = pd.DataFrame({
        'Annual Return': ann_returns,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Daily Mean': daily_stats['mean'],
        'Daily Std': daily_stats['std']
    })
    
    return returns, metrics_df

def calculate_market_weights_from_caps(market_caps):
    """Calculate market cap weights from actual market capitalization"""
    if not market_caps:
        return None
    
    # Convert to numpy array and normalize
    caps_array = np.array(list(market_caps.values()))
    weights = caps_array / caps_array.sum()
    return weights

def calculate_price_based_weights(prices_df):
    """Fallback: calculate weights based on current prices"""
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

def create_efficient_frontier(expected_returns, cov_matrix, risk_free_rate=0.02, n_points=50):
    """Create efficient frontier with risk-free rate"""
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(expected_returns))
        
        # Minimum variance portfolio
        w_minvar = inv_cov @ ones
        w_minvar = w_minvar / w_minvar.sum() if w_minvar.sum() != 0 else ones / len(ones)
        
        # Maximum Sharpe ratio portfolio (tangency portfolio)
        excess_returns = expected_returns - risk_free_rate
        w_tangency = inv_cov @ excess_returns
        w_tangency = np.maximum(w_tangency, 0)  # Long-only constraint
        if w_tangency.sum() > 0:
            w_tangency = w_tangency / w_tangency.sum()
        else:
            w_tangency = w_minvar
        
        # Create frontier
        returns_frontier = []
        volatilities_frontier = []
        sharpe_frontier = []
        
        for alpha in np.linspace(0, 2, n_points):
            w = alpha * w_tangency + (1 - alpha) * w_minvar
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
                ret = w @ expected_returns
                vol = np.sqrt(w @ cov_matrix @ w)
                sharpe_val = (ret - risk_free_rate) / vol if vol > 0 else 0
                returns_frontier.append(ret)
                volatilities_frontier.append(vol)
                sharpe_frontier.append(sharpe_val)
        
        return returns_frontier, volatilities_frontier, sharpe_frontier
        
    except:
        return [], [], []

# Main app logic
if fetch_data or 'prices_df' not in st.session_state:
    if len(selected_tickers) >= 2:
        with st.spinner(f"Downloading data for {len(selected_tickers)} stocks..."):
            prices_data, valid_tickers, market_caps = fetch_stock_data(selected_tickers, start, end)
            
            if prices_data is not None and not prices_data.empty and len(valid_tickers) >= 2:
                st.session_state.prices_df = prices_data
                st.session_state.tickers = valid_tickers
                st.session_state.market_caps = market_caps
                st.success(f" Successfully downloaded data for {len(valid_tickers)} stocks")
            else:
                st.error("Failed to fetch sufficient data. Please check tickers and date range.")
                if 'prices_df' in st.session_state:
                    del st.session_state.prices_df
    else:
        st.error("Please select at least 2 valid ticker symbols.")

if 'prices_df' in st.session_state and 'tickers' in st.session_state:
    prices_df = st.session_state.prices_df
    tickers = st.session_state.tickers
    market_caps = st.session_state.get('market_caps', {})
    n_assets = len(tickers)
    
    # Calculate returns and metrics
    returns_df, metrics_df = calculate_metrics(prices_df, risk_free_rate)
    cov_matrix = returns_df.cov() * 252
    
    # Calculate market weights from market caps if available
    if market_caps and len(market_caps) == len(tickers):
        market_weights = calculate_market_weights_from_caps(market_caps)
        if market_weights is not None:
            st.sidebar.success("Using actual market cap weights")
        else:
            market_weights = calculate_price_based_weights(prices_df)
            st.sidebar.warning("Using price-based weights (market caps not available for all stocks)")
    else:
        market_weights = calculate_price_based_weights(prices_df)
        st.sidebar.warning("Using price-based weights (market caps not available)")
    
    pi = risk_aversion * cov_matrix.values @ market_weights
    
    # Display data section
    st.header("Historical Data & Statistics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Movement", "Returns Analysis", "Correlation Matrix", "Market Caps"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            # Price movement chart (actual prices)
            fig = go.Figure()
            for ticker in tickers:
                fig.add_trace(go.Scatter(
                    x=prices_df.index,
                    y=prices_df[ticker],
                    name=ticker,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Stock Price Movement",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Number of Assets", n_assets)
            st.metric("Time Period", f"{len(prices_df)} days")
            st.metric("Start Date", prices_df.index[0].strftime('%Y-%m-%d'))
            st.metric("End Date", prices_df.index[-1].strftime('%Y-%m-%d'))
            st.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")
            
            # Show current prices
            st.subheader("Current Prices")
            last_prices = prices_df.iloc[-1]
            for ticker in tickers[:5]:  # Show first 5
                st.write(f"{ticker}: ${last_prices[ticker]:.2f}")
            if len(tickers) > 5:
                with st.expander("Show all prices"):
                    for ticker in tickers[5:]:
                        st.write(f"{ticker}: ${last_prices[ticker]:.2f}")
    
    with tab2:
        formatted_metrics = metrics_df.copy()
        formatted_metrics['Annual Return'] = formatted_metrics['Annual Return'].apply(lambda x: f"{x:.2%}")
        formatted_metrics['Annual Volatility'] = formatted_metrics['Annual Volatility'].apply(lambda x: f"{x:.2%}")
        formatted_metrics['Sharpe Ratio'] = formatted_metrics['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        formatted_metrics['Max Drawdown'] = formatted_metrics['Max Drawdown'].apply(lambda x: f"{x:.2%}")
        formatted_metrics['Daily Mean'] = formatted_metrics['Daily Mean'].apply(lambda x: f"{x:.4%}")
        formatted_metrics['Daily Std'] = formatted_metrics['Daily Std'].apply(lambda x: f"{x:.4%}")
        
        st.dataframe(formatted_metrics, use_container_width=True)
        
        # Daily returns distribution
        fig = px.box(returns_df, title="Daily Returns Distribution")
        fig.update_layout(height=400)
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
        fig.update_layout(
            title="Returns Correlation Matrix",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if market_caps:
            # Create market cap DataFrame
            cap_df = pd.DataFrame({
                'Ticker': list(market_caps.keys()),
                'Market Cap ($)': list(market_caps.values())
            })
            cap_df['Market Cap (Billions)'] = cap_df['Market Cap ($)'] / 1e9
            cap_df = cap_df.sort_values('Market Cap ($)', ascending=False)
            
            # Format for display
            display_df = cap_df.copy()
            display_df['Market Cap (Billions)'] = display_df['Market Cap (Billions)'].apply(lambda x: f"${x:,.1f}B")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(display_df[['Ticker', 'Market Cap (Billions)']].set_index('Ticker'), 
                           use_container_width=True)
            
            with col2:
                # Market cap pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=cap_df['Ticker'],
                    values=cap_df['Market Cap ($)'],
                    hole=0.4,
                    textinfo='label+percent',
                    hoverinfo='label+value+percent'
                )])
                fig.update_layout(title="Market Capitalization Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Market capitalization data not available for all stocks.")
    
    # Market Equilibrium
    st.header("Market Equilibrium Returns")
    st.markdown("Returns implied by current market equilibrium (CAPM-based).")
    
    # Display market weights and caps
    pi_df = pd.DataFrame({
        'Market Weight': [f"{w:.2%}" for w in market_weights],
        'Implied Return': [f"{r:.2%}" for r in pi]
    }, index=tickers)
    
    if market_caps:
        # Add market caps to the display
        caps_list = []
        for ticker in tickers:
            if ticker in market_caps:
                cap_billions = market_caps[ticker] / 1e9
                caps_list.append(f"${cap_billions:,.1f}B")
            else:
                caps_list.append("N/A")
        pi_df['Market Cap'] = caps_list
    
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
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # User Views
    st.header("Your Market Views")
    
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
                    step=0.5,
                    key=f"view_abs_{ticker}_{idx}",
                    help=f"Current implied: {default_view:.1f}%"
                )
                views[idx] = view / 100
                
                confidence = st.slider(
                    "Confidence",
                    0, 100, 50,
                    key=f"conf_abs_{ticker}_{idx}",
                    label_visibility="collapsed",
                    help="0% = no confidence, 100% = complete confidence"
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
                outperformance = st.number_input(
                    f"Outperformance (%)", 
                    -50.0, 50.0, 0.0, 0.5, 
                    key=f"outperf_{i}",
                    help="Positive = A outperforms B, Negative = B outperforms A"
                )
                Q_vector[i] = outperformance / 100
            with col4:
                confidence = st.slider(
                    "Conf", 0, 100, 50, 
                    key=f"rel_conf_{i}", 
                    label_visibility="collapsed",
                    help="Confidence level (0-100%)"
                )
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
    if st.button("Calculate Black-Litterman Portfolio", type="primary", use_container_width=True):
        st.header("Black-Litterman Results")
        
        with st.spinner("Calculating optimal portfolio..."):
            # BL posterior
            bl_returns, bl_cov = black_litterman_model(pi, cov_matrix.values, P_matrix, Q_vector, omega_matrix, tau)
            
            # Optimize
            bl_weights, bl_return, bl_vol, bl_sharpe = optimize_portfolio_with_constraints(
                bl_returns, cov_matrix.values, risk_free_rate
            )
            
            # Benchmarks
            mkt_return = market_weights @ pi
            mkt_vol = np.sqrt(market_weights @ cov_matrix.values @ market_weights)
            mkt_sharpe = (mkt_return - risk_free_rate) / mkt_vol if mkt_vol > 0 else 0
            
            eq_weights = np.ones(n_assets) / n_assets
            eq_return = eq_weights @ pi
            eq_vol = np.sqrt(eq_weights @ cov_matrix.values @ eq_weights)
            eq_sharpe = (eq_return - risk_free_rate) / eq_vol if eq_vol > 0 else 0
        
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
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>Weight: %{percent}<br>Difference: %{customdata:.2%}',
                    customdata=allocation_df['Difference']
                )])
                fig.update_layout(
                    title="Black-Litterman Portfolio Allocation",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Portfolio Comparison")
                formatted_alloc = allocation_df.copy()
                formatted_alloc['BL Weight'] = formatted_alloc['BL Weight'].apply(lambda x: f"{x:.2%}")
                formatted_alloc['Market Weight'] = formatted_alloc['Market Weight'].apply(lambda x: f"{x:.2%}")
                formatted_alloc['Difference'] = formatted_alloc['Difference'].apply(lambda x: f"{x:+.2%}")
                st.dataframe(formatted_alloc.set_index('Ticker'), use_container_width=True)
                
                # Calculate portfolio metrics
                active_share = np.sum(np.abs(allocation_df['Difference'])) / 2
                turnover = np.sum(np.abs(allocation_df['BL Weight'] - allocation_df['Market Weight']))
                
                st.metric("Active Share", f"{active_share:.1%}")
                st.metric("Turnover vs Market", f"{turnover:.1%}")
                
                # Show top holdings
                st.subheader("Top 5 Holdings")
                top5 = allocation_df.head(5)
                for _, row in top5.iterrows():
                    st.write(f"{row['Ticker']}: {row['BL Weight']:.2%}")
        
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
                fig.update_layout(
                    title="Expected Returns Comparison", 
                    yaxis_title="Annual Return (%)", 
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with result_tab3:
            metrics_data = {
                'Metric': ['Expected Return', 'Expected Volatility', 'Sharpe Ratio'],
                'Black-Litterman': [f"{bl_return:.2%}", f"{bl_vol:.2%}", f"{bl_sharpe:.2f}"],
                'Market Portfolio': [f"{mkt_return:.2%}", f"{mkt_vol:.2%}", f"{mkt_sharpe:.2f}"],
                'Equal Weight': [f"{eq_return:.2%}", f"{eq_vol:.2%}", f"{eq_sharpe:.2f}"]
            }
            
            st.dataframe(pd.DataFrame(metrics_data).set_index('Metric'), use_container_width=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Beta", f"{bl_vol/mkt_vol:.2f}" if mkt_vol > 0 else "N/A")
            with col2:
                excess_return = bl_return - risk_free_rate
                st.metric("Excess Return", f"{excess_return:.2%}")
            with col3:
                information_ratio = (bl_return - mkt_return) / bl_vol if bl_vol > 0 else 0
                st.metric("Information Ratio", f"{information_ratio:.2f}")
        
        with result_tab4:
            frontier_returns, frontier_vols, frontier_sharpes = create_efficient_frontier(
                bl_returns, bl_cov, risk_free_rate
            )
            
            if frontier_returns:
                fig = go.Figure()
                
                # Efficient frontier
                fig.add_trace(go.Scatter(
                    x=frontier_vols, 
                    y=frontier_returns, 
                    mode='lines', 
                    name='Efficient Frontier',
                    line=dict(color='gray', width=2),
                    hovertemplate='Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
                
                # Capital Market Line
                max_sharpe_idx = np.argmax(frontier_sharpes)
                if frontier_sharpes[max_sharpe_idx] > 0:
                    cml_x = [0, frontier_vols[max_sharpe_idx] * 1.2]
                    cml_y = [risk_free_rate, risk_free_rate + frontier_sharpes[max_sharpe_idx] * cml_x[1]]
                    fig.add_trace(go.Scatter(
                        x=cml_x,
                        y=cml_y,
                        mode='lines',
                        name='Capital Market Line',
                        line=dict(color='green', dash='dash', width=1)
                    ))
                
                portfolios = [
                    ('BL Portfolio', bl_vol, bl_return, 'green', 'circle'),
                    ('Market', mkt_vol, mkt_return, 'blue', 'square'),
                    ('Equal Weight', eq_vol, eq_return, 'orange', 'diamond')
                ]
                
                for name, vol, ret, color, symbol in portfolios:
                    fig.add_trace(go.Scatter(
                        x=[vol], 
                        y=[ret], 
                        mode='markers+text', 
                        name=name,
                        marker=dict(size=15, color=color, symbol=symbol),
                        text=[name], 
                        textposition="top center",
                        hovertemplate=f'{name}<br>Vol: {vol:.2%}<br>Return: {ret:.2%}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Efficient Frontier", 
                    xaxis_title="Volatility", 
                    yaxis_title="Return",
                    hovermode='closest',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show maximum Sharpe portfolio
                if frontier_sharpes:
                    max_sharpe_idx = np.argmax(frontier_sharpes)
                    st.info(f"Maximum Sharpe Ratio on Frontier: {frontier_sharpes[max_sharpe_idx]:.2f} "
                           f"(Return: {frontier_returns[max_sharpe_idx]:.2%}, Vol: {frontier_vols[max_sharpe_idx]:.2%})")
            else:
                st.warning("Could not calculate efficient frontier with the given parameters.")

else:
    st.info("Select at least 2 stocks in the sidebar and click 'Fetch Data' to begin")
    
    with st.expander("Instructions", expanded=True):
        st.markdown("""
        ### How to use:
        
        1. **Select Stocks** - Choose from popular stocks or enter custom tickers
        2. **Set Date Range** (5+ years recommended for stable estimates)
        3. **Adjust Model Parameters**:
           - Risk Aversion (higher = more conservative)
           - Uncertainty Scale (lower = trust market equilibrium more)
           - Risk-Free Rate (used for Sharpe ratio calculation)
        4. **Click 'Fetch Data'**
        5. **Enter Your Views**:
           - Absolute views: Set expected returns for each stock
           - Relative views: Compare stocks against each other
        6. **Calculate Portfolio** - See your optimized Black-Litterman portfolio
        
        ### About Market Cap Weights:
        - The app tries to fetch actual market capitalization from Yahoo Finance
        - If unavailable, it uses price-based weights as a fallback
        - Market caps are used to calculate market equilibrium weights
        
        ### Tips:
        - Start with 5-10 stocks for faster processing
        - Use longer time periods (5+ years) for more stable covariance estimates
        - Higher confidence in views = smaller adjustments from market equilibrium
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Black-Litterman Portfolio Optimizer | Educational Tool | Not Financial Advice</p>
</div>
""", unsafe_allow_html=True)
