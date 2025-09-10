"""
Crypto Trading Intelligence Dashboard
=========================================================

Interactive Streamlit dashboard using Hyperliquid trading data
and Fear & Greed Index analysis.

Author: dshail
Date: September 2025
Version: 1.0.0 - REAL DATA INTEGRATION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Real Crypto Trading Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stDeployButton {display:none;}
    .stDecoration {display:none;}
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .signal-bullish {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .signal-bearish {
        background: linear-gradient(90deg, #ff4444 0%, #cc3333 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .signal-neutral {
        background: linear-gradient(90deg, #ffaa00 0%, #cc8800 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .real-data-badge {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_real_data():
    """Load REAL analysis results from CSV files."""
    try:
        # Load real analysis results
        real_data = pd.read_csv('hyperliquid_analysis_results.csv')
        trader_profiles = pd.read_csv('hyperliquid_trader_profiles.csv')
        
        # Calculate real performance metrics
        total_trades = len(real_data)
        total_pnl = real_data['closedPnL'].sum()
        win_rate = (real_data['closedPnL'] > 0).mean() * 100
        unique_traders = real_data['account'].nunique()
        unique_symbols = real_data['symbol'].nunique()
        
        # Real sentiment performance
        sentiment_performance = {}
        for sentiment in ['Fear', 'Neutral', 'Greed']:
            sentiment_data = real_data[real_data['Classification'] == sentiment]
            if len(sentiment_data) > 0:
                sentiment_performance[sentiment] = {
                    'avg_pnl': sentiment_data['closedPnL'].mean(),
                    'win_rate': (sentiment_data['closedPnL'] > 0).mean() * 100,
                    'trade_count': len(sentiment_data),
                    'total_pnl': sentiment_data['closedPnL'].sum()
                }
        
        # Real trader type performance
        trader_type_performance = {}
        for trader_type in trader_profiles['trader_type_name'].unique():
            type_traders = trader_profiles[trader_profiles['trader_type_name'] == trader_type]
            trader_type_performance[trader_type] = {
                'count': len(type_traders),
                'avg_total_pnl': type_traders['total_pnl'].mean(),
                'avg_win_rate': type_traders['profit_consistency'].mean() * 100,
                'avg_trades': type_traders['total_trades'].mean()
            }
            
            # Get sentiment-specific performance for this trader type
            type_accounts = type_traders.index.tolist()
            type_data = real_data[real_data['account'].isin(type_accounts)]
            
            for sentiment in ['Fear', 'Neutral', 'Greed']:
                sentiment_type_data = type_data[type_data['Classification'] == sentiment]
                if len(sentiment_type_data) > 0:
                    trader_type_performance[trader_type][f'{sentiment.lower()}_pnl'] = sentiment_type_data['closedPnL'].mean()
                else:
                    trader_type_performance[trader_type][f'{sentiment.lower()}_pnl'] = 0
        
        # Create Fear & Greed time series from real data
        fear_greed_df = real_data[['Date', 'FearGreedIndex', 'Classification']].drop_duplicates().sort_values('Date')
        
        return {
            'real_data': real_data,
            'trader_profiles': trader_profiles,
            'fear_greed_df': fear_greed_df,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'unique_traders': unique_traders,
            'unique_symbols': unique_symbols,
            'sentiment_performance': sentiment_performance,
            'trader_type_performance': trader_type_performance,
            'symbols': sorted(real_data['symbol'].unique()),
            'is_real_data': True
        }
        
    except FileNotFoundError:
        st.error("⚠️ Real data files not found! Please run the analysis first to generate CSV files.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading real data: {str(e)}")
        return None

def generate_real_trading_signal(fear_greed_index, hour, trader_type, symbol, real_results):
    """Generate trading signals based on REAL data analysis."""
    signals = []
    confidence = 50
    signal_type = "neutral"
    
    if real_results is None:
        return {
            'signals': ["⚠️ Real data not available"],
            'confidence': 0,
            'signal_type': 'neutral'
        }
    
    # Get real sentiment performance data
    sentiment_perf = real_results['sentiment_performance']
    
    # Real sentiment-based signals
    if fear_greed_index <= 30:  # Fear
        if 'Fear' in sentiment_perf:
            fear_avg = sentiment_perf['Fear']['avg_pnl']
            fear_win_rate = sentiment_perf['Fear']['win_rate']
            if fear_avg > 0:
                signals.append(f"🟢 BUY SIGNAL - Fear periods: ${fear_avg:.2f} avg PnL, {fear_win_rate:.1f}% win rate")
                signal_type = "bullish"
                confidence += 25
            else:
                signals.append(f"🔴 CAUTION - Fear periods underperform: ${fear_avg:.2f} avg PnL")
                signal_type = "bearish"
                confidence += 15
                
    elif fear_greed_index >= 70:  # Greed
        if 'Greed' in sentiment_perf:
            greed_avg = sentiment_perf['Greed']['avg_pnl']
            greed_win_rate = sentiment_perf['Greed']['win_rate']
            if greed_avg > sentiment_perf.get('Fear', {}).get('avg_pnl', 0):
                signals.append(f"🟢 MOMENTUM - Greed periods: ${greed_avg:.2f} avg PnL, {greed_win_rate:.1f}% win rate")
                signal_type = "bullish"
                confidence += 20
            else:
                signals.append(f"⚠️ GREED WARNING - Lower performance: ${greed_avg:.2f} avg PnL")
                signal_type = "bearish"
                confidence += 10
    
    # Real trader type specific signals
    trader_perf = real_results['trader_type_performance'].get(trader_type, {})
    if trader_perf:
        win_rate = trader_perf.get('avg_win_rate', 0)
        if win_rate > 60:
            signals.append(f"🌟 TRADER ADVANTAGE - Your type: {win_rate:.1f}% win rate")
            confidence += 15
        elif win_rate < 40:
            signals.append(f"🚨 TRADER WARNING - Your type: {win_rate:.1f}% win rate")
            confidence -= 15
    
    # Real symbol analysis
    symbol_data = real_results['real_data'][real_results['real_data']['symbol'] == symbol]
    if len(symbol_data) > 0:
        symbol_pnl = symbol_data['closedPnL'].mean()
        symbol_trades = len(symbol_data)
        signals.append(f"📊 {symbol} Analysis: ${symbol_pnl:.2f} avg PnL ({symbol_trades} real trades)")
    
    return {
        'signals': signals,
        'confidence': min(100, max(0, confidence)),
        'signal_type': signal_type
    }

def create_real_fear_greed_chart(fear_greed_df):
    """Create Fear & Greed chart from REAL data."""
    fig = go.Figure()
    
    # Color mapping
    colors = fear_greed_df['Classification'].map({
        'Fear': '#ff4444',
        'Neutral': '#ffaa00', 
        'Greed': '#00ff88'
    })
    
    fig.add_trace(go.Scatter(
        x=fear_greed_df['Date'],
        y=fear_greed_df['FearGreedIndex'],
        mode='lines+markers',
        name='Real Fear & Greed Index',
        line=dict(width=3),
        marker=dict(
            size=6,
            color=colors,
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>%{x}</b><br>F&G Index: %{y:.1f}<extra></extra>'
    ))
    
    # Add sentiment zones
    fig.add_hline(y=30, line_dash="dash", line_color="red", 
                  annotation_text="Fear Threshold", annotation_position="right")
    fig.add_hline(y=70, line_dash="dash", line_color="green",
                  annotation_text="Greed Threshold", annotation_position="right")
    
    fig.update_layout(
        title="Real Bitcoin Fear & Greed Index (From Your Data)",
        xaxis_title="Date",
        yaxis_title="Fear & Greed Index",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_real_performance_heatmap(trader_type_performance):
    """Create performance heatmap from REAL data."""
    traders = list(trader_type_performance.keys())
    sentiments = ['Fear', 'Neutral', 'Greed']
    
    # Prepare real data matrix
    performance_matrix = []
    for trader in traders:
        row = [
            trader_type_performance[trader].get('fear_pnl', 0),
            trader_type_performance[trader].get('neutral_pnl', 0),
            trader_type_performance[trader].get('greed_pnl', 0)
        ]
        performance_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=performance_matrix,
        x=sentiments,
        y=traders,
        colorscale='RdYlGn',
        zmid=0,
        hovertemplate='<b>%{y}</b><br>%{x}: $%{z:.2f} (Real Data)<extra></extra>',
        colorbar=dict(title="Real Avg PnL ($)")
    ))
    
    fig.update_layout(
        title="Real Trader Performance by Market Sentiment",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_real_win_rate_chart(trader_type_performance):
    """Create win rate chart from REAL data."""
    traders = list(trader_type_performance.keys())
    win_rates = [trader_type_performance[trader]['avg_win_rate'] for trader in traders]
    
    # Dynamic colors based on performance
    colors = []
    for rate in win_rates:
        if rate >= 70:
            colors.append('#00ff88')  # Green for high performers
        elif rate >= 50:
            colors.append('#ffaa00')  # Yellow for average
        else:
            colors.append('#ff4444')  # Red for poor performers
    
    fig = go.Figure(data=[
        go.Bar(
            x=traders,
            y=win_rates,
            marker_color=colors,
            text=[f"{rate:.1f}%" for rate in win_rates],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Real Win Rate: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Real Win Rates by Trader Type (From Your Data)",
        xaxis_title="Trader Type",
        yaxis_title="Win Rate (%)",
        template="plotly_white",
        height=400
    )
    
    return fig

def main():
    """Main dashboard application with REAL data."""
    
    # Header with REAL DATA badge
    st.title("🚀 Crypto Trading Intelligence Dashboard")
    st.markdown("""
    <div class="real-data-badge">
        ✅ REAL DATA - Hyperliquid Analysis
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Live Analysis from Your Actual Trading History")
    
    # Load REAL data
    with st.spinner("Loading real trading data..."):
        real_results = load_real_data()
    
    if real_results is None:
        st.error("Unable to load real data. Please run the analysis first.")
        st.stop()
    
    # Display real data statistics
    st.success(f"""
    📊 **Real Data Loaded Successfully!**
    - **{real_results['total_trades']:,} real trades** analyzed
    - **{real_results['unique_traders']} real traders** from Hyperliquid
    - **{real_results['unique_symbols']} different symbols** traded
    - **${real_results['total_pnl']:,.2f} total PnL** analyzed
    - **{real_results['win_rate']:.1f}% overall win rate**
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Real Trading Parameters")
    
    # Current conditions
    current_fg = st.sidebar.slider("Current Fear & Greed Index", 0, 100, 45)
    current_hour = st.sidebar.slider("Current Hour (UTC)", 0, 23, 14)
    trader_type = st.sidebar.selectbox("Your Trader Type", 
                                     list(real_results['trader_type_performance'].keys()))
    symbol = st.sidebar.selectbox("Trading Symbol", real_results['symbols'][:20])  # Top 20 symbols
    
    # Generate REAL signal
    signal_data = generate_real_trading_signal(current_fg, current_hour, trader_type, symbol, real_results)
    
    # Main dashboard layout
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("📊 Real Market Sentiment Data")
        st.plotly_chart(create_real_fear_greed_chart(real_results['fear_greed_df']), use_container_width=True)
        
    with col2:
        st.subheader("🎯 Real Trader Performance")
        st.plotly_chart(create_real_performance_heatmap(real_results['trader_type_performance']), use_container_width=True)
        
    with col3:
        st.subheader("🚨 Live Trading Signal")
        
        # Signal display
        signal_class = f"signal-{signal_data['signal_type']}"
        confidence = signal_data['confidence']
        
        if signal_data['signal_type'] == 'bullish':
            signal_emoji = "🟢"
            signal_text = "BULLISH"
        elif signal_data['signal_type'] == 'bearish':
            signal_emoji = "🔴"
            signal_text = "BEARISH"
        else:
            signal_emoji = "🟡"
            signal_text = "NEUTRAL"
            
        st.markdown(f"""
        <div class="{signal_class}">
            {signal_emoji} {signal_text}<br>
            Confidence: {confidence}%<br>
            <small>Based on Real Data</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal details
        st.subheader("📝 Real Signal Details")
        for signal in signal_data['signals']:
            st.write(signal)
    
    # REAL Performance metrics row
    st.markdown("---")
    st.subheader("📈 Real Performance Metrics from Your Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get real sentiment performance
    sentiment_perf = real_results['sentiment_performance']
    best_sentiment = max(sentiment_perf.keys(), key=lambda x: sentiment_perf[x]['avg_pnl'])
    best_sentiment_pnl = sentiment_perf[best_sentiment]['avg_pnl']
    
    with col1:
        neutral_pnl = sentiment_perf.get('Neutral', {}).get('avg_pnl', 0)
        greed_pnl = sentiment_perf.get('Greed', {}).get('avg_pnl', 0)
        delta_value = neutral_pnl - greed_pnl
        st.metric(
            label="Neutral Period Real Avg PnL",
            value=f"${neutral_pnl:.2f}",
            delta=f"${delta_value:.2f} vs Greed",
            delta_color="normal"
        )
        
    with col2:
        # Find best trader type win rate
        best_trader_type = max(real_results['trader_type_performance'].keys(), 
                              key=lambda x: real_results['trader_type_performance'][x]['avg_win_rate'])
        best_win_rate = real_results['trader_type_performance'][best_trader_type]['avg_win_rate']
        overall_win_rate = real_results['win_rate']
        st.metric(
            label=f"Real {best_trader_type} Win Rate", 
            value=f"{best_win_rate:.1f}%",
            delta=f"+{best_win_rate - overall_win_rate:.1f}pp vs Avg",
            delta_color="normal"
        )
        
    with col3:
        st.metric(
            label="Real Data Coverage",
            value=f"{len(real_results['real_data']):,} trades",
            delta=f"{real_results['unique_symbols']} symbols",
            delta_color="normal"
        )
        
    with col4:
        current_sentiment = "Fear" if current_fg <= 30 else "Greed" if current_fg >= 70 else "Neutral"
        expected_pnl = sentiment_perf.get(current_sentiment, {}).get('avg_pnl', 0)
        st.metric(
            label="Current Regime Expected PnL",
            value=f"${expected_pnl:.2f}",
            delta=f"F&G: {current_fg} ({current_sentiment})"
        )
    
    # Detailed REAL analysis section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Real Trader Type Analysis")
        st.plotly_chart(create_real_win_rate_chart(real_results['trader_type_performance']), use_container_width=True)
        
        # Real trader type breakdown
        st.markdown("**Real Trader Distribution:**")
        for trader_type_name, data in real_results['trader_type_performance'].items():
            st.write(f"• **{trader_type_name}**: {data['count']} traders, {data['avg_win_rate']:.1f}% win rate, ${data['avg_total_pnl']:,.2f} avg PnL")
    
    with col2:
        st.subheader("📊 Real Strategy Recommendations")
        
        # Get actual best performing sentiment
        if best_sentiment == "Fear":
            st.success(f"""
            **🔥 REAL DATA INSIGHT: FEAR STRATEGY**
            - Your data shows Fear periods perform best: ${best_sentiment_pnl:.2f} avg PnL
            - Win rate during Fear: {sentiment_perf['Fear']['win_rate']:.1f}%
            - Total Fear trades: {sentiment_perf['Fear']['trade_count']:,}
            - **Recommendation**: Increase positions during market fear
            """)
        elif best_sentiment == "Greed":
            st.success(f"""
            **🚀 REAL DATA INSIGHT: GREED MOMENTUM**
            - Your data shows Greed periods perform best: ${best_sentiment_pnl:.2f} avg PnL
            - Win rate during Greed: {sentiment_perf['Greed']['win_rate']:.1f}%
            - Total Greed trades: {sentiment_perf['Greed']['trade_count']:,}
            - **Recommendation**: Follow momentum during greedy markets
            """)
        else:
            st.info(f"""
            **⚖️ REAL DATA INSIGHT: NEUTRAL OUTPERFORMS**
            - Your data shows Neutral periods perform best: ${best_sentiment_pnl:.2f} avg PnL
            - Win rate during Neutral: {sentiment_perf['Neutral']['win_rate']:.1f}%
            - Total Neutral trades: {sentiment_perf['Neutral']['trade_count']:,}
            - **Recommendation**: Focus on stable market conditions
            """)
    
    # Real advanced analytics
    st.markdown("---")
    st.subheader("🔬 Real Data Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Real Sentiment Analysis", "🏷️ Real Trader Types", "💰 Real PnL Distribution"])
    
    with tab1:
        st.markdown("**Real Sentiment Performance Analysis:**")
        for sentiment, perf in sentiment_perf.items():
            st.markdown(f"""
            **{sentiment} Periods (Real Data):**
            - Average PnL: ${perf['avg_pnl']:.2f}
            - Win Rate: {perf['win_rate']:.1f}%
            - Total Trades: {perf['trade_count']:,}
            - Total PnL: ${perf['total_pnl']:,.2f}
            """)
        
    with tab2:
        st.markdown("**Real Trader Type Performance:**")
        for trader_type, perf in real_results['trader_type_performance'].items():
            st.markdown(f"""
            **{trader_type} ({perf['count']} real traders):**
            - Average Total PnL: ${perf['avg_total_pnl']:,.2f}
            - Average Win Rate: {perf['avg_win_rate']:.1f}%
            - Average Trades per Trader: {perf['avg_trades']:.0f}
            """)
        
    with tab3:
        # Create PnL distribution chart
        fig_pnl = px.histogram(
            real_results['real_data'], 
            x='closedPnL', 
            nbins=50,
            title="Real PnL Distribution from Your Trading Data",
            labels={'closedPnL': 'PnL ($)', 'count': 'Number of Trades'}
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # PnL statistics
        pnl_stats = real_results['real_data']['closedPnL'].describe()
        st.markdown(f"""
        **Real PnL Statistics:**
        - Mean: ${pnl_stats['mean']:.2f}
        - Median: ${pnl_stats['50%']:.2f}  
        - Std Dev: ${pnl_stats['std']:.2f}
        - Min: ${pnl_stats['min']:.2f}
        - Max: ${pnl_stats['max']:.2f}
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Real Data Crypto Trading Intelligence Dashboard v2.0.0</p>
        <p>Built with YOUR actual Hyperliquid trading data and Fear & Greed Index</p>
        <p><em>Based on real trading performance. Past results inform but don't guarantee future performance.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()