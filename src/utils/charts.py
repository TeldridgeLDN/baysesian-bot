import plotly.graph_objects as go
from typing import List, Dict, Any

def plot_performance(results: List[Dict[str, Any]]):
    """Plot trading performance."""
    if not results:
        return
        
    timestamps = [r['timestamp'] for r in results]
    prices = [r['price'] for r in results]
    portfolio_values = [r['portfolio_value'] for r in results]
    signals = [r['signal'] for r in results]
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(x=timestamps, y=prices, mode='lines', name='Price'))
    
    # Portfolio value line
    fig.add_trace(go.Scatter(x=timestamps, y=portfolio_values, mode='lines', name='Portfolio Value', yaxis='y2'))
    
    # Buy/Sell markers
    buy_signals = [r for r in results if r['signal'] == 'LONG']
    sell_signals = [r for r in results if r['signal'] == 'SHORT']
    
    fig.add_trace(go.Scatter(
        x=[s['timestamp'] for s in buy_signals],
        y=[s['price'] for s in buy_signals],
        mode='markers', name='Buy Signal', marker_symbol='triangle-up', marker_color='green', marker_size=10
    ))
    
    fig.add_trace(go.Scatter(
        x=[s['timestamp'] for s in sell_signals],
        y=[s['price'] for s in sell_signals],
        mode='markers', name='Sell Signal', marker_symbol='triangle-down', marker_color='red', marker_size=10
    ))
    
    fig.update_layout(
        title_text="Trading Performance",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        yaxis2=dict(
            title="Portfolio Value",
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0, y=1, traceorder='normal')
    )
    
    fig.show() 