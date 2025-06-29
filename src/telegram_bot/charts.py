import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

class TradingVisualizer:
    @staticmethod
    def create_performance_chart(trading_data: List[Dict[str, Any]]) -> go.Figure:
        """Create a comprehensive performance chart showing price, signals, and portfolio value."""
        # Convert trading data to DataFrame
        df = pd.DataFrame(trading_data)
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                          shared_xaxes=True,
                          vertical_spacing=0.03,
                          subplot_titles=('Price and Signals', 'Portfolio Value'))
        
        # Add price line
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['price'],
                      mode='lines',
                      name='Price',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add signals as markers
        for signal in ['LONG', 'SHORT']:
            signal_points = df[df['signal'] == signal]
            fig.add_trace(
                go.Scatter(x=signal_points['timestamp'],
                          y=signal_points['price'],
                          mode='markers',
                          name=signal,
                          marker=dict(
                              symbol='triangle-up' if signal == 'LONG' else 'triangle-down',
                              color='green' if signal == 'LONG' else 'red',
                              size=10
                          )),
                row=1, col=1
            )
        
        # Add portfolio value
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['portfolio_value'],
                      mode='lines',
                      name='Portfolio Value',
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Trading Performance Analysis",
            xaxis_title="Time",
            yaxis_title="Price",
            yaxis2_title="Portfolio Value"
        )
        
        return fig
    
    @staticmethod
    def create_prediction_accuracy_chart(trading_data: List[Dict[str, Any]]) -> go.Figure:
        """Create a chart showing prediction accuracy over time."""
        df = pd.DataFrame(trading_data)
        
        fig = go.Figure()
        
        # Add actual price
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['actual_price'],
                      mode='lines',
                      name='Actual Price',
                      line=dict(color='blue'))
        )
        
        # Add predicted price
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['predicted_price'],
                      mode='lines',
                      name='Predicted Price',
                      line=dict(color='red', dash='dash'))
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Price Prediction Accuracy",
            xaxis_title="Time",
            yaxis_title="Price"
        )
        
        return fig
    
    @staticmethod
    def save_chart(fig: go.Figure, filename: str) -> str:
        """Save chart to a file and return the filename."""
        fig.write_image(filename)
        return filename 