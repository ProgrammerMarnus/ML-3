"""
Backtest Dashboard for visualizing trading strategy results.

Provides comprehensive visualization of equity curves, drawdowns, 
trade distributions, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


class BacktestDashboard:
    """
    Comprehensive dashboard for visualizing backtest results.
    
    Generates professional-quality plots for:
    - Equity curves (cumulative returns)
    - Drawdown analysis
    - Trade distribution and frequency
    - Monthly returns heatmap
    - Position sizing evolution
    - Feature importance (if available)
    """
    
    def __init__(self, results: Dict[str, Any], style: str = 'seaborn-v0_8'):
        """
        Initialize dashboard with backtest results.
        
        Args:
            results: Dictionary containing backtest output including:
                - equity_curve: DataFrame or array with cumulative returns
                - trades: DataFrame with trade details
                - positions: DataFrame or array with position history
                - metrics: Dictionary of performance metrics
                - feature_importance: Optional DataFrame
                - returns: Array of strategy returns
                - indices: Array of indices
            style: Matplotlib style to use
        """
        self.results = results
        self.metrics = results.get('metrics', {})
        self.feature_importance = results.get('feature_importance', None)
        
        # Convert numpy arrays to DataFrames for consistency
        equity_curve_data = results.get('equity_curve', np.array([]))
        returns_data = results.get('returns', np.array([]))
        indices_data = results.get('indices', np.arange(len(equity_curve_data)))
        
        if isinstance(equity_curve_data, np.ndarray):
            self.equity_curve = pd.DataFrame({
                'equity': equity_curve_data,
                'return': returns_data if len(returns_data) > 0 else np.zeros_like(equity_curve_data)
            }, index=indices_data)
        else:
            self.equity_curve = equity_curve_data
        
        positions_data = results.get('positions', np.array([]))
        if isinstance(positions_data, np.ndarray):
            self.positions = pd.DataFrame({'position': positions_data}, index=indices_data)
        else:
            self.positions = positions_data
        
        self.trades = results.get('trades', pd.DataFrame())
        
        # Set plot style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn')
        
        # Configure default figure size and DPI
        self.fig_size = (14, 10)
        self.dpi = 100
    
    def plot_all(self, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
        """
        Generate all dashboard plots in a single figure.
        
        Args:
            save_path: Optional path to save the figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # Plot 1: Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_equity_curve(ax=ax1)
        
        # Plot 2: Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_drawdown(ax=ax2)
        
        # Plot 3: Monthly Returns Heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_monthly_returns(ax=ax3)
        
        # Plot 4: Trade Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self.plot_trade_distribution(ax=ax4)
        
        # Plot 5: Position Evolution
        ax5 = fig.add_subplot(gs[2, 1])
        self.plot_position_evolution(ax=ax5)
        
        # Add title
        fig.suptitle('Market Predictor ML - Backtest Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_equity_curve(self, ax: Optional[plt.Axes] = None, 
                         benchmark: Optional[pd.Series] = None) -> plt.Axes:
        """
        Plot cumulative equity curve.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            benchmark: Optional benchmark series for comparison
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        if self.equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity curve data available', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        # Calculate cumulative returns
        if 'equity' in self.equity_curve.columns:
            equity = self.equity_curve['equity']
        elif 'cumulative_return' in self.equity_curve.columns:
            equity = self.equity_curve['cumulative_return']
        else:
            equity = (1 + self.equity_curve['return']).cumprod()
        
        # Plot equity
        ax.plot(equity.index, equity.values, label='Strategy', 
               linewidth=2, color='#2E86AB')
        
        # Add benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values, label='Benchmark', 
                   linewidth=2, color='#A23B72', linestyle='--')
        
        # Format
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add metrics annotation
        if self.metrics:
            sharpe = self.metrics.get('sharpe_ratio', 0)
            total_ret = self.metrics.get('total_return', 0)
            max_dd = self.metrics.get('max_drawdown', 0)
            
            textstr = f'Sharpe: {sharpe:.2f}\nTotal Ret: {total_ret:.1%}\nMax DD: {max_dd:.1%}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        return ax
    
    def plot_drawdown(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot drawdown curve.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        if self.equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity curve data available', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        # Calculate drawdown
        if 'drawdown' in self.equity_curve.columns:
            drawdown = self.equity_curve['drawdown']
        else:
            equity = self.equity_curve.get('equity', 
                     (1 + self.equity_curve['return']).cumprod())
            running_max = equity.cummax()
            drawdown = (equity - running_max) / running_max
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       color='#F18F01', alpha=0.7, label='Drawdown')
        
        # Format
        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Annotate max drawdown
        if len(drawdown) > 0:
            max_dd = drawdown.min()
            max_dd_date = drawdown.idxmin()
            ax.annotate(f'Max DD: {max_dd:.1%}', 
                       xy=(max_dd_date, max_dd), 
                       xytext=(0.02, 0.05), textcoords='axes fraction',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                       fontsize=10)
        
        return ax
    
    def plot_monthly_returns(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot monthly returns heatmap.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        if self.equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity curve data available', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        # Calculate monthly returns
        returns = self.equity_curve.get('return', 
                  self.equity_curve['equity'].pct_change())
        returns = returns.dropna()
        
        # Resample to month-end requires a DatetimeIndex; guard when the
        # equity curve was built from a naked numpy array (integer index).
        if not isinstance(returns.index, pd.DatetimeIndex):
            ax.text(0.5, 0.5, 'Monthly heatmap requires a datetime index',
                    transform=ax.transAxes, ha='center')
            return ax
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })
        
        # Pivot for heatmap
        heatmap_data = monthly_df.pivot(index='year', columns='month', values='return')
        
        if heatmap_data.empty:
            ax.text(0.5, 0.5, 'Insufficient data for monthly heatmap', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        # Create heatmap
        months_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', 
                      vmin=-0.2, vmax=0.2)
        
        # Set ticks
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels([months_labels[m - 1] for m in heatmap_data.columns])
        ax.set_yticklabels(heatmap_data.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Monthly Return')
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not np.isnan(value):
                    color = 'white' if abs(value) > 0.1 else 'black'
                    ax.text(j, i, f'{value:.1%}', ha='center', va='center', 
                           color=color, fontsize=8)
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        return ax
    
    def plot_trade_distribution(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot distribution of trade returns.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        if self.trades.empty or 'return' not in self.trades.columns:
            ax.text(0.5, 0.5, 'No trade data available', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        trade_returns = self.trades['return'].dropna()
        
        if len(trade_returns) == 0:
            ax.text(0.5, 0.5, 'No trade returns to plot', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        # Histogram
        ax.hist(trade_returns, bins=30, color='#2E86AB', alpha=0.7, 
               edgecolor='black', label='Trade Returns')
        
        # Add mean and median lines
        mean_ret = trade_returns.mean()
        median_ret = trade_returns.median()
        ax.axvline(mean_ret, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_ret:.2%}')
        ax.axvline(median_ret, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_ret:.2%}')
        
        # Format
        ax.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        win_rate = (trade_returns > 0).mean()
        avg_win = trade_returns[trade_returns > 0].mean()
        avg_loss = trade_returns[trade_returns < 0].mean()
        
        stats_text = f'Win Rate: {win_rate:.1%}\nAvg Win: {avg_win:.2%}\nAvg Loss: {avg_loss:.2%}'
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    def plot_position_evolution(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot position sizing over time.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        if self.positions.empty:
            ax.text(0.5, 0.5, 'No position data available', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        # Plot position size
        if 'position' in self.positions.columns:
            position = self.positions['position']
        elif 'size' in self.positions.columns:
            position = self.positions['size']
        else:
            ax.text(0.5, 0.5, 'No position column found', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        ax.plot(position.index, position.values, linewidth=1.5, color='#A23B72')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Fill areas for long/short
        ax.fill_between(position.index, position.values, 0, 
                       where=(position > 0), color='green', alpha=0.3, label='Long')
        ax.fill_between(position.index, position.values, 0, 
                       where=(position < 0), color='red', alpha=0.3, label='Short')
        
        # Format
        ax.set_title('Position Sizing Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Position Size')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        return ax
    
    def plot_feature_importance(self, top_n: int = 20, 
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot feature importance from model.
        
        Args:
            top_n: Number of top features to display
            ax: Matplotlib axes (creates new if None)
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        if self.feature_importance is None:
            ax.text(0.5, 0.5, 'No feature importance data available', 
                   transform=ax.transAxes, ha='center')
            return ax
        
        # Sort and select top features
        fi = self.feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        # Plot horizontal bar chart
        ax.barh(fi['feature'], fi['importance'], color='#2E86AB')
        ax.invert_yaxis()
        
        # Format
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.grid(True, alpha=0.3, axis='x')
        
        return ax
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a text summary report of backtest results.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("MARKET PREDICTOR ML - BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                report.append(f"{key.replace('_', ' ').title()}: {value:.4f}" if 'ratio' in key or 'sharpe' in key else f"{key.replace('_', ' ').title()}: {value:.2%}")
            else:
                report.append(f"{key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Trade Statistics
        if not self.trades.empty:
            report.append("TRADE STATISTICS")
            report.append("-" * 40)
            report.append(f"Total Trades: {len(self.trades)}")
            if 'return' in self.trades.columns:
                wins = (self.trades['return'] > 0).sum()
                losses = (self.trades['return'] < 0).sum()
                report.append(f"Winning Trades: {wins} ({wins/len(self.trades):.1%})")
                report.append(f"Losing Trades: {losses} ({losses/len(self.trades):.1%})")
                report.append(f"Avg Win: {self.trades[self.trades['return'] > 0]['return'].mean():.2%}")
                report.append(f"Avg Loss: {self.trades[self.trades['return'] < 0]['return'].mean():.2%}")
            report.append("")
        
        # Data Period
        if not self.equity_curve.empty:
            report.append("BACKTEST PERIOD")
            report.append("-" * 40)
            try:
                report.append(f"Start Date: {self.equity_curve.index[0].strftime('%Y-%m-%d')}")
                report.append(f"End Date: {self.equity_curve.index[-1].strftime('%Y-%m-%d')}")
            except:
                report.append(f"Start Index: {self.equity_curve.index[0]}")
                report.append(f"End Index: {self.equity_curve.index[-1]}")
            report.append(f"Trading Days: {len(self.equity_curve)}")
            report.append("")
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            print(f"Report saved to {save_path}")
        
        return report_str
