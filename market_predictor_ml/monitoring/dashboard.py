"""
Dashboard Module

Provides simple dashboard creation for visualizing metrics and system status.
Generates HTML reports with charts and key performance indicators.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class DashboardPanel:
    """
    Represents a panel/widget in the dashboard.
    
    Attributes:
        title: Panel title
        panel_type: Type of panel (metric, chart, table, alert)
        data: Data to display
        config: Additional configuration for rendering
    """
    title: str
    panel_type: str  # metric, chart, table, alert, text
    data: Any
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert panel to dictionary for serialization."""
        return {
            "title": self.title,
            "type": self.panel_type,
            "data": self._serialize_data(),
            "config": self.config,
        }
    
    def _serialize_data(self) -> Any:
        """Serialize data for JSON export."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.to_dict("records")
        elif isinstance(self.data, pd.Series):
            return self.data.to_dict()
        elif isinstance(self.data, (datetime,)):
            return self.data.isoformat()
        elif hasattr(self.data, "to_dict"):
            return self.data.to_dict()
        else:
            try:
                json.dumps(self.data)
                return self.data
            except (TypeError, ValueError):
                return str(self.data)


class Dashboard:
    """
    Creates and manages dashboards for monitoring trading system performance.
    
    Features:
    - Multiple panel types (metrics, charts, tables, alerts)
    - HTML export with embedded charts
    - JSON export for API consumption
    - Automatic refresh timestamps
    - Customizable layouts
    """
    
    def __init__(self, title: str = "Trading System Dashboard"):
        """
        Initialize dashboard.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.panels: List[DashboardPanel] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_metric_panel(
        self,
        title: str,
        value: float,
        previous_value: Optional[float] = None,
        suffix: str = "",
        precision: int = 2,
    ) -> "Dashboard":
        """
        Add a metric panel showing a single value with optional change indicator.
        
        Args:
            title: Metric title
            value: Current value
            previous_value: Previous value for change calculation
            suffix: Unit suffix (e.g., "%", "$")
            precision: Decimal precision
        
        Returns:
            Self for method chaining
        """
        change = None
        change_pct = None
        
        if previous_value is not None and previous_value != 0:
            change = value - previous_value
            change_pct = (change / abs(previous_value)) * 100
        
        panel = DashboardPanel(
            title=title,
            panel_type="metric",
            data={
                "value": round(value, precision),
                "previous": round(previous_value, precision) if previous_value is not None else None,
                "change": round(change, precision) if change is not None else None,
                "change_pct": round(change_pct, 2) if change_pct is not None else None,
                "suffix": suffix,
            },
            config={"precision": precision},
        )
        
        self.panels.append(panel)
        self.updated_at = datetime.now()
        return self
    
    def add_chart_panel(
        self,
        title: str,
        data: pd.DataFrame,
        chart_type: str = "line",
        x_column: Optional[str] = None,
        y_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "Dashboard":
        """
        Add a chart panel.
        
        Args:
            title: Chart title
            data: DataFrame with chart data
            chart_type: Type of chart (line, bar, scatter, area)
            x_column: Column for x-axis
            y_columns: Columns for y-axis series
            **kwargs: Additional chart configuration
        
        Returns:
            Self for method chaining
        """
        panel = DashboardPanel(
            title=title,
            panel_type="chart",
            data=data,
            config={
                "chart_type": chart_type,
                "x_column": x_column,
                "y_columns": y_columns,
                **kwargs,
            },
        )
        
        self.panels.append(panel)
        self.updated_at = datetime.now()
        return self
    
    def add_table_panel(
        self,
        title: str,
        data: pd.DataFrame,
        max_rows: int = 10,
        **kwargs,
    ) -> "Dashboard":
        """
        Add a table panel.
        
        Args:
            title: Table title
            data: DataFrame with table data
            max_rows: Maximum rows to display
            **kwargs: Additional table configuration
        
        Returns:
            Self for method chaining
        """
        panel = DashboardPanel(
            title=title,
            panel_type="table",
            data=data.head(max_rows),
            config={"max_rows": max_rows, **kwargs},
        )
        
        self.panels.append(panel)
        self.updated_at = datetime.now()
        return self
    
    def add_alert_panel(
        self,
        title: str,
        alerts: List[Dict[str, Any]],
        **kwargs,
    ) -> "Dashboard":
        """
        Add an alert panel showing active alerts.
        
        Args:
            title: Panel title
            alerts: List of alert dictionaries
            **kwargs: Additional configuration
        
        Returns:
            Self for method chaining
        """
        panel = DashboardPanel(
            title=title,
            panel_type="alert",
            data=alerts,
            config={"show_resolved": False, **kwargs},
        )
        
        self.panels.append(panel)
        self.updated_at = datetime.now()
        return self
    
    def add_text_panel(
        self,
        title: str,
        text: str,
        **kwargs,
    ) -> "Dashboard":
        """
        Add a text panel for descriptions or annotations.
        
        Args:
            title: Panel title
            text: Text content (supports Markdown)
            **kwargs: Additional configuration
        
        Returns:
            Self for method chaining
        """
        panel = DashboardPanel(
            title=title,
            panel_type="text",
            data=text,
            config=kwargs,
        )
        
        self.panels.append(panel)
        self.updated_at = datetime.now()
        return self
    
    def set_metadata(self, **kwargs) -> "Dashboard":
        """Set dashboard metadata."""
        self.metadata.update(kwargs)
        self.updated_at = datetime.now()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to dictionary."""
        return {
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "panels": [panel.to_dict() for panel in self.panels],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert dashboard to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def export_json(self, filepath: str) -> None:
        """Export dashboard to JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())
    
    def to_html(self, template: Optional[str] = None) -> str:
        """
        Generate HTML report for the dashboard.
        
        Args:
            template: Optional custom HTML template
        
        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .dashboard-title {{
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }}
        .dashboard-meta {{
            font-size: 14px;
            opacity: 0.9;
            margin-top: 8px;
        }}
        .panels-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .panel {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .panel-title {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin: 0 0 12px 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }}
        .metric-change {{
            font-size: 14px;
            margin-left: 8px;
        }}
        .metric-change.positive {{
            color: #10b981;
        }}
        .metric-change.negative {{
            color: #ef4444;
        }}
        .chart-container {{
            height: 300px;
            position: relative;
        }}
        .table-container {{
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #e5e5e5;
        }}
        th {{
            background-color: #f9fafb;
            font-weight: 600;
        }}
        .alert-item {{
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 8px;
            border-left: 4px solid;
        }}
        .alert-info {{
            background-color: #dbeafe;
            border-color: #3b82f6;
        }}
        .alert-warning {{
            background-color: #fef3c7;
            border-color: #f59e0b;
        }}
        .alert-error {{
            background-color: #fee2e2;
            border-color: #ef4444;
        }}
        .alert-critical {{
            background-color: #fce7f3;
            border-color: #ec4899;
        }}
        .text-content {{
            line-height: 1.6;
            color: #4b5563;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1 class="dashboard-title">{self.title}</h1>
        <div class="dashboard-meta">
            Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')} | 
            Updated: {self.updated_at.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <div class="panels-grid">
"""
        
        for panel in self.panels:
            html += self._render_panel(panel)
        
        html += """
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Initialize charts
        document.querySelectorAll('.chart-panel').forEach(function(panelEl) {
            const canvas = panelEl.querySelector('canvas');
            if (!canvas) return;
            
            const config = JSON.parse(canvas.getAttribute('data-config'));
            const ctx = canvas.getContext('2d');
            
            new Chart(ctx, {
                type: config.type,
                data: config.data,
                options: config.options || {}
            });
        });
    </script>
</body>
</html>
"""
        
        return html
    
    def _render_panel(self, panel: DashboardPanel) -> str:
        """Render a single panel to HTML."""
        html = f'<div class="panel {panel.panel_type}-panel">\n'
        html += f'<h3 class="panel-title">{panel.title}</h3>\n'
        
        if panel.panel_type == "metric":
            data = panel.data
            value = data.get("value", 0)
            suffix = data.get("suffix", "")
            html += f'<div class="metric-value">{value}{suffix}</div>\n'
            
            if data.get("change") is not None:
                change = data["change"]
                change_pct = data.get("change_pct", 0)
                sign = "+" if change > 0 else ""
                css_class = "positive" if change > 0 else "negative"
                html += f'<span class="metric-change {css_class}">({sign}{change}{suffix}, {sign}{change_pct}%)</span>\n'
        
        elif panel.panel_type == "chart":
            html += '<div class="chart-container">\n'
            html += '<canvas data-config=\'{}\'></canvas>\n'.format(
                self._prepare_chart_config(panel)
            )
            html += '</div>\n'
        
        elif panel.panel_type == "table":
            html += '<div class="table-container">\n'
            html += panel.data.to_html(index=False, classes='') if isinstance(panel.data, pd.DataFrame) else str(panel.data)
            html += '</div>\n'
        
        elif panel.panel_type == "alert":
            alerts = panel.data if isinstance(panel.data, list) else []
            for alert in alerts:
                severity = alert.get("severity", "info")
                message = alert.get("message", "Unknown alert")
                html += f'<div class="alert-item alert-{severity}"><strong>{alert.get("name", "Alert")}</strong>: {message}</div>\n'
        
        elif panel.panel_type == "text":
            html += f'<div class="text-content">{panel.data}</div>\n'
        
        html += '</div>\n'
        return html
    
    def _prepare_chart_config(self, panel: DashboardPanel) -> Dict[str, Any]:
        """Prepare Chart.js configuration for a chart panel."""
        config = panel.config
        data = panel.data
        
        if not isinstance(data, pd.DataFrame):
            return {"type": "line", "data": {"labels": [], "datasets": []}}
        
        chart_type = config.get("chart_type", "line")
        x_column = config.get("x_column", data.columns[0] if len(data.columns) > 0 else None)
        y_columns = config.get("y_columns", list(data.columns[1:]) if len(data.columns) > 1 else [])
        
        labels = data[x_column].tolist() if x_column else list(range(len(data)))
        
        datasets = []
        colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b', '#ef4444']
        
        for i, y_col in enumerate(y_columns):
            datasets.append({
                "label": y_col,
                "data": data[y_col].tolist(),
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)] + '40',
                "fill": chart_type == "area",
                "tension": 0.4 if chart_type == "line" else 0,
            })
        
        return {
            "type": "area" if chart_type == "area" else chart_type,
            "data": {
                "labels": labels,
                "datasets": datasets,
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top",
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": config.get("begin_at_zero", False),
                    }
                },
            },
        }
    
    def export_html(self, filepath: str) -> None:
        """Export dashboard to HTML file."""
        with open(filepath, "w") as f:
            f.write(self.to_html())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dashboard":
        """Create dashboard from dictionary."""
        dashboard = cls(title=data.get("title", "Dashboard"))
        
        for panel_data in data.get("panels", []):
            panel = DashboardPanel(
                title=panel_data["title"],
                panel_type=panel_data["type"],
                data=panel_data["data"],
                config=panel_data.get("config", {}),
            )
            dashboard.panels.append(panel)
        
        if "metadata" in data:
            dashboard.metadata = data["metadata"]
        
        return dashboard
