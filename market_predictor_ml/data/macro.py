"""
Macro-Economic Data Module.
Fetches and processes macroeconomic indicators for regime detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime


class MacroDataLoader:
    """
    Loads macroeconomic data from FRED or mock sources.
    
    Supports:
    - Interest rates
    - Inflation (CPI)
    - GDP growth
    - Yield curve spread
    """
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        
        try:
            from fredapi import Fred
            self.Fred = Fred
            self.fred_available = True
        except ImportError:
            self.fred_available = False
    
    def get_fred_client(self, api_key: Optional[str] = None):
        """Get FRED API client."""
        if not self.fred_available:
            return None
        
        import os
        key = api_key or os.getenv("FRED_API_KEY")
        if not key:
            return None
        
        return self.Fred(api_key=key)
    
    def fetch_macro_data(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch all macroeconomic indicators."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if self.use_mock or not self.fred_available:
            return self._get_mock_macro_data(start_date, end_date)
        
        try:
            fred = self.get_fred_client()
            if fred is None:
                return self._get_mock_macro_data(start_date, end_date)
            
            # Fetch series
            series_map = {
                "DFF": "federal_funds_rate",
                "CPIAUCSL": "cpi",
                "GDP": "gdp",
                "T10Y2Y": "yield_spread",
            }
            
            data = {}
            for series_id, name in series_map.items():
                try:
                    s = fred.get_series(series_id)
                    data[name] = s
                except Exception:
                    continue
            
            if len(data) == 0:
                return self._get_mock_macro_data(start_date, end_date)
            
            df = pd.DataFrame(data)
            df = df.loc[start_date:end_date]
            
            return df
            
        except Exception:
            return self._get_mock_macro_data(start_date, end_date)
    
    def _get_mock_macro_data(self, start: str, end: str) -> pd.DataFrame:
        """Generate realistic mock macro data."""
        dates = pd.date_range(start=start, end=end, freq="M")
        
        np.random.seed(42)
        
        # Simulate realistic macro series
        n = len(dates)
        
        # Federal funds rate (mean reverting around 2-5%)
        ff_rate = 2.5 + np.cumsum(np.random.randn(n) * 0.1)
        ff_rate = np.clip(ff_rate, 0, 10)
        
        # CPI (inflation, trending up)
        cpi = 250 + np.cumsum(np.random.randn(n) * 0.5)
        
        # GDP (quarterly, growing)
        gdp = np.interp(np.arange(n), np.linspace(0, n, n//3), 
                       20000 + np.cumsum(np.random.randn(n//3) * 100))
        
        # Yield spread (10Y-2Y, can invert)
        yield_spread = 1.0 + np.cumsum(np.random.randn(n) * 0.1)
        
        df = pd.DataFrame({
            "federal_funds_rate": ff_rate,
            "cpi": cpi,
            "gdp": gdp,
            "yield_spread": yield_spread,
        }, index=dates)
        
        return df
    
    def calculate_inflation_rate(self, cpi_series: pd.Series) -> pd.Series:
        """Calculate YoY inflation rate from CPI."""
        return cpi_series.pct_change(periods=12) * 100
    
    def calculate_gdp_growth(self, gdp_series: pd.Series) -> pd.Series:
        """Calculate YoY GDP growth rate."""
        return gdp_series.pct_change(periods=4) * 100  # Quarterly data


class RegimeDetector:
    """
    Detects market regimes based on macroeconomic indicators.
    
    Regimes:
    - Expansion: High GDP, low unemployment, normal yield curve
    - Recession: Low/negative GDP, inverted yield curve
    - Stagflation: High inflation, low growth
    - Recovery: Improving GDP, accommodative policy
    """
    
    def __init__(self):
        self.macro_loader = MacroDataLoader()
    
    def detect_regime(
        self,
        macro_df: Optional[pd.DataFrame] = None,
        latest_date: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Detect current economic regime.
        
        Returns:
            Dictionary with regime name, confidence, and indicators
        """
        if macro_df is None:
            macro_df = self.macro_loader.fetch_macro_data()
        
        if len(macro_df) == 0:
            return {"regime": "Unknown", "confidence": 0.0}
        
        # Get latest values
        latest = macro_df.iloc[-1]
        
        # Calculate derived metrics
        inflation = self.macro_loader.calculate_inflation_rate(macro_df["cpi"]).iloc[-1]
        gdp_growth = self.macro_loader.calculate_gdp_growth(macro_df["gdp"]).iloc[-1] if "gdp" in macro_df else 2.0
        yield_spread = latest.get("yield_spread", 1.0)
        ff_rate = latest.get("federal_funds_rate", 2.5)
        
        # Rule-based regime classification
        scores = {
            "Expansion": 0,
            "Recession": 0,
            "Stagflation": 0,
            "Recovery": 0,
        }
        
        # Expansion signals
        if gdp_growth > 2:
            scores["Expansion"] += 2
        if yield_spread > 0.5:
            scores["Expansion"] += 1
        
        # Recession signals
        if gdp_growth < 0:
            scores["Recession"] += 2
        if yield_spread < 0:
            scores["Recession"] += 2
        
        # Stagflation signals
        if inflation > 5 and gdp_growth < 1:
            scores["Stagflation"] += 3
        
        # Recovery signals
        if gdp_growth > 0 and gdp_growth < 2 and ff_rate < 2:
            scores["Recovery"] += 2
        
        # Determine regime
        regime = max(scores, key=scores.get)
        confidence = scores[regime] / 6.0  # Normalize to 0-1
        
        return {
            "regime": regime,
            "confidence": min(confidence, 1.0),
            "indicators": {
                "gdp_growth": gdp_growth,
                "inflation": inflation,
                "yield_spread": yield_spread,
                "federal_funds_rate": ff_rate,
            },
            "scores": scores,
        }
    
    def create_regime_features(
        self,
        price_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add regime indicator to price data.
        
        Returns DataFrame with regime column (0=Expansion, 1=Recession, 2=Stagflation, 3=Recovery)
        """
        if macro_df is None:
            macro_df = self.macro_loader.fetch_macro_data()
        
        regime_map = {
            "Expansion": 0,
            "Recession": 1,
            "Stagflation": 2,
            "Recovery": 3,
        }
        
        # Simple approach: assign regime based on date
        regime_series = pd.Series(index=price_df.index, dtype=int)
        
        for idx in price_df.index:
            # Find closest macro date
            macro_dates = macro_df[macro_df.index <= idx].index
            if len(macro_dates) > 0:
                closest = macro_dates[-1]
                regime_info = self.detect_regime(macro_df.loc[:closest])
                regime_series[idx] = regime_map.get(regime_info["regime"], 0)
            else:
                regime_series[idx] = 0  # Default expansion
        
        result = price_df.copy()
        result["regime"] = regime_series
        
        return result
