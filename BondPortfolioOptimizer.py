# BondPortfolioOptimizer.py

import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple, Optional, Dict
from datetime import datetime

class BondPortfolioOptimizer:
    def __init__(
        self,
        bonds_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ):
        """
        Initialize the Bond Portfolio Optimizer with DataFrame inputs

        :param bonds_df: DataFrame with bond details from Bonds.xlsx
        :param prices_df: DataFrame with historical prices from HP.xlsx
        """
        # Prepare bond details
        self.bonds_df = bonds_df
        self.prices_df = prices_df

        # Extract key bond characteristics
        self.names = bonds_df['Issuer Name'].tolist()
        self.tickers = bonds_df['Ticker'].tolist()
        self.ratings = bonds_df['PEFINDO Rating'].tolist()

        # Convert coupon rates to decimal
        self.yields = bonds_df['Cpn'].astype(float) / 100

        # Parse maturity dates
        self.maturities = bonds_df['Maturity'].tolist()

        # Use Macaulay Duration
        self.durations = bonds_df['Mac Dur (Mid)'].astype(float)

        # Get latest prices
        latest_prices = prices_df.iloc[-1, 1:].tolist()
        self.prices = np.array(latest_prices)

        # Assume face value of 100 for each bond
        self.face_values = np.array([100] * len(self.names))

        # Number of bonds
        self.n_bonds = len(self.names)

        # Enhanced rating quality scores
        self.rating_scores = self._create_rating_scores()

    def _create_rating_scores(self) -> np.ndarray:
        """Create a more comprehensive rating scoring system."""
        rating_scores = {
            'idAAA': 1.0, 'idAA+': 0.95, 'idAA': 0.9, 'idAA-': 0.85,
            'idA+': 0.8, 'idA': 0.75, 'idA-': 0.7,
            'idBBB+': 0.65, 'idBBB': 0.6, 'idBBB-': 0.55,
            'idBB+': 0.5, 'idBB': 0.45, 'idBB-': 0.4,
            'idB+': 0.35, 'idB': 0.3, 'idB-': 0.25
        }
        return np.array([rating_scores.get(r, 0.0) for r in self.ratings])

    def calculate_covariance_matrix(self) -> np.ndarray:
        """Calculate the covariance matrix of price changes."""
        # Calculate price changes
        price_changes = self.prices_df.iloc[:, 1:].pct_change()
        return price_changes.cov().values
    
    def calculate_standard_deviations(self) -> np.ndarray:
        """Calculate the standard deviations of price changes for each bond."""
        price_changes = self.prices_df.iloc[:, 1:].pct_change()
        return price_changes.std().values

    def optimize_portfolio(
        self,
        total_investment: float,  # total_investment is now in Rupiah
        min_return: float,
        risk_aversion: float = 1.0,
        max_position_size: float = 0.4,
        min_rating_score: float = 0.6
    ) -> Tuple[np.ndarray, dict]:
        """
        Optimize the bond portfolio with enhanced constraints.
        
        :param total_investment: Total investment amount in Rupiah
        :param min_return: Minimum acceptable return for the portfolio
        :param risk_aversion: Risk aversion coefficient (default is 1.0)
        :param max_position_size: Maximum allowable position size as a fraction of total portfolio (default is 0.4)
        :param min_rating_score: Minimum average rating score for the portfolio (default is 0.6)
        :return: Tuple containing the optimized weights and a dictionary of results
        """
        w = cp.Variable(self.n_bonds)

        # Use covariance matrix from price changes
        cov_matrix = self.calculate_covariance_matrix()

        expected_returns = self.yields
        portfolio_return = w @ expected_returns
        portfolio_risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_position_size,
            portfolio_return >= min_return,
            w @ self.rating_scores >= min_rating_score
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Get weights and ensure they are non-negative
        weights = w.value
        weights = np.clip(weights, 0, None)
        weights = weights / np.sum(weights)  # Normalize to sum to 1

        # Store total investment for report generation
        self.total_investment = total_investment

        results = {
            'status': problem.status,
            'optimal_value': problem.value,
            'expected_return': float(portfolio_return.value),
            'portfolio_risk': float(portfolio_risk.value),
            'sharpe_ratio': float(portfolio_return.value / np.sqrt(portfolio_risk.value))
            if portfolio_risk.value > 0 else 0
        }

        return weights, results

    def generate_report(self, weights: np.ndarray) -> Tuple[pd.DataFrame, Dict, Dict[str, go.Figure]]:
        """Generate an enhanced report with detailed portfolio analysis."""
        combined_names = [f"{ticker} {maturity}" for ticker, maturity in zip(self.tickers, self.maturities)]
        std_devs = self.calculate_standard_deviations()
        portfolio_data = pd.DataFrame({
            'Issuer': combined_names,
            'Ticker': self.tickers,
            'Weight (%)': weights * 100,
            'Price ($)': self.prices,
            'Yield (%)': self.yields * 100,
            'Duration (years)': self.durations,
            'Rating': self.ratings,
            'Maturity': self.maturities,
            'Investment ($)': weights * self.total_investment,
            'Expected Return (%)': weights * self.yields * 100,
            'Standard Deviation': std_devs * 100
        })

        # Create summary statistics
        total_investment = self.total_investment
        portfolio_yield = np.sum(weights * self.yields) * 100
        portfolio_duration = np.sum(weights * self.durations)
        avg_rating_score = np.sum(weights * self.rating_scores)

        summary_data = {
            'Total Investment (Rp)': f"{total_investment:,.2f}",
            'Portfolio Yield (%)': f"{portfolio_yield:.2f}",
            'Portfolio Duration (years)': f"{portfolio_duration:.2f}",
            'Average Rating Score': f"{avg_rating_score:.2f}",
            'Number of Positions': f"{(weights > 0.001).sum()}",
            'Analysis Date': datetime.now().strftime('%Y-%m-%d')
        }

        # Generate interactive plots
        figures = self._create_interactive_plots(portfolio_data, weights)

        return portfolio_data, summary_data, figures

    def _create_interactive_plots(self, portfolio_data: pd.DataFrame, weights: np.ndarray) -> Dict[str, go.Figure]:
        """Create enhanced interactive plotly visualizations."""
        figures = {}
        combined_names = [f"{ticker} {maturity}" for ticker, maturity in zip(self.tickers, self.maturities)]

        # 1. Portfolio Composition Sunburst
        figures['composition'] = go.Figure(go.Sunburst(
            labels=combined_names + ['Portfolio'],
            parents=['Portfolio'] * len(self.names) + [''],
            values=np.append(weights * 100, weights.sum() * 100),
            branchvalues='total',
            textinfo='label+percent entry',
            maxdepth=2
        ))
        figures['composition'].update_layout(
            title='Portfolio Composition',
            width=800,
            height=800
        )

        # 2. Risk-Return Bubble Chart
        figures['risk_return'] = go.Figure()
        figures['risk_return'].add_trace(go.Scatter(
            x=self.durations,
            y=self.yields * 100,
            mode='markers',
            marker=dict(
                size=np.maximum(weights, 0) * 1000,
                color=self.rating_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Rating Score')
            ),
            text=combined_names,
            hovertemplate=
            '<b>%{text}</b><br>' +
            'Duration: %{x:.1f} years<br>' +
            'Yield: %{y:.2f}%<br>' +
            '<extra></extra>'
        ))
        figures['risk_return'].update_layout(
            title='Risk-Return Profile',
            xaxis_title='Duration (years)',
            yaxis_title='Yield (%)',
            width=800,
            height=600
        )

        # 3. Rating Distribution Treemap
        rating_data = pd.DataFrame({
            'Rating': self.ratings,
            'Weight': weights * 100
        }).groupby('Rating').sum().reset_index()

        figures['ratings'] = go.Figure(go.Treemap(
            labels=rating_data['Rating'],
            parents=[''] * len(rating_data),
            values=rating_data['Weight'],
            textinfo='label+percent parent',
            textfont=dict(size=20)
        ))
        figures['ratings'].update_layout(
            title='Portfolio Rating Distribution',
            width=800,
            height=600
        )

        return figures
