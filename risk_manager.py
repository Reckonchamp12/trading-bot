import logging
from datetime import datetime
from models import Trade, Position, Portfolio
from app import db
import numpy as np

class RiskManager:
    def __init__(self, risk_config=None):
        # Configurable parameters with defaults
        config = risk_config or {}
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% loss
        self.max_portfolio_correlation = config.get('max_portfolio_correlation', 0.7)
        self.min_liquidity_ratio = config.get('min_liquidity_ratio', 0.2)

        self.risk_score_weights = config.get('risk_score_weights', {
            'position_size': 25,
            'daily_loss': 25,
            'liquidity': 20,
            'exposure': 30,
            'margin': 20
        })

    def check_trade_risk(self, portfolio, symbol, side, quantity, price):
        """Comprehensive risk check for trade execution"""
        try:
            risk_checks = [
                self._check_position_size(portfolio, symbol, quantity, price),
                self._check_daily_loss_limit(portfolio),
                self._check_liquidity(portfolio, side, quantity, price),
                self._check_portfolio_exposure(portfolio, symbol, side, quantity, price),
                self._check_margin_requirements(portfolio, side, quantity, price)
            ]

            all_approved = all(check['approved'] for check in risk_checks)
            failed_checks = [check for check in risk_checks if not check['approved']]

            stop_loss_info = self.calculate_stop_loss(price, 0.75)  # default confidence

            return {
                'approved': all_approved,
                'reason': failed_checks[0]['reason'] if failed_checks else 'All risk checks passed',
                'risk_score': self._calculate_risk_score(risk_checks),
                'checks': risk_checks,
                'stop_loss': stop_loss_info
            }
        except Exception as e:
            logging.error(f"Risk check error: {e}")
            return {'approved': False, 'reason': f'Risk check failed: {str(e)}'}

    def _check_position_size(self, portfolio, symbol, quantity, price):
        try:
            trade_value = quantity * price
            position_percentage = trade_value / portfolio.current_capital
            if position_percentage > self.max_position_size:
                return {
                    'approved': False,
                    'reason': f'Position size ({position_percentage:.2%}) exceeds limit ({self.max_position_size:.2%})',
                    'check_type': 'position_size'
                }
            return {
                'approved': True,
                'reason': 'Position size within limits',
                'check_type': 'position_size',
                'value': position_percentage
            }
        except Exception as e:
            return {'approved': False, 'reason': str(e), 'check_type': 'position_size'}

    def _check_daily_loss_limit(self, portfolio):
        try:
            today = datetime.now().date()
            daily_trades = Trade.query.filter(
                Trade.portfolio_id == portfolio.id,
                Trade.executed_at >= today
            ).all()
            daily_pnl = sum(trade.pnl for trade in daily_trades)
            loss_pct = abs(daily_pnl) / portfolio.current_capital if daily_pnl < 0 else 0
            if loss_pct > self.max_daily_loss:
                return {
                    'approved': False,
                    'reason': f'Daily loss limit ({loss_pct:.2%}) exceeded',
                    'check_type': 'daily_loss'
                }
            return {
                'approved': True,
                'reason': 'Daily loss within limits',
                'check_type': 'daily_loss',
                'value': loss_pct
            }
        except Exception as e:
            return {'approved': False, 'reason': str(e), 'check_type': 'daily_loss'}

    def _check_liquidity(self, portfolio, side, quantity, price):
        try:
            if side == 'buy':
                cost = quantity * price * 1.01
                if portfolio.current_capital < cost:
                    return {
                        'approved': False,
                        'reason': f'Insufficient capital for buy',
                        'check_type': 'liquidity'
                    }
                remaining = portfolio.current_capital - cost
                ratio = remaining / portfolio.current_capital
                if ratio < self.min_liquidity_ratio:
                    return {
                        'approved': False,
                        'reason': f'Liquidity ratio ({ratio:.2%}) too low',
                        'check_type': 'liquidity'
                    }
            return {
                'approved': True,
                'reason': 'Sufficient liquidity',
                'check_type': 'liquidity'
            }
        except Exception as e:
            return {'approved': False, 'reason': str(e), 'check_type': 'liquidity'}

    def _check_portfolio_exposure(self, portfolio, symbol, side, quantity, price):
        try:
            positions = Position.query.filter_by(portfolio_id=portfolio.id).all()
            symbol_exposure = {}
            total_exposure = 0
            for pos in positions:
                val = pos.quantity * (pos.current_price or pos.avg_price)
                symbol_exposure[pos.symbol] = symbol_exposure.get(pos.symbol, 0) + val
                total_exposure += val
            if side == 'buy':
                trade_val = quantity * price
                symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + trade_val
                total_exposure += trade_val
            max_expo = max(symbol_exposure.values(), default=0)
            concentration = max_expo / portfolio.current_capital
            if concentration > self.max_position_size * 2:
                return {'approved': False, 'reason': 'Symbol concentration too high', 'check_type': 'exposure'}
            exposure_ratio = total_exposure / portfolio.current_capital
            if exposure_ratio > 0.9:
                return {'approved': False, 'reason': 'Total portfolio exposure too high', 'check_type': 'exposure'}
            return {
                'approved': True,
                'reason': 'Exposure within limits',
                'check_type': 'exposure',
                'total_exposure': exposure_ratio,
                'symbol_concentration': concentration
            }
        except Exception as e:
            return {'approved': False, 'reason': str(e), 'check_type': 'exposure'}

    def _check_margin_requirements(self, portfolio, side, quantity, price):
        try:
            trade_value = quantity * price
            margin_required = trade_value * 0.5
            if side == 'buy' and portfolio.current_capital < margin_required:
                return {
                    'approved': False,
                    'reason': 'Insufficient margin',
                    'check_type': 'margin'
                }
            return {
                'approved': True,
                'reason': 'Margin sufficient',
                'check_type': 'margin'
            }
        except Exception as e:
            return {'approved': False, 'reason': str(e), 'check_type': 'margin'}

    def _calculate_risk_score(self, checks):
        try:
            score = 0
            for check in checks:
                weight = self.risk_score_weights.get(check['check_type'], 10)
                if not check['approved']:
                    score += weight
                elif 'value' in check:
                    if check['check_type'] == 'position_size':
                        score += (check['value'] / self.max_position_size) * weight * 0.6
                    elif check['check_type'] == 'daily_loss':
                        score += (check['value'] / self.max_daily_loss) * weight * 0.6
                    elif check['check_type'] == 'exposure':
                        score += check.get('total_exposure', 0) * weight * 0.6
            return min(100, max(0, round(score)))
        except Exception as e:
            logging.error(f"Risk score calc error: {e}")
            return 50

    def calculate_stop_loss(self, entry_price, signal_confidence, volatility=None):
        try:
            base = 0.02
            confidence = 0.5 + 0.5 * signal_confidence
            vol_factor = 1 + (volatility * 10 if volatility else 0)
            sl_pct = min(base * confidence * vol_factor, 0.05)
            return {
                'stop_loss_price': entry_price * (1 - sl_pct),
                'stop_loss_percentage': sl_pct,
                'risk_amount': entry_price * sl_pct
            }
        except Exception as e:
            logging.error(f"Stop loss error: {e}")
            return {
                'stop_loss_price': entry_price * 0.98,
                'stop_loss_percentage': 0.02
            }
