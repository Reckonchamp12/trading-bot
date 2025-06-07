from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Index, UniqueConstraint
# from sqlalchemy.dialects.postgresql import JSON  # Uncomment if using PostgreSQL

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    portfolios = db.relationship('Portfolio', backref='user', lazy=True)
    trades = db.relationship('Trade', backref='user', lazy=True)
    positions = db.relationship('Position', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    __table_args__ = (
        Index('idx_user_username', 'username'),
        Index('idx_user_email', 'email'),
    )


class Portfolio(db.Model):
    __tablename__ = 'portfolios'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    initial_capital = db.Column(db.Float, nullable=False, default=100000.0)
    current_capital = db.Column(db.Float, nullable=False)
    is_paper_trading = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trades = db.relationship('Trade', backref='portfolio', lazy=True)
    positions = db.relationship('Position', backref='portfolio', lazy=True)


class Trade(db.Model):
    __tablename__ = 'trades'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    side = db.Column(db.String(10), nullable=False)  # 'buy' or 'sell'
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    commission = db.Column(db.Float, default=0.0)
    pnl = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='executed')  # 'pending', 'executed', 'cancelled'
    signal_confidence = db.Column(db.Float, nullable=True)
    strategy_name = db.Column(db.String(50), nullable=True)
    executed_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_trade_symbol_date', 'symbol', 'executed_at'),
        Index('idx_trade_user_portfolio', 'user_id', 'portfolio_id'),
        Index('idx_trade_executed_at', 'executed_at'),
    )


class MarketData(db.Model):
    __tablename__ = 'market_data'

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    open_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    timeframe = db.Column(db.String(10), nullable=False)

    __table_args__ = (
        Index('idx_market_data_symbol_time', 'symbol', 'timestamp'),
        Index('idx_market_data_timeframe', 'timeframe'),
    )


class TradingSignal(db.Model):
    __tablename__ = 'trading_signals'

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    signal_type = db.Column(db.String(10), nullable=False)  # 'buy', 'sell', 'hold'
    confidence = db.Column(db.Float, nullable=False)
    strategy_name = db.Column(db.String(50), nullable=False)
    target_price = db.Column(db.Float, nullable=True)
    stop_loss_price = db.Column(db.Float, nullable=True)
    timeframe = db.Column(db.String(10), nullable=False)
    features_json = db.Column(db.Text, nullable=True)  # Or use JSON
    model_version = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)

    __table_args__ = (
        Index('idx_signal_symbol_time', 'symbol', 'created_at'),
        Index('idx_signal_strategy', 'strategy_name'),
        Index('idx_signal_expires_at', 'expires_at'),
    )


class BacktestResult(db.Model):
    __tablename__ = 'backtest_results'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    strategy_name = db.Column(db.String(50), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    initial_capital = db.Column(db.Float, nullable=False)
    final_capital = db.Column(db.Float, nullable=False)
    total_return = db.Column(db.Float, nullable=False)
    sharpe_ratio = db.Column(db.Float, nullable=True)
    sortino_ratio = db.Column(db.Float, nullable=True)
    max_drawdown = db.Column(db.Float, nullable=True)
    win_rate = db.Column(db.Float, nullable=True)
    total_trades = db.Column(db.Integer, nullable=True)
    avg_trade_duration = db.Column(db.Float, nullable=True)  # hours
    results_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ModelPerformance(db.Model):
    __tablename__ = 'model_performance'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    model_version = db.Column(db.String(20), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    validation_start = db.Column(db.DateTime, nullable=True)
    validation_end = db.Column(db.DateTime, nullable=True)
    feature_importance_json = db.Column(db.Text, nullable=True)
    hyperparameters_json = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)

    __table_args__ = (
        UniqueConstraint('model_name', 'model_version', name='uq_model_name_version'),
    )


class Position(db.Model):
    __tablename__ = 'positions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    avg_price = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float, nullable=True)
    unrealized_pnl = db.Column(db.Float, default=0.0)
    stop_loss_price = db.Column(db.Float, nullable=True)
    take_profit_price = db.Column(db.Float, nullable=True)
    opened_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_position_user_symbol', 'user_id', 'symbol'),
    )
