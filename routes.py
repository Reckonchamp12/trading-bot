from flask import render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from app import app, db
from models import User, Portfolio, Trade, TradingSignal, Position, BacktestResult
from ml_engine import MLEngine
from backtest_engine import BacktestEngine
from risk_manager import RiskManager
from trading_signals import SignalGenerator
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, FloatField, SubmitField
from wtforms.validators import DataRequired, Email, Length
from flask_wtf.csrf import CSRFProtect
from datetime import timedelta
import json
import logging

# CSRF protection
csrf = CSRFProtect(app)

# App config updates for secure sessions
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,
    REMEMBER_COOKIE_DURATION=timedelta(days=1),
    SECRET_KEY='your-secret-key'
)

# Initialize components
ml_engine = MLEngine()
risk_manager = RiskManager()
signal_generator = SignalGenerator()

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Register')

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html', form=LoginForm())

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists', 'error')
            return redirect(url_for('login'))

        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered', 'error')
            return redirect(url_for('login'))

        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)

        try:
            db.session.add(user)
            db.session.commit()
            portfolio = Portfolio(user_id=user.id, name="Main Portfolio", initial_capital=100000.0, current_capital=100000.0, is_paper_trading=True)
            db.session.add(portfolio)
            db.session.commit()
            flash('Registration successful', 'success')
            login_user(user)
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Registration error: {e}")
            flash('Registration failed', 'error')
            return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/generate_signal', methods=['POST'])
@login_required
def generate_signal():
    try:
        symbol = request.json['symbol']
        signal = signal_generator.generate_signal(symbol)
        if signal:
            trading_signal = TradingSignal(
                symbol=symbol,
                signal_type=signal['type'],
                confidence=signal['confidence'],
                strategy_name=signal['strategy'],
                target_price=signal.get('target_price'),
                stop_loss_price=signal.get('stop_loss'),
                timeframe=signal['timeframe'],
                features_json=json.dumps(signal.get('features', {})),
                model_version=signal.get('model_version')
            )
            db.session.add(trading_signal)
            db.session.commit()
            return jsonify({'success': True, 'signal': signal})
        return jsonify({'success': False, 'message': 'No signal generated'})
    except Exception as e:
        logging.error(f"Signal generation error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500
