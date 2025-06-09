from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.feature_engineer import FeatureEngineer
from utils.performance_metrics import PerformanceMetrics
from models.model_selector import ModelSelector
from risk.risk_manager import RiskManager

class BacktestEngine:
    def __init__(self, data_handler, feature_engineer=None, model_selector=None, risk_manager=None):
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.model_selector = model_selector or ModelSelector()
        self.risk_manager = risk_manager or RiskManager()

    def run_backtest(self, symbol, start_date, end_date, strategy='ml_signals', params=None):
        df = self.data_handler.get_data(symbol, start_date, end_date)
        features_df = self.feature_engineer.create_features(df)

        capital = 100000  # Starting capital
        commission_rate = 0.001  # Commission per trade
        trades = []
        portfolio_values = []

        if strategy == 'ml_signals':
            results = self._run_ml_strategy_backtest(features_df, symbol, capital, commission_rate, trades, portfolio_values)
        elif strategy == 'technical':
            results = self._run_technical_strategy_backtest(features_df, symbol, capital, commission_rate, trades, portfolio_values)
        elif strategy == 'buy_hold':
            results = self._run_buy_hold_backtest(features_df, capital, portfolio_values)
        else:
            raise ValueError("Invalid strategy specified.")

        metrics = PerformanceMetrics.calculate_all(portfolio_values, trades)
        return results, metrics, trades, portfolio_values

    def _run_ml_strategy_backtest(self, features_df, symbol, capital, commission_rate, trades, portfolio_values):
        position = 0
        entry_price = 0
        stop_loss = 0
        cash = capital
        shares_held = 0

        train_window = 90
        retrain_freq = 5

        for i in range(train_window, len(features_df)):
            if (i - train_window) % retrain_freq == 0:
                train = features_df.iloc[i-train_window:i]
                X_train = train[["Open", "High", "Low", "Close", "Volume"]]
                y_train = train["Target"]
                model = LogisticRegression()
                model.fit(X_train, y_train)

            test = features_df.iloc[i]
            X_test = test[["Open", "High", "Low", "Close", "Volume"]].values.reshape(1, -1)
            signal = model.predict(X_test)[0]

            current_price = test['Close']
            current_date = test.name

            if signal == 1 and position == 0:
                position_size = self.risk_manager.calculate_position_size(symbol, current_price, capital)
                shares_held = int(position_size / current_price)
                position = 1
                entry_price = current_price
                stop_loss = self.risk_manager.calculate_stop_loss(symbol, current_price)
                cash -= shares_held * current_price * (1 + commission_rate)
                trades.append({'entry_date': current_date, 'entry_price': entry_price, 'shares': shares_held})

            elif signal == -1 and position == 1:
                exit_price = current_price
                cash += shares_held * exit_price * (1 - commission_rate)
                trades[-1].update({'exit_date': current_date, 'exit_price': exit_price})
                position = 0
                shares_held = 0

            elif position == 1 and current_price < stop_loss:
                exit_price = current_price
                cash += shares_held * exit_price * (1 - commission_rate)
                trades[-1].update({'exit_date': current_date, 'exit_price': exit_price, 'stop_loss_triggered': True})
                position = 0
                shares_held = 0

            portfolio_value = cash + shares_held * current_price
            portfolio_values.append({'date': current_date, 'value': portfolio_value})

        return {'final_value': portfolio_value, 'initial_capital': capital}

    def _run_technical_strategy_backtest(self, features_df, symbol, capital, commission_rate, trades, portfolio_values):
        position = 0
        entry_price = 0
        cash = capital
        shares_held = 0

        for i in range(1, len(features_df)):
            prev_row = features_df.iloc[i - 1]
            row = features_df.iloc[i]
            current_price = row['Close']
            current_date = row.name

            signal = 0
            if prev_row['SMA_10'] > prev_row['SMA_50'] and row['SMA_10'] <= row['SMA_50']:
                signal = -1
            elif prev_row['SMA_10'] < prev_row['SMA_50'] and row['SMA_10'] >= row['SMA_50']:
                signal = 1

            if signal == 1 and position == 0:
                shares_held = int(cash / current_price)
                position = 1
                entry_price = current_price
                cash -= shares_held * current_price * (1 + commission_rate)
                trades.append({'entry_date': current_date, 'entry_price': entry_price, 'shares': shares_held})

            elif signal == -1 and position == 1:
                exit_price = current_price
                cash += shares_held * exit_price * (1 - commission_rate)
                trades[-1].update({'exit_date': current_date, 'exit_price': exit_price})
                position = 0
                shares_held = 0

            portfolio_value = cash + shares_held * current_price
            portfolio_values.append({'date': current_date, 'value': portfolio_value})

        return {'final_value': portfolio_value, 'initial_capital': capital}

    def _run_buy_hold_backtest(self, features_df, capital, portfolio_values):
        entry_price = features_df.iloc[0]['Close']
        exit_price = features_df.iloc[-1]['Close']
        shares = int(capital / entry_price)
        final_value = shares * exit_price

        for i in range(len(features_df)):
            current_price = features_df.iloc[i]['Close']
            current_date = features_df.index[i]
            portfolio_value = shares * current_price
            portfolio_values.append({'date': current_date, 'value': portfolio_value})

        return {'final_value': final_value, 'initial_capital': capital}
