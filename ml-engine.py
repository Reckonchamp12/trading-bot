import os
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from feature_engineering import FeatureEngineer
from market_data import MarketDataProvider
from app import db
from models import ModelPerformance
import json

class MLEngine:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.market_data = MarketDataProvider()
        self.models = {}
        self.model_versions = {}

    def prepare_features(self, symbol, lookback_days=365):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            df = self.market_data.get_ohlcv_data(symbol, start_date, end_date)

            if df.empty:
                logging.warning(f"No market data available for {symbol}")
                return None, None

            features_df = self.feature_engineer.create_features(df)
            features_df['target'] = (features_df['close'].shift(-1) > features_df['close']).astype(int)
            features_df = features_df[:-1]

            feature_columns = [col for col in features_df.columns if col not in ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            X = features_df[feature_columns].fillna(0)
            y = features_df['target']
            return X, y
        except Exception as e:
            logging.error(f"Feature preparation error for {symbol}: {e}")
            return None, None

    def train_model(self, symbol, model_type='xgboost', retrain=False):
        try:
            model_key = f"{symbol}_{model_type}"

            if model_key in self.models and not retrain:
                logging.info(f"Model {model_key} already trained")
                return True

            X, y = self.prepare_features(symbol, lookback_days=730)
            if X is None or len(X) < 100:
                logging.warning(f"Insufficient data to train model for {symbol}")
                return False

            tscv = TimeSeriesSplit(n_splits=5)

            model = xgb.XGBClassifier(...) if model_type == 'xgboost' else lgb.LGBMClassifier(...)
            if model_type not in ['xgboost', 'lightgbm']:
                raise ValueError(f"Unsupported model type: {model_type}")

            cv_scores, feature_importance = [], None

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                cv_scores.append(accuracy_score(y_val, y_pred))
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))

            model.fit(X, y)
            self.models[model_key], self.model_versions[model_key] = model, datetime.now().strftime('%Y%m%d_%H%M')

            y_pred_final = model.predict(X)
            accuracy = accuracy_score(y, y_pred_final)
            precision = precision_score(y, y_pred_final, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred_final, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred_final, average='weighted', zero_division=0)

            model_perf = ModelPerformance(
                model_name=model_key,
                model_version=self.model_versions[model_key],
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_date=datetime.now(),
                feature_importance_json=json.dumps(feature_importance) if feature_importance else None,
                is_active=True
            )

            db.session.add(model_perf)
            db.session.commit()

            logging.info(f"Model {model_key} trained successfully. Accuracy: {accuracy:.4f}")
            return True
        except Exception as e:
            logging.error(f"Model training error for {symbol}: {e}")
            return False

    def predict(self, symbol, model_type='xgboost'):
        try:
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.models:
                if not self.train_model(symbol, model_type):
                    return None

            X, _ = self.prepare_features(symbol, lookback_days=365)
            if X is None or len(X) == 0:
                return None

            latest_features = X.iloc[-1:].fillna(0)
            model = self.models[model_key]
            prediction = model.predict(latest_features)[0]
            confidence = model.predict_proba(latest_features)[0][prediction]

            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'signal': 'buy' if prediction == 1 else 'sell',
                'model_version': self.model_versions.get(model_key, 'unknown'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logging.error(f"Prediction error for {symbol}: {e}")
            return None

    def get_feature_importance(self, symbol, model_type='xgboost'):
        try:
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.models:
                return None

            X, _ = self.prepare_features(symbol, lookback_days=30)
            if X is None:
                return None

            model = self.models[model_key]
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(X.columns, model.feature_importances_))
                return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
            return None
        except Exception as e:
            logging.error(f"Feature importance error for {symbol}: {e}")
            return None

    def model_drift_detection(self, symbol, model_type='xgboost', window_days=30):
        try:
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.models:
                return {'drift_detected': False, 'reason': 'Model not found'}

            X, y = self.prepare_features(symbol, lookback_days=window_days + 100)
            if X is None or len(X) < window_days:
                return {'drift_detected': False, 'reason': 'Insufficient data'}

            recent_X, recent_y = X.iloc[-window_days:], y.iloc[-window_days:]
            model = self.models[model_key]
            recent_accuracy = accuracy_score(recent_y, model.predict(recent_X))

            historical_perf = ModelPerformance.query.filter_by(model_name=model_key, is_active=True).order_by(ModelPerformance.training_date.desc()).first()
            if not historical_perf:
                return {'drift_detected': False, 'reason': 'No historical performance data'}

            drift = (historical_perf.accuracy - recent_accuracy) > 0.1
            return {
                'drift_detected': drift,
                'historical_accuracy': historical_perf.accuracy,
                'recent_accuracy': recent_accuracy,
                'accuracy_drop': historical_perf.accuracy - recent_accuracy,
                'recommendation': 'Retrain model' if drift else 'Model performing well'
            }
        except Exception as e:
            logging.error(f"Drift detection error for {symbol}: {e}")
            return {'drift_detected': False, 'reason': f'Error: {str(e)}'}

    def save_model(self, symbol, model_type='xgboost', filepath=None):
        try:
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.models:
                return False

            os.makedirs('models', exist_ok=True)
            filepath = filepath or f"models/{model_key}_{self.model_versions[model_key]}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(self.models[model_key], f)

            logging.info(f"Model {model_key} saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Model saving error: {e}")
            return False

    def load_model(self, symbol, model_type='xgboost', filepath=None):
        try:
            model_key = f"{symbol}_{model_type}"
            if not filepath:
                files = [f for f in os.listdir('models') if f.startswith(model_key)]
                if not files:
                    return False
                filepath = f"models/{max(files)}"

            with open(filepath, 'rb') as f:
                model = pickle.load(f)

            self.models[model_key] = model
            self.model_versions[model_key] = 'loaded'
            logging.info(f"Model {model_key} loaded from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Model loading error: {e}")
            return False
