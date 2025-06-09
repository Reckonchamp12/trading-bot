import pandas as pd
import numpy as np
import talib
import logging
from typing import Optional

class FeatureEngineer:
    def __init__(self):
        self.feature_config = {
            'technical_indicators': True,
            'price_patterns': True,
            'volume_indicators': True,
            'volatility_indicators': True,
            'momentum_indicators': True,
            'statistical_features': True
        }

    def create_features(self, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        """Create comprehensive features from OHLCV data"""
        try:
            if df.empty:
                return df

            if config is None:
                config = self.feature_config

            features_df = df.copy()

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in features_df.columns for col in required_cols):
                logging.error(f"Missing required columns. Need: {required_cols}")
                return df

            open_prices = features_df['open'].values
            high_prices = features_df['high'].values
            low_prices = features_df['low'].values
            close_prices = features_df['close'].values
            volume = features_df['volume'].values

            if config.get('technical_indicators', True):
                features_df = self._add_technical_indicators(features_df, open_prices, high_prices,
                                                             low_prices, close_prices, volume)

            if config.get('price_patterns', True):
                features_df = self._add_price_patterns(features_df, close_prices, high_prices, low_prices)

            if config.get('volume_indicators', True):
                features_df = self._add_volume_indicators(features_df, close_prices, volume)

            if config.get('volatility_indicators', True):
                features_df = self._add_volatility_indicators(features_df, high_prices, low_prices, close_prices)

            if config.get('momentum_indicators', True):
                features_df = self._add_momentum_indicators(features_df, close_prices, volume)

            if config.get('statistical_features', True):
                features_df = self._add_statistical_features(features_df, close_prices)

            features_df = features_df.fillna(method='ffill').fillna(0)

            logging.debug(f"Created {len(features_df.columns)} features")
            return features_df

        except Exception as e:
            logging.error(f"Feature engineering error: {e}")
            return df

    def _add_technical_indicators(self, df, open_prices, high_prices, low_prices, close_prices, volume):
        try:
            df['sma_5'] = talib.SMA(close_prices, timeperiod=5)
            df['sma_10'] = talib.SMA(close_prices, timeperiod=10)
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['sma_200'] = talib.SMA(close_prices, timeperiod=200)

            df['ema_5'] = talib.EMA(close_prices, timeperiod=5)
            df['ema_10'] = talib.EMA(close_prices, timeperiod=10)
            df['ema_20'] = talib.EMA(close_prices, timeperiod=20)
            df['ema_50'] = talib.EMA(close_prices, timeperiod=50)

            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)

            df['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
            df['rsi_7'] = talib.RSI(close_prices, timeperiod=7)

            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist

            stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices, 
                                           fastk_period=14, slowk_period=3, slowk_matype=0,
                                           slowd_period=3, slowd_matype=0)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)

            df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

            return df
        except Exception as e:
            logging.error(f"Technical indicators error: {e}")
            return df

    def _add_price_patterns(self, df, close_prices, high_prices, low_prices):
        try:
            # Use shift instead of np.roll for time series consistency
            df['price_change'] = close_prices - df['close'].shift(1)
            df['price_change_pct'] = df['price_change'] / df['close'].shift(1)

            df['hl_spread'] = high_prices - low_prices
            df['hl_spread_pct'] = df['hl_spread'] / close_prices

            df['gap_up'] = (df['low'].shift(-1) > high_prices).astype(int)
            df['gap_down'] = (df['high'].shift(-1) < low_prices).astype(int)

            df['local_high'] = ((high_prices > df['high'].shift(1)) & (high_prices > df['high'].shift(-1))).astype(int)
            df['local_low'] = ((low_prices < df['low'].shift(1)) & (low_prices < df['low'].shift(-1))).astype(int)

            # Doji pattern: small body relative to range
            df['doji'] = ((abs(df['open'] - df['close']) / (df['high'] - df['low'] + 1e-6)) < 0.1).astype(int)

            body_size = abs(df['close'] - df['open'])
            upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
            lower_shadow = df[['open', 'close']].min(axis=1) - df['low']

            df['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < 0.1 * body_size)).astype(int)

            df['price_position'] = (close_prices - low_prices) / (high_prices - low_prices + 1e-6)

            return df
        except Exception as e:
            logging.error(f"Price patterns error: {e}")
            return df

    def _add_volume_indicators(self, df, close_prices, volume):
        try:
            df['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)

            df['volume_ratio'] = volume / (df['volume_sma_20'] + 1e-6)

            df['obv'] = talib.OBV(close_prices, volume)

            # VPT: Volume Price Trend
            # Use cumulative sum of volume * price change percentage (avoid NaNs)
            df['vpt'] = (volume * df['price_change_pct'].fillna(0)).rolling(10).sum()

            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], volume, timeperiod=14)
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], volume)
            df['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], volume)

            df['vwap_approx'] = (volume * close_prices).rolling(20).sum() / (volume.rolling(20).sum() + 1e-6)

            return df
        except Exception as e:
            logging.error(f"Volume indicators error: {e}")
            return df

    def _add_volatility_indicators(self, df, high_prices, low_prices, close_prices):
        try:
            df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            df['atr_pct'] = df['atr'] / (close_prices + 1e-6)
            df['true_range'] = talib.TRANGE(high_prices, low_prices, close_prices)

            close_series = pd.Series(close_prices)
            df['volatility_10'] = close_series.rolling(10).std()
            df['volatility_20'] = close_series.rolling(20).std()
            df['volatility_50'] = close_series.rolling(50).std()

            df['norm_volatility'] = df['volatility_20'] / (close_prices + 1e-6)
            df['vol_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-6)

            ema_20 = talib.EMA(close_prices, 20)
            atr_20 = talib.ATR(high_prices, low_prices, close_prices, 20)
            df['keltner_upper'] = ema_20 + (2 * atr_20)
            df['keltner_lower'] = ema_20 - (2 * atr_20)
            df['keltner_position'] = (close_prices - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'] + 1e-6)

            return df
        except Exception as e:
            logging.error(f"Volatility indicators error: {e}")
            return df

    def _add_momentum_indicators(self, df, close_prices, volume):
        try:
            df['roc_5'] = talib.ROC(close_prices, timeperiod=5)
            df['roc_10'] = talib.ROC(close_prices, timeperiod=10)
            df['roc_20'] = talib.ROC(close_prices, timeperiod=20)

            df['momentum_10'] = talib.MOM(close_prices, timeperiod=10)
            df['momentum_20'] = talib.MOM(close_prices, timeperiod=20)

            df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'])

            df['trix'] = talib.TRIX(close_prices, timeperiod=14)

            price_ema_12 = talib.EMA(close_prices, 12)
            price_ema_26 = talib.EMA(close_prices, 26)
            df['price_osc'] = ((price_ema_12 - price_ema_26) / (price_ema_26 + 1e-6)) * 100

            df['rs_volume'] = df['roc_10'] / np.log1p(volume)

            return df
        except Exception as e:
            logging.error(f"Momentum indicators error: {e}")
            return df

    def _add_statistical_features(self, df, close_prices):
        try:
            price_series = pd.Series(close_prices)
            for window in [5, 10, 20, 50]:
                df[f'rolling_mean_{window}'] = price_series.rolling(window).mean()
                df[f'rolling_std_{window}'] = price_series.rolling(window).std()
                df[f'zscore_{window}'] = (close_prices - df[f'rolling_mean_{window}']) / (df[f'rolling_std_{window}'] + 1e-6)
                df[f'skewness_{window}'] = price_series.rolling(window).skew()
                df[f'kurtosis_{window}'] = price_series.rolling(window).kurt()
                # Rolling percentile rank workaround: rank within rolling window divided by window size
                df[f'percentile_rank_{window}'] = price_series.rolling(window).apply(lambda x: x.rank(pct=True).iloc[-1] if len(x) == window else np.nan)

            for lag in [1, 5, 10]:
                df[f'autocorr_lag_{lag}'] = price_series.rolling(50).apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan)

            def calculate_trend_slope(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                return slope

            for window in [10, 20, 50]:
                df[f'trend_slope_{window}'] = price_series.rolling(window).apply(calculate_trend_slope)

            rolling_high = price_series.rolling(50).max()
            rolling_low = price_series.rolling(50).min()
            fib_range = rolling_high - rolling_low

            df['fib_23.6'] = rolling_high - (fib_range * 0.236)
            df['fib_38.2'] = rolling_high - (fib_range * 0.382)
            df['fib_50.0'] = rolling_high - (fib_range * 0.500)
            df['fib_61.8'] = rolling_high - (fib_range * 0.618)

            for level in ['23.6', '38.2', '50.0', '61.8']:
                df[f'dist_to_fib_{level}'] = abs(close_prices - df[f'fib_{level}']) / (close_prices + 1e-6)

            return df
        except Exception as e:
            logging.error(f"Statistical features error: {e}")
            return df

    def get_feature_importance_groups(self):
        return {
            'technical_indicators': [
                'sma_', 'ema_', 'bb_', 'rsi_', 'macd', 'stoch_', 'adx', 'cci', 'williams_r'
            ],
            'price_patterns': [
                'price_change', 'hl_spread', 'gap_', 'local_', 'doji', 'hammer', 'price_position'
            ],
            'volume_indicators': [
                'volume_', 'obv', 'vpt', 'mfi', 'ad', 'vwap'
            ],
            'volatility_indicators': [
                'atr', 'true_range', 'volatility_', 'keltner_'
            ],
            'momentum_indicators': [
                'roc_', 'momentum_', 'ultosc', 'trix', 'price_osc'
            ],
            'statistical_features': [
                'rolling_', 'zscore_', 'skewness_', 'kurtosis_', 'percentile_', 
                'autocorr_', 'trend_', 'fib_', 'dist_to_fib'
            ]
        }
