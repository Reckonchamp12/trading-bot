import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from ml_engine import MLEngine
from feature_engineering import FeatureEngineer
from market_data import MarketDataProvider
import json

class SignalGenerator:
    def __init__(self):
        self.ml_engine = MLEngine()
        self.feature_engineer = FeatureEngineer()
        self.market_data = MarketDataProvider()
        self.confidence_threshold = 0.6
        self.signal_timeframes = ['1h', '4h', '1d']
        
    def generate_signal(self, symbol, strategy='ensemble', timeframe='1h'):
        """Generate trading signal for given symbol"""
        try:
            logging.info(f"Generating signal for {symbol} using {strategy} strategy")
            
            if strategy == 'ensemble':
                return self._generate_ensemble_signal(symbol, timeframe)
            elif strategy == 'ml_only':
                return self._generate_ml_signal(symbol, timeframe)
            elif strategy == 'technical_only':
                return self._generate_technical_signal(symbol, timeframe)
            elif strategy == 'multi_timeframe':
                return self._generate_multi_timeframe_signal(symbol)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logging.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def _generate_ensemble_signal(self, symbol, timeframe):
        """Generate signal using ensemble of ML and technical analysis"""
        try:
            # Get ML signal
            ml_signal = self._generate_ml_signal(symbol, timeframe)
            
            # Get technical signal
            tech_signal = self._generate_technical_signal(symbol, timeframe)
            
            if not ml_signal or not tech_signal:
                return None
            
            # Combine signals using weighted average
            ml_weight = 0.6
            tech_weight = 0.4
            
            # Convert signal types to numeric values
            signal_map = {'buy': 1, 'sell': -1, 'hold': 0}
            ml_value = signal_map.get(ml_signal['type'], 0)
            tech_value = signal_map.get(tech_signal['type'], 0)
            
            # Calculate ensemble confidence and type
            ensemble_value = (ml_value * ml_weight * ml_signal['confidence'] + 
                            tech_value * tech_weight * tech_signal['confidence'])
            
            ensemble_confidence = (ml_signal['confidence'] * ml_weight + 
                                 tech_signal['confidence'] * tech_weight)
            
            # Determine signal type
            if ensemble_value > 0.3:
                signal_type = 'buy'
            elif ensemble_value < -0.3:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Only return signal if confidence is above threshold
            if ensemble_confidence < self.confidence_threshold:
                return None
            
            return {
                'type': signal_type,
                'confidence': ensemble_confidence,
                'strategy': 'ensemble',
                'timeframe': timeframe,
                'target_price': self._calculate_target_price(symbol, signal_type, ensemble_confidence),
                'stop_loss': self._calculate_stop_loss(symbol, signal_type, ensemble_confidence),
                'components': {
                    'ml_signal': ml_signal,
                    'technical_signal': tech_signal
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Ensemble signal error: {e}")
            return None
    
    def _generate_ml_signal(self, symbol, timeframe):
        """Generate signal using ML model"""
        try:
            # Get prediction from ML engine
            prediction = self.ml_engine.predict(symbol, model_type='xgboost')
            
            if not prediction:
                return None
            
            # Convert prediction to signal
            signal_type = prediction['signal']
            confidence = prediction['confidence']
            
            # Enhance confidence based on additional factors
            enhanced_confidence = self._enhance_ml_confidence(symbol, prediction)
            
            return {
                'type': signal_type,
                'confidence': enhanced_confidence,
                'strategy': 'ml_xgboost',
                'timeframe': timeframe,
                'model_version': prediction.get('model_version'),
                'raw_prediction': prediction['prediction'],
                'features': self._get_signal_features(symbol)
            }
            
        except Exception as e:
            logging.error(f"ML signal error: {e}")
            return None
    
    def _generate_technical_signal(self, symbol, timeframe):
        """Generate signal using technical indicators"""
        try:
            # Get market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            df = self.market_data.get_ohlcv_data(symbol, start_date, end_date, timeframe)
            if df.empty:
                return None
            
            # Create features
            features_df = self.feature_engineer.create_features(df)
            
            # Get latest values
            latest = features_df.iloc[-1]
            
            # Technical analysis scoring
            scores = []
            
            # 1. RSI Signal
            rsi = latest.get('rsi_14', 50)
            if rsi < 30:
                scores.append(('buy', 0.8))  # Oversold
            elif rsi > 70:
                scores.append(('sell', 0.8))  # Overbought
            elif 40 <= rsi <= 60:
                scores.append(('hold', 0.6))  # Neutral
            
            # 2. MACD Signal
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            if macd > macd_signal and macd > 0:
                scores.append(('buy', 0.7))
            elif macd < macd_signal and macd < 0:
                scores.append(('sell', 0.7))
            else:
                scores.append(('hold', 0.5))
            
            # 3. Bollinger Bands Signal
            bb_position = latest.get('bb_position', 0.5)
            if bb_position < 0.2:
                scores.append(('buy', 0.6))  # Near lower band
            elif bb_position > 0.8:
                scores.append(('sell', 0.6))  # Near upper band
            else:
                scores.append(('hold', 0.4))
            
            # 4. Moving Average Signal
            sma_20 = latest.get('sma_20', 0)
            sma_50 = latest.get('sma_50', 0)
            close = latest.get('close', 0)
            
            if close > sma_20 > sma_50:
                scores.append(('buy', 0.7))  # Uptrend
            elif close < sma_20 < sma_50:
                scores.append(('sell', 0.7))  # Downtrend
            else:
                scores.append(('hold', 0.5))
            
            # 5. Volume Signal
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                # High volume - amplify other signals
                volume_multiplier = 1.2
            elif volume_ratio < 0.5:
                # Low volume - reduce confidence
                volume_multiplier = 0.8
            else:
                volume_multiplier = 1.0
            
            # Calculate overall signal
            buy_scores = [score for signal, score in scores if signal == 'buy']
            sell_scores = [score for signal, score in scores if signal == 'sell']
            hold_scores = [score for signal, score in scores if signal == 'hold']
            
            buy_strength = sum(buy_scores) * volume_multiplier if buy_scores else 0
            sell_strength = sum(sell_scores) * volume_multiplier if sell_scores else 0
            hold_strength = sum(hold_scores) if hold_scores else 0
            
            # Determine final signal
            max_strength = max(buy_strength, sell_strength, hold_strength)
            
            if max_strength == buy_strength and buy_strength > 0.6:
                signal_type = 'buy'
                confidence = min(buy_strength / len(scores), 1.0)
            elif max_strength == sell_strength and sell_strength > 0.6:
                signal_type = 'sell'
                confidence = min(sell_strength / len(scores), 1.0)
            else:
                signal_type = 'hold'
                confidence = 0.5
            
            return {
                'type': signal_type,
                'confidence': confidence,
                'strategy': 'technical_analysis',
                'timeframe': timeframe,
                'indicators': {
                    'rsi': rsi,
                    'macd_bullish': macd > macd_signal,
                    'bb_position': bb_position,
                    'trend': 'up' if close > sma_20 > sma_50 else 'down' if close < sma_20 < sma_50 else 'sideways',
                    'volume_ratio': volume_ratio
                },
                'score_breakdown': {
                    'buy_strength': buy_strength,
                    'sell_strength': sell_strength,
                    'hold_strength': hold_strength
                }
            }
            
        except Exception as e:
            logging.error(f"Technical signal error: {e}")
            return None
    
    def _generate_multi_timeframe_signal(self, symbol):
        """Generate signal using multiple timeframes"""
        try:
            timeframe_signals = {}
            timeframe_weights = {'1h': 0.3, '4h': 0.4, '1d': 0.3}
            
            # Get signals for different timeframes
            for tf in self.signal_timeframes:
                signal = self._generate_ensemble_signal(symbol, tf)
                if signal:
                    timeframe_signals[tf] = signal
            
            if not timeframe_signals:
                return None
            
            # Calculate weighted consensus
            signal_values = {'buy': 0, 'sell': 0, 'hold': 0}
            total_weight = 0
            
            for tf, signal in timeframe_signals.items():
                weight = timeframe_weights.get(tf, 0.33)
                signal_values[signal['type']] += weight * signal['confidence']
                total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                for signal_type in signal_values:
                    signal_values[signal_type] /= total_weight
            
            # Determine consensus signal
            max_signal = max(signal_values, key=signal_values.get)
            max_confidence = signal_values[max_signal]
            
            # Require minimum agreement across timeframes
            agreement_threshold = 0.4
            if max_confidence < agreement_threshold:
                return None
            
            return {
                'type': max_signal,
                'confidence': max_confidence,
                'strategy': 'multi_timeframe',
                'timeframe': 'consensus',
                'timeframe_signals': timeframe_signals,
                'signal_breakdown': signal_values,
                'target_price': self._calculate_target_price(symbol, max_signal, max_confidence),
                'stop_loss': self._calculate_stop_loss(symbol, max_signal, max_confidence),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Multi-timeframe signal error: {e}")
            return None
    
    def _enhance_ml_confidence(self, symbol, prediction):
        """Enhance ML confidence with additional market factors"""
        try:
            base_confidence = prediction['confidence']
            
            # Get current market data for enhancement factors
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            df = self.market_data.get_ohlcv_data(symbol, start_date, end_date, '1h')
            if df.empty:
                return base_confidence
            
            features_df = self.feature_engineer.create_features(df)
            latest = features_df.iloc[-1]
            
            # Enhancement factors
            enhancement_factor = 1.0
            
            # 1. Volume confirmation
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                enhancement_factor *= 1.1  # High volume increases confidence
            elif volume_ratio < 0.5:
                enhancement_factor *= 0.9  # Low volume decreases confidence
            
            # 2. Trend alignment
            sma_20 = latest.get('sma_20', 0)
            sma_50 = latest.get('sma_50', 0)
            close = latest.get('close', 0)
            
            if prediction['signal'] == 'buy' and close > sma_20 > sma_50:
                enhancement_factor *= 1.1  # Signal aligns with trend
            elif prediction['signal'] == 'sell' and close < sma_20 < sma_50:
                enhancement_factor *= 1.1
            elif prediction['signal'] in ['buy', 'sell']:
                enhancement_factor *= 0.9  # Signal against trend
            
            # 3. Volatility check
            volatility = latest.get('norm_volatility', 0)
            if volatility > 0.05:  # High volatility
                enhancement_factor *= 0.9  # Reduce confidence in volatile markets
            
            # 4. RSI divergence
            rsi = latest.get('rsi_14', 50)
            if prediction['signal'] == 'buy' and rsi < 70:
                enhancement_factor *= 1.05
            elif prediction['signal'] == 'sell' and rsi > 30:
                enhancement_factor *= 1.05
            
            # Apply enhancement and cap at 1.0
            enhanced_confidence = min(base_confidence * enhancement_factor, 1.0)
            
            return enhanced_confidence
            
        except Exception as e:
            logging.error(f"Confidence enhancement error: {e}")
            return prediction['confidence']
    
    def _calculate_target_price(self, symbol, signal_type, confidence):
        """Calculate target price based on signal"""
        try:
            # Get current price
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            df = self.market_data.get_ohlcv_data(symbol, start_date, end_date, '1h')
            if df.empty:
                return None
            
            current_price = df.iloc[-1]['close']
            
            # Calculate ATR for target setting
            features_df = self.feature_engineer.create_features(df)
            atr = features_df.iloc[-1].get('atr', current_price * 0.02)
            
            # Base target multiplier based on confidence
            base_multiplier = 1.5 + (confidence * 1.5)  # 1.5x to 3x ATR
            
            if signal_type == 'buy':
                target_price = current_price + (atr * base_multiplier)
            elif signal_type == 'sell':
                target_price = current_price - (atr * base_multiplier)
            else:
                target_price = current_price
            
            return round(target_price, 2)
            
        except Exception as e:
            logging.error(f"Target price calculation error: {e}")
            return None
    
    def _calculate_stop_loss(self, symbol, signal_type, confidence):
        """Calculate stop loss price based on signal"""
        try:
            # Get current price and ATR
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            df = self.market_data.get_ohlcv_data(symbol, start_date, end_date, '1h')
            if df.empty:
                return None
            
            current_price = df.iloc[-1]['close']
            
            features_df = self.feature_engineer.create_features(df)
            atr = features_df.iloc[-1].get('atr', current_price * 0.02)
            
            # Stop loss multiplier (lower confidence = tighter stop)
            stop_multiplier = 1.0 + (1.0 - confidence)  # 1x to 2x ATR
            
            if signal_type == 'buy':
                stop_loss = current_price - (atr * stop_multiplier)
            elif signal_type == 'sell':
                stop_loss = current_price + (atr * stop_multiplier)
            else:
                stop_loss = current_price
            
            return round(stop_loss, 2)
            
        except Exception as e:
            logging.error(f"Stop loss calculation error: {e}")
            return None
    
    def _get_signal_features(self, symbol):
        """Get key features used in signal generation"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            df = self.market_data.get_ohlcv_data(symbol, start_date, end_date, '1h')
            if df.empty:
                return {}
            
            features_df = self.feature_engineer.create_features(df)
            latest = features_df.iloc[-1]
            
            # Key features for signal interpretation
            key_features = {
                'rsi_14': latest.get('rsi_14'),
                'macd': latest.get('macd'),
                'bb_position': latest.get('bb_position'),
                'volume_ratio': latest.get('volume_ratio'),
                'atr_pct': latest.get('atr_pct'),
                'volatility_20': latest.get('volatility_20'),
                'trend_slope_20': latest.get('trend_slope_20'),
                'price_position': latest.get('price_position')
            }
            
            # Remove None values
            return {k: v for k, v in key_features.items() if v is not None}
            
        except Exception as e:
            logging.error(f"Signal features error: {e}")
            return {}
    
    def validate_signal(self, signal, symbol):
        """Validate signal before execution"""
        try:
            if not signal:
                return {'valid': False, 'reason': 'No signal provided'}
            
            # Check required fields
            required_fields = ['type', 'confidence', 'strategy', 'timeframe']
            for field in required_fields:
                if field not in signal:
                    return {'valid': False, 'reason': f'Missing field: {field}'}
            
            # Check signal type
            if signal['type'] not in ['buy', 'sell', 'hold']:
                return {'valid': False, 'reason': 'Invalid signal type'}
            
            # Check confidence range
            if not 0 <= signal['confidence'] <= 1:
                return {'valid': False, 'reason': 'Confidence out of range'}
            
            # Check minimum confidence threshold
            if signal['confidence'] < self.confidence_threshold:
                return {'valid': False, 'reason': 'Confidence below threshold'}
            
            # Market hours check (if applicable)
            current_time = datetime.now()
            if self._is_market_closed(current_time):
                return {'valid': False, 'reason': 'Market is closed'}
            
            return {'valid': True, 'reason': 'Signal validated'}
            
        except Exception as e:
            logging.error(f"Signal validation error: {e}")
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}
    
    def _is_market_closed(self, timestamp):
        """Check if market is closed (simplified check)"""
        try:
            # For crypto markets, always open
            # For stock markets, check business hours
            # This is a simplified implementation
            weekday = timestamp.weekday()
            hour = timestamp.hour
            
            # Weekend check (Saturday=5, Sunday=6)
            if weekday >= 5:
                return True
            
            # Business hours check (9 AM to 4 PM EST, simplified)
            if hour < 9 or hour >= 16:
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Market hours check error: {e}")
            return False  # Assume market is open if check fails
