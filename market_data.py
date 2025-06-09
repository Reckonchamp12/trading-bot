import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
import time
import os
from app import db
from models import MarketData

class MarketDataProvider:
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls per minute
        self.last_request_time = 0
        
        # Mock data configuration for testing
        self.use_mock_data = True  # Set to False when API key is available
        
    def get_ohlcv_data(self, symbol, start_date, end_date, timeframe='1h'):
        """Get OHLCV data for given symbol and date range"""
        try:
            # Check database first
            cached_data = self._get_cached_data(symbol, start_date, end_date, timeframe)
            if not cached_data.empty:
                logging.info(f"Using cached data for {symbol}")
                return cached_data
            
            # Fetch from API or generate mock data
            if self.use_mock_data or self.api_key == "demo":
                data = self._generate_mock_data(symbol, start_date, end_date, timeframe)
            else:
                data = self._fetch_from_api(symbol, start_date, end_date, timeframe)
            
            if not data.empty:
                # Cache the data
                self._cache_data(symbol, data, timeframe)
                logging.info(f"Fetched and cached {len(data)} records for {symbol}")
            
            return data
            
        except Exception as e:
            logging.error(f"Market data error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_cached_data(self, symbol, start_date, end_date, timeframe):
        """Get cached data from database"""
        try:
            query = MarketData.query.filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe,
                MarketData.timestamp >= start_date,
                MarketData.timestamp <= end_date
            ).order_by(MarketData.timestamp)
            
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for record in results:
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open_price,
                    'high': record.high_price,
                    'low': record.low_price,
                    'close': record.close_price,
                    'volume': record.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Cache retrieval error: {e}")
            return pd.DataFrame()
    
    def _cache_data(self, symbol, df, timeframe):
        """Cache data to database"""
        try:
            for timestamp, row in df.iterrows():
                # Check if record already exists
                existing = MarketData.query.filter_by(
                    symbol=symbol,
                    timestamp=timestamp,
                    timeframe=timeframe
                ).first()
                
                if not existing:
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=row['open'],
                        high_price=row['high'],
                        low_price=row['low'],
                        close_price=row['close'],
                        volume=row['volume'],
                        timeframe=timeframe
                    )
                    db.session.add(market_data)
            
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            logging.error(f"Cache storage error: {e}")
    
    def _fetch_from_api(self, symbol, start_date, end_date, timeframe):
        """Fetch data from Alpha Vantage API"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
            
            # Map timeframe to Alpha Vantage function
            if timeframe == '1d':
                function = 'TIME_SERIES_DAILY'
                interval = None
            elif timeframe in ['1h', '4h']:
                function = 'TIME_SERIES_INTRADAY'
                interval = '60min'  # Alpha Vantage doesn't support 4h directly
            else:
                function = 'TIME_SERIES_INTRADAY'
                interval = '5min'
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            if interval:
                params['interval'] = interval
            
            response = requests.get(self.base_url, params=params)
            self.last_request_time = time.time()
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}")
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise Exception(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise Exception(f"API Rate Limit: {data['Note']}")
            
            # Parse response
            if function == 'TIME_SERIES_DAILY':
                time_series_key = 'Time Series (Daily)'
            else:
                time_series_key = f'Time Series ({interval})'
            
            if time_series_key not in data:
                raise Exception("Invalid API response format")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp_str, values in time_series.items():
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S' if interval else '%Y-%m-%d')
                
                # Filter by date range
                if start_date <= timestamp <= end_date:
                    df_data.append({
                        'timestamp': timestamp,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': float(values['5. volume'])
                    })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"API fetch error: {e}")
            return pd.DataFrame()
    
    def _generate_mock_data(self, symbol, start_date, end_date, timeframe):
        """Generate realistic mock OHLCV data for testing"""
        try:
            # Determine frequency based on timeframe
            freq_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            freq = freq_map.get(timeframe, '1H')
            
            # Generate time index
            timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            if len(timestamps) == 0:
                return pd.DataFrame()
            
            # Generate realistic price data using random walk with trend
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed based on symbol
            
            # Starting price based on symbol
            base_price = self._get_base_price(symbol)
            
            # Generate price movements
            n_periods = len(timestamps)
            
            # Add slight trend component
            trend = np.linspace(0, 0.1, n_periods)  # 10% trend over period
            
            # Random walk component
            returns = np.random.normal(0, 0.02, n_periods)  # 2% volatility
            returns[0] = 0  # First return is zero
            
            # Add some autocorrelation for realism
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            # Calculate cumulative prices
            price_multipliers = np.exp(np.cumsum(returns) + trend)
            close_prices = base_price * price_multipliers
            
            # Generate OHLV data
            data = []
            for i, (timestamp, close) in enumerate(zip(timestamps, close_prices)):
                # Generate intraday volatility
                volatility = np.random.uniform(0.005, 0.02)  # 0.5% to 2%
                
                # Open price (previous close + gap)
                if i == 0:
                    open_price = close
                else:
                    gap = np.random.normal(0, 0.001)  # Small overnight gap
                    open_price = close_prices[i-1] * (1 + gap)
                
                # High and low prices
                high_multiplier = 1 + np.random.uniform(0, volatility)
                low_multiplier = 1 - np.random.uniform(0, volatility)
                
                high = max(open_price, close) * high_multiplier
                low = min(open_price, close) * low_multiplier
                
                # Ensure OHLC consistency
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # Volume (correlated with price movement)
                price_change = abs(close - open_price) / open_price
                base_volume = np.random.uniform(100000, 1000000)
                volume = base_volume * (1 + price_change * 5)  # Higher volume on big moves
                
                data.append({
                    'timestamp': timestamp,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': round(volume)
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Mock data generation error: {e}")
            return pd.DataFrame()
    
    def _get_base_price(self, symbol):
        """Get base price for symbol (for mock data generation)"""
        price_map = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0,
            'TSLA': 800.0,import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
import os
import time
from typing import Optional
import json

class MarketDataProvider:
    """
    Market data provider for fetching OHLCV data
    Uses multiple data sources with fallbacks
    """
    
    def __init__(self):
        # API keys from environment variables
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # seconds between requests
        
        # Supported timeframes
        self.timeframe_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '1d': 'daily',
            '1w': 'weekly',
            '1M': 'monthly'
        }
        
    def get_ohlcv_data(self, symbol, start_date, end_date, timeframe='1h') -> pd.DataFrame:
        """
        Get OHLCV data for a symbol within date range
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)
            timeframe: Time interval ('1m', '5m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert dates to datetime if strings
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            logging.info(f"Fetching {timeframe} data for {symbol} from {start_date} to {end_date}")
            
            # Try different data sources in order of preference
            data_sources = [
                self._get_alpha_vantage_data,
                self._get_polygon_data,
                self._get_finnhub_data,
                self._generate_synthetic_data  # Fallback for development
            ]
            
            for source_func in data_sources:
                try:
                    df = source_func(symbol, start_date, end_date, timeframe)
                    if not df.empty:
                        # Validate and clean data
                        df = self._validate_and_clean_data(df)
                        if not df.empty:
                            logging.info(f"Successfully fetched {len(df)} records from {source_func.__name__}")
                            return df
                except Exception as e:
                    logging.warning(f"Data source {source_func.__name__} failed: {e}")
                    continue
            
            logging.error(f"All data sources failed for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logging.error(f"Market data fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_alpha_vantage_data(self, symbol, start_date, end_date, timeframe):
        """Fetch data from Alpha Vantage API"""
        try:
            if not self.alpha_vantage_key or self.alpha_vantage_key == "demo":
                raise ValueError("Alpha Vantage API key not available")
            
            # Rate limiting
            self._rate_limit('alpha_vantage')
            
            # Map timeframe to Alpha Vantage format
            av_timeframe = self.timeframe_map.get(timeframe, '60min')
            
            if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                function = 'TIME_SERIES_INTRADAY'
                interval = av_timeframe
                url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&apikey={self.alpha_vantage_key}&outputsize=full"
            else:
                function = 'TIME_SERIES_DAILY'
                url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={self.alpha_vantage_key}&outputsize=full"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
            
            # Extract time series data
            if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                time_series_key = f'Time Series ({interval})'
            else:
                time_series_key = 'Time Series (Daily)'
            
            if time_series_key not in data:
                raise ValueError("No time series data found in response")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
            
        except Exception as e:
            logging.error(f"Alpha Vantage data fetch error: {e}")
            raise
    
    def _get_polygon_data(self, symbol, start_date, end_date, timeframe):
        """Fetch data from Polygon.io API"""
        try:
            if not self.polygon_key:
                raise ValueError("Polygon API key not available")
            
            # Rate limiting
            self._rate_limit('polygon')
            
            # Convert timeframe to Polygon format
            if timeframe == '1m':
                multiplier, timespan = 1, 'minute'
            elif timeframe == '5m':
                multiplier, timespan = 5, 'minute'
            elif timeframe == '1h':
                multiplier, timespan = 1, 'hour'
            elif timeframe == '1d':
                multiplier, timespan = 1, 'day'
            else:
                raise ValueError(f"Unsupported timeframe for Polygon: {timeframe}")
            
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
            params = {
                'apikey': self.polygon_key,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'OK':
                raise ValueError(f"Polygon API error: {data.get('error', 'Unknown error')}")
            
            if 'results' not in data or not data['results']:
                raise ValueError("No data returned from Polygon API")
            
            # Convert to DataFrame
            df_data = []
            for item in data['results']:
                df_data.append({
                    'timestamp': pd.to_datetime(item['t'], unit='ms'),
                    'open': float(item['o']),
                    'high': float(item['h']),
                    'low': float(item['l']),
                    'close': float(item['c']),
                    'volume': float(item['v'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Polygon data fetch error: {e}")
            raise
    
    def _get_finnhub_data(self, symbol, start_date, end_date, timeframe):
        """Fetch data from Finnhub API"""
        try:
            if not self.finnhub_key:
                raise ValueError("Finnhub API key not available")
            
            # Rate limiting
            self._rate_limit('finnhub')
            
            # Finnhub uses different endpoint for different timeframes
            if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                # Use stock candles endpoint
                resolution_map = {
                    '1m': '1',
                    '5m': '5',
                    '15m': '15',
                    '30m': '30',
                    '1h': '60'
                }
                resolution = resolution_map[timeframe]
                
                url = "https://finnhub.io/api/v1/stock/candle"
                params = {
                    'symbol': symbol,
                    'resolution': resolution,
                    'from': int(start_date.timestamp()),
                    'to': int(end_date.timestamp()),
                    'token': self.finnhub_key
                }
            else:
                raise ValueError(f"Unsupported timeframe for Finnhub: {timeframe}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('s') != 'ok':
                raise ValueError(f"Finnhub API error: {data.get('s', 'Unknown error')}")
            
            # Convert to DataFrame
            timestamps = [pd.to_datetime(t, unit='s') for t in data['t']]
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Finnhub data fetch error: {e}")
            raise
    
    def _generate_synthetic_data(self, symbol, start_date, end_date, timeframe):
        """
        Generate synthetic data for development/testing
        This should only be used as a last resort when no real data is available
        """
        try:
            logging.warning(f"Generating synthetic data for {symbol} - USE ONLY FOR DEVELOPMENT")
            
            # Calculate time delta based on timeframe
            if timeframe == '1m':
                delta = timedelta(minutes=1)
            elif timeframe == '5m':
                delta = timedelta(minutes=5)
            elif timeframe == '15m':
                delta = timedelta(minutes=15)
            elif timeframe == '30m':
                delta = timedelta(minutes=30)
            elif timeframe == '1h':
                delta = timedelta(hours=1)
            elif timeframe == '1d':
                delta = timedelta(days=1)
            else:
                delta = timedelta(hours=1)
            
            # Generate timestamps
            timestamps = []
            current_time = start_date
            while current_time <= end_date:
                timestamps.append(current_time)
                current_time += delta
            
            if not timestamps:
                return pd.DataFrame()
            
            # Generate realistic price data using random walk
            np.random.seed(42)  # For reproducible results
            num_points = len(timestamps)
            
            # Starting price
            base_price = 100.0
            
            # Generate returns using random walk
            returns = np.random.normal(0.0001, 0.02, num_points)  # Small drift, 2% daily volatility
            prices = [base_price]
            
            for i in range(1, num_points):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 0.01))  # Ensure positive prices
            
            # Generate OHLCV data
            df_data = []
            for i, timestamp in enumerate(timestamps):
                if i == 0:
                    open_price = prices[i]
                else:
                    open_price = prices[i-1]
                
                close_price = prices[i]
                
                # Generate high/low around open/close
                price_range = abs(close_price - open_price) * 2 + close_price * 0.005
                high_price = max(open_price, close_price) + np.random.uniform(0, price_range * 0.5)
                low_price = min(open_price, close_price) - np.random.uniform(0, price_range * 0.5)
                
                # Ensure high >= max(open, close) and low <= min(open, close)
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # Generate volume (higher volume on larger moves)
                volume_base = 1000000
                volume_multiplier = 1 + abs(returns[i]) * 10
                volume = int(volume_base * volume_multiplier * np.random.uniform(0.5, 2.0))
                
                df_data.append({
                    'timestamp': timestamp,
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Synthetic data generation error: {e}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, df):
        """Validate and clean OHLCV data"""
        try:
            if df.empty:
                return df
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logging.error(f"Missing required columns. Have: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Remove rows with invalid data
            initial_len = len(df)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Remove rows with zero or negative prices
            for col in ['open', 'high', 'low', 'close']:
                df = df[df[col] > 0]
            
            # Remove rows with negative volume
            df = df[df['volume'] >= 0]
            
            # Validate OHLC relationships
            df = df[
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ]
            
            # Remove extreme outliers (price changes > 50% in one period)
            if len(df) > 1:
                price_changes = df['close'].pct_change().abs()
                df = df[price_changes <= 0.5]  # Remove 50%+ moves
            
            final_len = len(df)
            if final_len < initial_len:
                logging.info(f"Cleaned data: {initial_len} -> {final_len} rows")
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Data validation error: {e}")
            return pd.DataFrame()
    
    def _rate_limit(self, source):
        """Implement rate limiting for API calls"""
        try:
            current_time = time.time()
            last_time = self.last_request_time.get(source, 0)
            
            time_since_last = current_time - last_time
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logging.debug(f"Rate limiting {source}: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self.last_request_time[source] = time.time()
            
        except Exception as e:
            logging.error(f"Rate limiting error: {e}")
    
    def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            df = self.get_ohlcv_data(symbol, start_date, end_date, '1h')
            if df.empty:
                return None
            
            latest_data = df.iloc[-1]
            return {
                'symbol': symbol,
                'price': latest_data['close'],
                'timestamp': df.index[-1],
                'volume': latest_data['volume'],
                'change': latest_data['close'] - latest_data['open'],
                'change_pct': ((latest_data['close'] - latest_data['open']) / latest_data['open']) * 100
            }
            
        except Exception as e:
            logging.error(f"Latest price fetch error for {symbol}: {e}")
            return None
    
    def get_multiple_symbols(self, symbols, timeframe='1h', days=30):
        """Get data for multiple symbols"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            results = {}
            for symbol in symbols:
                try:
                    df = self.get_ohlcv_data(symbol, start_date, end_date, timeframe)
                    if not df.empty:
                        results[symbol] = df
                    else:
                        logging.warning(f"No data available for {symbol}")
                except Exception as e:
                    logging.error(f"Error fetching data for {symbol}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logging.error(f"Multiple symbols fetch error: {e}")
            return {}
    
    def get_symbol_info(self, symbol):
        """Get basic information about a trading symbol"""
        try:
            # This is a simplified implementation
            # In production, you would fetch from a proper symbol directory API
            
            # Common stock symbols and their info
            symbol_info = {
                'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'currency': 'USD'},
                'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'currency': 'USD'},
                'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'currency': 'USD'},
                'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Discretionary', 'currency': 'USD'},
                'TSLA': {'name': 'Tesla Inc.', 'sector': 'Consumer Discretionary', 'currency': 'USD'},
                'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology', 'currency': 'USD'},
                'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'currency': 'USD'},
                'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services', 'currency': 'USD'},
                'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare', 'currency': 'USD'},
                'V': {'name': 'Visa Inc.', 'sector': 'Financial Services', 'currency': 'USD'}
            }
            
            return symbol_info.get(symbol, {
                'name': symbol,
                'sector': 'Unknown',
                'currency': 'USD'
            })
            
        except Exception as e:
            logging.error(f"Symbol info error: {e}")
            return {'name': symbol, 'sector': 'Unknown', 'currency': 'USD'}
