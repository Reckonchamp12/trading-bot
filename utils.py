"""
Walford Capitals Trading Bot - Utility Functions
Common utility functions used across the application
"""

import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from functools import wraps
import time
import re


def setup_logging(level=logging.INFO):
    """Setup logging configuration for the application"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_bot.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)


def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = [
        'DATABASE_URL',
        'SESSION_SECRET'
    ]
    
    optional_vars = [
        'ALPHA_VANTAGE_API_KEY',
        'POLYGON_API_KEY',
        'FINNHUB_API_KEY'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_required)}")
    
    if missing_optional:
        logging.warning(f"Missing optional environment variables: {', '.join(missing_optional)}")
    
    return True


def format_currency(amount: Union[float, int, Decimal], currency: str = 'USD') -> str:
    """Format amount as currency string"""
    try:
        if amount is None:
            return f"${0:.2f}"
        
        amount = float(amount)
        
        if currency == 'USD':
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    except (ValueError, TypeError):
        return f"${0:.2f}"


def format_percentage(value: Union[float, int], decimals: int = 2) -> str:
    """Format value as percentage string"""
    try:
        if value is None:
            return f"{0:.{decimals}f}%"
        
        value = float(value) * 100
        return f"{value:.{decimals}f}%"
    except (ValueError, TypeError):
        return f"{0:.{decimals}f}%"


def format_number(value: Union[float, int], decimals: int = 2) -> str:
    """Format number with proper decimal places and commas"""
    try:
        if value is None:
            return f"{0:.{decimals}f}"
        
        value = float(value)
        return f"{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return f"{0:.{decimals}f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with default fallback"""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int with default fallback"""
    try:
        if value is None or value == '':
            return default
        return int(float(value))  # Handle string floats like "5.0"
    except (ValueError, TypeError):
        return default


def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation: 1-5 uppercase letters
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, symbol.upper()))


def validate_email(email: str) -> bool:
    """Validate email address format"""
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def generate_api_key(length: int = 32) -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(length)


def hash_password(password: str) -> str:
    """Hash password with salt using SHA-256"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        salt, password_hash = hashed.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except ValueError:
        return False


def calculate_portfolio_metrics(trades: List[Dict]) -> Dict[str, float]:
    """Calculate portfolio performance metrics from trades"""
    if not trades:
        return {
            'total_return': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    total_pnl = sum(trade.get('pnl', 0) for trade in trades)
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    profit_factor = abs(avg_win / avgimport os
import json
import logging
import hashlib
import hmac
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from functools import wraps
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self):
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables"""
        self.config = {
            'database_url': os.getenv('DATABASE_URL'),
            'session_secret': os.getenv('SESSION_SECRET'),
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
            'polygon_key': os.getenv('POLYGON_API_KEY', ''),
            'finnhub_key': os.getenv('FINNHUB_API_KEY', ''),
            'debug_mode': os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO').upper(),
            'max_positions': int(os.getenv('MAX_POSITIONS', '10')),
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.05')),
            'commission_rate': float(os.getenv('COMMISSION_RATE', '0.001')),
            'risk_free_rate': float(os.getenv('RISK_FREE_RATE', '0.02')),
            'model_retrain_frequency': int(os.getenv('MODEL_RETRAIN_FREQ', '24')),  # hours
            'signal_expiry_hours': int(os.getenv('SIGNAL_EXPIRY_HOURS', '4')),
            'market_data_cache_ttl': int(os.getenv('MARKET_DATA_CACHE_TTL', '300')),  # seconds
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
    
    def validate_required_config(self) -> List[str]:
        """Validate that required configuration is present"""
        required_keys = ['database_url', 'session_secret']
        missing = []
        
        for key in required_keys:
            if not self.config.get(key):
                missing.append(key)
        
        return missing

class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic symbol validation (alphanumeric, 1-5 characters)
        pattern = r'^[A-Z]{1,5}$'
        return bool(re.match(pattern, symbol.upper()))
    
    @staticmethod
    def validate_price(price: Union[int, float, str]) -> bool:
        """Validate price value"""
        try:
            price_val = float(price)
            return price_val > 0 and price_val < 1000000  # Reasonable price range
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_quantity(quantity: Union[int, float, str]) -> bool:
        """Validate quantity value"""
        try:
            qty_val = float(quantity)
            return qty_val > 0 and qty_val < 1000000  # Reasonable quantity range
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
        """Validate date range"""
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            return False
        
        # Start date should be before end date
        if start_date >= end_date:
            return False
        
        # Date range should not be too large (max 5 years)
        max_range = timedelta(days=365 * 5)
        if (end_date - start_date) > max_range:
            return False
        
        # Start date should not be too far in the past (max 10 years)
        earliest_date = datetime.now() - timedelta(days=365 * 10)
        if start_date < earliest_date:
            return False
        
        # End date should not be in the future
        if end_date > datetime.now():
            return False
        
        return True
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email or not isinstance(email, str):
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, bool]:
        """Validate password strength"""
        if not password or not isinstance(password, str):
            return {'valid': False, 'length': False, 'lowercase': False, 
                   'uppercase': False, 'digit': False, 'special': False}
        
        checks = {
            'length': len(password) >= 8,
            'lowercase': bool(re.search(r'[a-z]', password)),
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'digit': bool(re.search(r'\d', password)),
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        }
        
        checks['valid'] = all(checks.values())
        return checks

class SecurityUtils:
    """Security-related utilities"""
    
    @staticmethod
    def generate_api_signature(data: str, secret: str) -> str:
        """Generate HMAC signature for API requests"""
        return hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def verify_api_signature(data: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = SecurityUtils.generate_api_signature(data, secret)
        return hmac.compare_digest(expected_signature, signature)
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Sanitize user input to prevent XSS"""
        if not input_str:
            return ''
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', 'javascript:', 'data:', 'vbscript:']
        sanitized = input_str
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = os.urandom(32)  # 32 bytes salt
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + password_hash.hex()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt = bytes.fromhex(hashed[:64])  # First 32 bytes (64 hex chars)
            stored_hash = bytes.fromhex(hashed[64:])  # Remaining bytes
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return hmac.compare_digest(stored_hash, password_hash)
        except (ValueError, TypeError):
            return False

class MathUtils:
    """Mathematical and financial calculation utilities"""
    
    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate percentage returns from price series"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return returns
    
    @staticmethod
    def calculate_volatility(returns: List[float], annualized: bool = True) -> float:
        """Calculate volatility from returns"""
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns, ddof=1)
        
        if annualized:
            # Assume daily returns, annualize with sqrt(252)
            volatility *= np.sqrt(252)
        
        return float(volatility)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)
        
        if volatility == 0:
            return 0.0
        
        # Annualize assuming daily returns
        annualized_return = mean_return * 252
        annualized_volatility = volatility * np.sqrt(252)
        
        excess_return = annualized_return - risk_free_rate
        return float(excess_return / annualized_volatility)
    
    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns, ddof=1)
        
        if downside_deviation == 0:
            return 0.0
        
        # Annualize
        annualized_return = mean_return * 252
        annualized_downside_dev = downside_deviation * np.sqrt(252)
        
        excess_return = annualized_return - risk_free_rate
        return float(excess_return / annualized_downside_dev)
    
    @staticmethod
    def calculate_max_drawdown(prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(MathUtils.calculate_returns(prices)))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        if len(returns) < 2:
            return 0.0
        
        return float(np.percentile(returns, confidence_level * 100))
    
    @staticmethod
    def calculate_correlation(series1: List[float], series2: List[float]) -> float:
        """Calculate correlation between two series"""
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
        
        correlation_matrix = np.corrcoef(series1, series2)
        correlation = correlation_matrix[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def round_to_tick_size(price: float, tick_size: float = 0.01) -> float:
        """Round price to nearest tick size"""
        return float(Decimal(str(price)).quantize(Decimal(str(tick_size)), rounding=ROUND_HALF_UP))

class DateTimeUtils:
    """Date and time utilities"""
    
    @staticmethod
    def is_market_hours(dt: datetime = None, timezone: str = 'US/Eastern') -> bool:
        """Check if given datetime is during market hours"""
        if dt is None:
            dt = datetime.now()
        
        # Convert to market timezone if needed
        # For simplicity, assume input is already in correct timezone
        
        # Check if weekday (0=Monday, 6=Sunday)
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if within trading hours (9:30 AM - 4:00 PM ET)
        market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= dt <= market_close
    
    @staticmethod
    def get_market_open_close(date: datetime) -> tuple:
        """Get market open and close times for given date"""
        market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = date.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open, market_close
    
    @staticmethod
    def get_business_days(start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of business days between start and end dates"""
        business_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                business_days.append(current_date)
            current_date += timedelta(days=1)
        
        return business_days
    
    @staticmethod
    def format_timestamp(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Format datetime to string"""
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_timestamp(timestamp_str: str, format_str: str = '%Y-%m-%d %H:%M:%S') -> datetime:
        """Parse string to datetime"""
        return datetime.strptime(timestamp_str, format_str)

class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def clean_numeric_data(data: List[Union[int, float, str]], remove_outliers: bool = False) -> List[float]:
        """Clean and convert numeric data"""
        cleaned = []
        
        for value in data:
            try:
                num_val = float(value)
                if not np.isnan(num_val) and not np.isinf(num_val):
                    cleaned.append(num_val)
            except (ValueError, TypeError):
                continue
        
        if remove_outliers and len(cleaned) > 4:
            # Remove outliers using IQR method
            q1 = np.percentile(cleaned, 25)
            q3 = np.percentile(cleaned, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            cleaned = [x for x in cleaned if lower_bound <= x <= upper_bound]
        
        return cleaned
    
    @staticmethod
    def interpolate_missing_values(series: List[float], method: str = 'linear') -> List[float]:
        """Interpolate missing values in a series"""
        if not series:
            return []
        
        # Convert to pandas Series for interpolation
        s = pd.Series(series)
        
        if method == 'linear':
            interpolated = s.interpolate(method='linear')
        elif method == 'forward':
            interpolated = s.fillna(method='ffill')
        elif method == 'backward':
            interpolated = s.fillna(method='bfill')
        else:
            interpolated = s.fillna(0)  # Fill with zeros as fallback
        
        return interpolated.tolist()
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """Normalize data using specified method"""
        if not data:
            return []
        
        data_array = np.array(data)
        
        if method == 'minmax':
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            if max_val == min_val:
                return [0.5] * len(data)  # All values same, return middle value
            normalized = (data_array - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = np.mean(data_array)
            std_val = np.std(data_array)
            if std_val == 0:
                return [0.0] * len(data)  # No variance, return zeros
            normalized = (data_array - mean_val) / std_val
        
        else:
            return data  # Unknown method, return original
        
        return normalized.tolist()
    
    @staticmethod
    def calculate_moving_average(data: List[float], window: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return []
        
        moving_averages = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            moving_averages.append(avg)
        
        return moving_averages
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[int]:
        """Detect outliers and return their indices"""
        if len(data) < 4:
            return []
        
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(data):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val > 0:
                for i, value in enumerate(data):
                    z_score = abs((value - mean_val) / std_val)
                    if z_score > threshold:
                        outlier_indices.append(i)
        
        return outlier_indices

class LoggingUtils:
    """Logging utilities"""
    
    @staticmethod
    def setup_logging(level: str = 'INFO', log_file: str = None) -> None:
        """Setup application logging"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Setup file handler if specified
        handlers = [console_handler]
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @staticmethod
    def log_trade_execution(symbol: str, side: str, quantity: float, price: float, 
                           user_id: int = None, strategy: str = None) -> None:
        """Log trade execution details"""
        logger = logging.getLogger('trade_execution')
        log_message = f"Trade executed - Symbol: {symbol}, Side: {side}, " \
                     f"Quantity: {quantity}, Price: {price}"
        
        if user_id:
            log_message += f", User: {user_id}"
        if strategy:
            log_message += f", Strategy: {strategy}"
        
        logger.info(log_message)
    
    @staticmethod
    def log_signal_generation(symbol: str, signal_type: str, confidence: float, 
                             strategy: str = None) -> None:
        """Log signal generation details"""
        logger = logging.getLogger('signal_generation')
        log_message = f"Signal generated - Symbol: {symbol}, Type: {signal_type}, " \
                     f"Confidence: {confidence:.3f}"
        
        if strategy:
            log_message += f", Strategy: {strategy}"
        
        logger.info(log_message)
    
    @staticmethod
    def log_error_with_context(error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error with additional context"""
        logger = logging.getLogger('error')
        error_message = f"Error occurred: {str(error)}"
        
        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
            error_message += f" | Context: {context_str}"
        
        logger.error(error_message, exc_info=True)

class CacheUtils:
    """Simple in-memory caching utilities"""
    
    _cache = {}
    _cache_timestamps = {}
    
    @classmethod
    def get(cls, key: str, ttl: int = 300) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in cls._cache:
            return None
        
        timestamp = cls._cache_timestamps.get(key, 0)
        if time.time() - timestamp > ttl:
            cls.delete(key)
            return None
        
        return cls._cache[key]
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set value in cache"""
        cls._cache[key] = value
        cls._cache_timestamps[key] = time.time()
    
    @classmethod
    def delete(cls, key: str) -> None:
        """Delete value from cache"""
        cls._cache.pop(key, None)
        cls._cache_timestamps.pop(key, None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all cache"""
        cls._cache.clear()
        cls._cache_timestamps.clear()
    
    @classmethod
    def cleanup_expired(cls, ttl: int = 300) -> None:
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in cls._cache_timestamps.items():
            if current_time - timestamp > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            cls.delete(key)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                   f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
            
            # All retries failed
            logging.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.debug(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper

# Global configuration instance
config = ConfigManager()

# Export commonly used functions
__all__ = [
    'ConfigManager', 'ValidationUtils', 'SecurityUtils', 'MathUtils',
    'DateTimeUtils', 'DataUtils', 'LoggingUtils', 'CacheUtils',
    'retry_on_failure', 'timing_decorator', 'config'
]
