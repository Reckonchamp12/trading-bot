import logging
import time
from datetime import datetime, timedelta
from threading import Thread
import schedule
from app import app, db
from models import User, Portfolio, TradingSignal, ModelPerformance, MarketData
from ml_engine import MLEngine
from trading_signals import SignalGenerator
from market_data import MarketDataProvider
from risk_manager import RiskManager
import json


class TradingScheduler:
    """
    Scheduler for automated trading tasks:
    - Market data updates
    - Signal generation
    - Model retraining
    - Portfolio monitoring
    - Risk management alerts
    """

    def __init__(self):
        self.ml_engine = MLEngine()
        self.signal_generator = SignalGenerator()
        self.market_data = MarketDataProvider()
        self.risk_manager = RiskManager()
        self.is_running = False
        self.scheduler_thread = None

        # Trading symbols to monitor
        self.monitored_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'META', 'NVDA', 'JPM', 'JNJ', 'V'
        ]

        # Setup the scheduled tasks
        self._setup_schedules()

    def _setup_schedules(self) -> None:
        """Setup all scheduled tasks"""
        try:
            # Market data updates every hour at 5 minutes past
            schedule.every().hour.at(":05").do(self._update_market_data)

            # Signal generation every 30 minutes
            schedule.every(30).minutes.do(self._generate_signals)

            # Model performance monitoring every 4 hours
            schedule.every(4).hours.do(self._monitor_model_performance)

            # Portfolio risk monitoring every hour at 15 minutes past
            schedule.every().hour.at(":15").do(self._monitor_portfolio_risk)

            # Model retraining daily at 18:00 (6 PM)
            schedule.every().day.at("18:00").do(self._retrain_models)

            # Database cleanup daily at midnight
            schedule.every().day.at("00:00").do(self._cleanup_old_data)

            # Weekly performance report every Sunday at 8 AM
            schedule.every().sunday.at("08:00").do(self._generate_weekly_report)

            logging.info("Scheduler tasks configured successfully")

        except Exception as e:
            logging.error(f"Schedule setup error: {e}")

    def start(self) -> None:
        """Start the scheduler in a separate daemon thread"""
        try:
            if self.is_running:
                logging.warning("Scheduler is already running")
                return

            self.is_running = True
            self.scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()

            logging.info("Trading scheduler started")

        except Exception as e:
            logging.error(f"Scheduler start error: {e}")
            self.is_running = False

    def stop(self) -> None:
        """Stop the scheduler"""
        try:
            self.is_running = False
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)

            logging.info("Trading scheduler stopped")

        except Exception as e:
            logging.error(f"Scheduler stop error: {e}")

    def _run_scheduler(self) -> None:
        """Main loop running scheduled tasks"""
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except Exception as e:
            logging.error(f"Scheduler loop error: {e}")
            self.is_running = False

    def _update_market_data(self) -> None:
        """Update market data for monitored symbols"""
        try:
            with app.app_context():
                logging.info("Starting market data update")

                current_time = datetime.now()
                if not self._is_market_hours(current_time):
                    logging.info("Market is closed, skipping data update")
                    return

                end_date = datetime.now()
                start_date = end_date - timedelta(hours=2)  # last 2 hours

                updated_symbols = 0

                for symbol in self.monitored_symbols:
                    try:
                        df = self.market_data.get_ohlcv_data(
                            symbol, start_date, end_date, '1h'
                        )

                        if df.empty:
                            continue

                        for timestamp, row in df.iterrows():
                            # Check if data exists
                            exists = MarketData.query.filter_by(
                                symbol=symbol,
                                timestamp=timestamp,
                                timeframe='1h'
                            ).first()

                            if not exists:
                                market_data = MarketData(
                                    symbol=symbol,
                                    timestamp=timestamp,
                                    open_price=row['open'],
                                    high_price=row['high'],
                                    low_price=row['low'],
                                    close_price=row['close'],
                                    volume=row['volume'],
                                    timeframe='1h'
                                )
                                db.session.add(market_data)

                        updated_symbols += 1

                    except Exception as e:
                        logging.error(f"Market data update error for {symbol}: {e}")
                        continue

                db.session.commit()
                logging.info(f"Market data updated for {updated_symbols} symbols")

        except Exception as e:
            logging.error(f"Market data update task error: {e}")
            if db.session.is_active:
                db.session.rollback()

    def _generate_signals(self) -> None:
        """Generate trading signals for monitored symbols"""
        try:
            with app.app_context():
                logging.info("Starting signal generation")

                current_time = datetime.now()
                if not self._is_market_hours(current_time):
                    logging.info("Market is closed, skipping signal generation")
                    return

                generated_count = 0

                for symbol in self.monitored_symbols:
                    try:
                        signal = self.signal_generator.generate_signal(
                            symbol, strategy='ensemble', timeframe='1h'
                        )

                        if signal and signal.get('confidence', 0) > 0.6:
                            trading_signal = TradingSignal(
                                symbol=symbol,
                                signal_type=signal['type'],
                                confidence=signal['confidence'],
                                strategy_name=signal['strategy'],
                                target_price=signal.get('target_price'),
                                stop_loss_price=signal.get('stop_loss'),
                                timeframe=signal['timeframe'],
                                features_json=json.dumps(signal.get('features', {})),
                                model_version=signal.get('model_version'),
                                expires_at=datetime.now() + timedelta(hours=4)
                            )

                            db.session.add(trading_signal)
                            generated_count += 1

                            logging.info(
                                f"Generated {signal['type']} signal for {symbol} "
                                f"with confidence {signal['confidence']:.3f}"
                            )

                    except Exception as e:
                        logging.error(f"Signal generation error for {symbol}: {e}")
                        continue

                db.session.commit()
                logging.info(f"Generated {generated_count} signals")

        except Exception as e:
            logging.error(f"Signal generation task error: {e}")
            if db.session.is_active:
                db.session.rollback()

    def _monitor_model_performance(self) -> None:
        """Monitor ML model performance and detect drift"""
        try:
            with app.app_context():
                logging.info("Starting model performance monitoring")

                drift_detected_count = 0

                for symbol in self.monitored_symbols:
                    try:
                        drift_result = self.ml_engine.model_drift_detection(
                            symbol, model_type='xgboost', window_days=7
                        )

                        if drift_result.get('drift_detected', False):
                            logging.warning(
                                f"Model drift detected for {symbol}: {drift_result.get('reason')}"
                            )
                            drift_detected_count += 1

                            # Schedule immediate retraining
                            self._schedule_model_retrain(symbol)

                    except Exception as e:
                        logging.error(f"Model monitoring error for {symbol}: {e}")
                        continue

                if drift_detected_count > 0:
                    logging.warning(f"Model drift detected for {drift_detected_count} symbols")
                else:
                    logging.info("No model drift detected")

        except Exception as e:
            logging.error(f"Model performance monitoring error: {e}")

    def _monitor_portfolio_risk(self) -> None:
        """Monitor portfolio risk across all users"""
        try:
            with app.app_context():
                logging.info("Starting portfolio risk monitoring")

                high_risk_portfolios = 0
                portfolios = Portfolio.query.filter_by(is_paper_trading=True).all()

                for portfolio in portfolios:
                    try:
                        risk_metrics = self.risk_manager.monitor_portfolio_risk(portfolio)

                        if risk_metrics.get('risk_level') == 'HIGH':
                            high_risk_portfolios += 1
                            logging.warning(
                                f"High risk portfolio detected: User {portfolio.user_id}, "
                                f"Risk score: {risk_metrics.get('risk_score', 0)}"
                            )
                            self._send_risk_alert(portfolio, risk_metrics)

                    except Exception as e:
                        logging.error(f"Risk monitoring error for portfolio {portfolio.id}: {e}")
                        continue

                logging.info(f"Risk monitoring completed. {high_risk_portfolios} high-risk portfolios found")

        except Exception as e:
            logging.error(f"Portfolio risk monitoring error: {e}")

    def _retrain_models(self) -> None:
        """Retrain ML models with latest data"""
        try:
            with app.app_context():
                logging.info("Starting model retraining")

                retrained_count = 0
                for symbol in self.monitored_symbols:
                    try:
                        success = self.ml_engine.train_model(
                            symbol, model_type='xgboost', retrain=True
                        )

                        if success:
                            retrained_count += 1
                            logging.info(f"Successfully retrained model for {symbol}")

                            self.ml_engine.save_model(symbol, model_type='xgboost')

                    except Exception as e:
                        logging.error(f"Model retraining error for {symbol}: {e}")
                        continue

                logging.info(f"Model retraining completed. {retrained_count} models retrained")

        except Exception as e:
            logging.error(f"Model retraining task error: {e}")

    def _cleanup_old_data(self) -> None:
        """Clean up old data from database"""
        try:
            with app.app_context():
                logging.info("Starting database cleanup")

                cutoff_date = datetime.now() - timedelta(days=90)
                old_market_data = MarketData.query.filter(
                    MarketData.timestamp < cutoff_date
                ).delete(synchronize_session=False)

                signal_cutoff = datetime.now() - timedelta(days=30)
                old_signals = TradingSignal.query.filter(
                    TradingSignal.created_at < signal_cutoff
                ).delete(synchronize_session=False)

                expired_signals = TradingSignal.query.filter(
                    TradingSignal.expires_at < datetime.now()
                ).delete(synchronize_session=False)

                db.session.commit()

                logging.info(
                    f"Database cleanup completed. Removed {old_market_data} "
                    f"market data records, {old_signals} old signals, "
                    f"{expired_signals} expired signals"
                )

        except Exception as e:
            logging.error(f"Database cleanup error: {e}")
            if db.session.is_active:
                db.session.rollback()

    def _generate_weekly_report(self) -> None:
        """Generate weekly performance report"""
        try:
            with app.app_context():
                logging.info("Generating weekly performance report")

                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)

                portfolios = Portfolio.query.filter_by(is_paper_trading=True).all()

                report_data = {
                    'period': f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}",
                    'total_portfolios': len(portfolios),
                    'portfolio_performance': [],
                    'signal_statistics': {},
                    'model_performance': {}  # Placeholder if needed
                }

                for portfolio in portfolios:
                    try:
                        weekly_return = (
                            (portfolio.current_capital - portfolio.initial_capital)
                            / portfolio.initial_capital * 100
                        )
                        report_data['portfolio_performance'].append({
                            'portfolio_id': portfolio.id,
                            'user_id': portfolio.user_id,
                            'weekly_return': weekly_return,
                            'current_capital': portfolio.current_capital
                        })

                    except Exception as e:
                        logging.error(f"Weekly report error for portfolio {portfolio.id}: {e}")
                        continue

                signals_count = TradingSignal.query.filter(
                    TradingSignal.created_at >= start_date
                ).count()

                report_data['signal_statistics'] = {
                    'total_signals': signals_count,
                    'signals_per_day': signals_count / 7
                }

                # Save or send the report here (currently just logs)
                logging.info(f"Weekly report generated: {json.dumps(report_data, indent=2, default=str)}")

        except Exception as e:
            logging.error(f"Weekly report generation error: {e}")

    def _schedule_model_retrain(self, symbol: str) -> None:
        """Schedule immediate model retraining for a specific symbol"""
        try:
            logging.info(f"Scheduling immediate retraining for {symbol}")
            retrain_thread = Thread(
                target=self._retrain_single_model,
                args=(symbol,),
                daemon=True
            )
            retrain_thread.start()

        except Exception as e:
            logging.error(f"Model retrain scheduling error: {e}")

    def _retrain_single_model(self, symbol: str) -> None:
        """Retrain a single model"""
        try:
            with app.app_context():
                logging.info(f"Retraining model for {symbol}")
                success = self.ml_engine.train_model(symbol, model_type='xgboost', retrain=True)

                if success:
                    self.ml_engine.save_model(symbol, model_type='xgboost')
                    logging.info(f"Model retrained successfully for {symbol}")
                else:
                    logging.error(f"Model retraining failed for {symbol}")

        except Exception as e:
            logging.error(f"Single model retrain error for {symbol}: {e}")

    def _send_risk_alert(self, portfolio: Portfolio, risk_metrics: dict) -> None:
        """Send risk alert (placeholder for email/notification system)"""
        try:
            alert_message = (
                f"High risk detected in portfolio {portfolio.id}\n"
                f"Risk level: {risk_metrics.get('risk_level')}\n"
                f"Risk score: {risk_metrics.get('risk_score')}\n"
                f"Recommendations: {risk_metrics.get('recommendations', [])}"
            )

            logging.warning(f"RISK ALERT: {alert_message}")

            # Implement email/push notification here if needed
            # e.g. send_email(portfolio.user.email, "Portfolio Risk Alert", alert_message)

        except Exception as e:
            logging.error(f"Risk alert error: {e}")

    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if current time is during market hours (EST 9 AM - 4 PM Mon-Fri)"""
        try:
            weekday = timestamp.weekday()
            hour = timestamp.hour

            # Weekend check
            if weekday >= 5:
                return False

            # Market hours check
            if hour < 9 or hour >= 16:
                return False

            return True

        except Exception as e:
            logging.error(f"Market hours check error: {e}")
            return False  # Fail safe

    def run_manual_task(self, task_name: str) -> bool:
        """Run a specific task manually"""
        try:
            task_map = {
                'update_data': self._update_market_data,
                'generate_signals': self._generate_signals,
                'monitor_models': self._monitor_model_performance,
                'monitor_risk': self._monitor_portfolio_risk,
                'retrain_models': self._retrain_models,
                'cleanup_data': self._cleanup_old_data,
                'weekly_report': self._generate_weekly_report
            }

            if task_name in task_map:
                logging.info(f"Running manual task: {task_name}")
                task_map[task_name]()
                logging.info(f"Manual task completed: {task_name}")
                return True
            else:
                logging.error(f"Unknown task: {task_name}")
                return False

        except Exception as e:
            logging.error(f"Manual task execution error: {e}")
            return False


# Global scheduler instance
trading_scheduler = TradingScheduler()


def start_scheduler() -> None:
    """Start the trading scheduler"""
    trading_scheduler.start()


def stop_scheduler() -> None:
    """Stop the trading scheduler"""
    trading_scheduler.stop()
