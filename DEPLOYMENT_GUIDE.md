# 7-Day Paper Trading Deployment Guide

This guide will help you deploy the Bayesian Crypto Trading Bot for 7 days of paper trading with live data from Binance and notifications via Telegram.

## Prerequisites Setup

### 1. Binance API Setup

#### Step 1: Create Binance Account
1. Go to [Binance.com](https://www.binance.com)
2. Sign up or log in to your account
3. Complete identity verification (required for API access)

#### Step 2: Create API Key
1. Go to **Account** → **API Management**
2. Click **Create API** 
3. Choose **System generated**
4. Enter a label like "Crypto Trading Bot"
5. Complete 2FA verification
6. **IMPORTANT**: Copy your API Key and Secret Key immediately
7. **Security Settings**:
   - ✅ Enable **"Enable Reading"**
   - ❌ Disable **"Enable Spot & Margin Trading"** (we're paper trading)
   - ❌ Disable **"Enable Futures"**
   - ❌ Disable **"Enable Withdrawals"**
8. **IP Access Restriction**: Add your server/computer IP for security

### 2. Telegram Bot Setup

#### Step 1: Create Telegram Bot
1. Open Telegram and search for **@BotFather**
2. Start a chat and send `/newbot`
3. Choose a name for your bot (e.g., "My Crypto Trading Bot")
4. Choose a username ending in "bot" (e.g., "mycrypto_trading_bot")
5. **Save the Bot Token** (format: `123456789:ABCdefGHIjklMNOpqrSTUvwxyz`)

#### Step 2: Get Your Chat ID
1. Send a message to your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Look for `"chat":{"id":YOUR_CHAT_ID}` in the response
4. **Save your Chat ID** (will be a number like `123456789`)

### 3. Install Required Dependencies

```bash
# Install Python dependencies
pip install python-telegram-bot==20.7
pip install ccxt  # For Binance API
pip install pandas numpy scipy scikit-learn
pip install pydantic PyYAML
pip install psutil  # For system monitoring

# For Apple Silicon (M1/M2/M3 Macs) - TensorFlow
pip install tensorflow-macos tensorflow-metal

# For Intel/AMD systems
pip install tensorflow
```

## Configuration Setup

### 1. Create Environment Variables File

Create `.env` file in your project root:

```bash
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram Configuration  
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Trading Configuration
PAPER_TRADING=true
MAX_POSITION_SIZE=0.10
CONFIDENCE_THRESHOLD=0.40

# Data Configuration
HISTORICAL_DAYS=180
UPDATE_FREQUENCY_MINUTES=60

# Logging
LOG_LEVEL=INFO
```

### 2. Create Configuration File

Create `config/deployment.yaml`:

```yaml
trading:
  paper_trading: true
  max_position_size: 0.10
  confidence_threshold: 0.40
  stop_loss_multiplier: 2.0
  take_profit_ratio: 1.5
  max_drawdown: 0.15
  position_timeout_hours: 24
  min_price_change_pct: 1.2
  max_interval_width_pct: 20.0
  max_active_positions_pct: 0.30
  historical_days: 180
  trading_fee: 0.001
  slippage: 0.0002
  risk_grading_enabled: true
  adaptive_parameters: true

model:
  sequence_length: 60
  retrain_frequency_hours: 24
  monte_carlo_samples: 100
  confidence_interval: 0.95
  lstm_units: [128, 64, 32]
  dropout_rate: 0.2
  recurrent_dropout_rate: 0.2
  dense_dropout_rate: 0.3
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
  validation_split: 0.2

data:
  historical_days: 180
  update_frequency_minutes: 60
  backup_retention_days: 30
  price_data_sources:
    primary: "binance"
    backup: "coingecko"
  features: ["open", "high", "low", "close", "volume"]
  data_validation:
    max_price_change_pct: 20.0
    min_volume_threshold: 1000

telegram:
  notification_level: "normal"
  chart_resolution: "1h"
  max_message_length: 4000
  notification_types:
    trade_entry: true
    trade_exit: true
    high_confidence: true
    model_retrained: true
    risk_alert: true
    daily_summary: true
    parameter_adjustments: true
  chart_indicators: ["price", "predictions", "confidence_bands", "volume"]

database:
  db_path: "data/trading_bot.db"
  backup_path: "data/backups/"
  connection_timeout: 30
  max_retries: 3

api:
  binance:
    name: "binance"
    base_url: "https://api.binance.com"
    rate_limit_per_minute: 1200
    timeout_seconds: 30
  coingecko:
    name: "coingecko"
    base_url: "https://api.coingecko.com/api/v3"
    rate_limit_per_minute: 50
    timeout_seconds: 30

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  log_file: "logs/trading_bot.log"
  max_file_size: 10485760
  backup_count: 5
  console_output: true

security:
  encrypt_database: false
  api_key_encryption: true
  session_timeout_minutes: 60
  max_login_attempts: 3

monitoring:
  health_check_interval: 300
  performance_log_interval: 3600
  alert_thresholds:
    max_memory_mb: 2048
    max_cpu_percent: 80
    min_disk_space_gb: 5
    max_api_errors_per_hour: 10
    min_win_rate_pct: 40.0
    max_drawdown_pct: 15.0
    min_confidence_avg: 0.30
    max_uncertainty_avg: 25.0
    min_trades_per_day: 0.5
    max_trades_per_day: 10.0
    parameter_drift_threshold: 0.20
```

## Deployment Steps

### 1. Test Configuration
```bash
# Test basic configuration loading
python test_config_simple.py

# Test alerts system
python test_alerts_system.py

# Test Telegram notifications (will use your real bot)
python test_telegram_notifications.py
```

### 2. Create Deployment Script

I'll create a deployment script that runs the 7-day paper trading session.

### 3. Create Data Directories
```bash
mkdir -p data/backups
mkdir -p logs
mkdir -p config
```

### 4. Security Checklist

#### ✅ Environment Variables
- Never commit API keys to git
- Use `.env` file (add to `.gitignore`)
- Set appropriate file permissions: `chmod 600 .env`

#### ✅ API Key Security
- Restrict Binance API key to reading only
- Add IP restrictions
- Monitor API usage regularly

#### ✅ Telegram Security
- Keep bot token secure
- Only share chat ID with trusted systems
- Monitor bot activity

## Next Steps

1. **Complete the setup above**
2. **Test with small amounts first**
3. **Monitor the 7-day deployment closely**
4. **Review performance metrics daily**

## Monitoring During Deployment

### Daily Checks
- [ ] Review Telegram notifications
- [ ] Check system logs
- [ ] Monitor performance metrics
- [ ] Verify data quality
- [ ] Check parameter adjustments

### Weekly Review
- [ ] Analyze overall performance
- [ ] Review parameter adaptation
- [ ] Check alert effectiveness
- [ ] Evaluate model performance
- [ ] Plan next steps

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Binance has strict rate limits
2. **Data Quality**: Monitor for missing or stale data
3. **Model Performance**: Watch for degradation
4. **Memory Usage**: Monitor system resources
5. **Network Issues**: Have backup data sources

### Support
- Check logs in `logs/trading_bot.log`
- Monitor Telegram alerts
- Review system status regularly