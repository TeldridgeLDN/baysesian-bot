# 🚀 Quick Start: 7-Day Paper Trading Deployment

This guide will get you up and running with the Bayesian Crypto Trading Bot for 7 days of paper trading with live data from Binance and notifications via Telegram.

## ⚡ Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
# Option A: Automatic installation
python install_dependencies.py

# Option B: Manual installation
pip install ccxt python-telegram-bot==20.7 pandas numpy tensorflow
```

### 2. Setup API Keys

#### Binance API (Required)
1. Go to [Binance.com](https://www.binance.com) → Account → API Management
2. Create new API key with **"Enable Reading"** only
3. Save API Key and Secret Key

#### Telegram Bot (Required) 
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Save the Bot Token
4. Send a message to your bot, then visit:
   `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
5. Find your Chat ID in the response

### 3. Configure Environment
```bash
# Copy template
cp .env.example .env

# Edit with your actual keys
nano .env  # or use any text editor
```

**Required fields in .env:**
- `BINANCE_API_KEY=your_actual_api_key`
- `BINANCE_SECRET_KEY=your_actual_secret_key`
- `TELEGRAM_BOT_TOKEN=your_actual_bot_token`
- `TELEGRAM_CHAT_ID=your_actual_chat_id`

### 4. Test Everything
```bash
python test_deployment_readiness.py
```

### 5. Deploy!
```bash
python deploy_paper_trading.py
```

## 📱 What You'll Get

### Real-Time Notifications
- **Trade Alerts**: Every buy/sell signal with confidence levels
- **Parameter Adjustments**: When the bot optimizes its strategy
- **Daily Summaries**: Performance at 6 PM daily
- **Risk Alerts**: When drawdown or win rate issues detected

### Performance Monitoring
- **Live Trading**: Paper trading with $10,000 virtual capital
- **Adaptive Parameters**: Bot adjusts strategy based on performance
- **Risk Management**: Automatic stop-losses and position sizing
- **Comprehensive Logging**: Everything logged to `logs/deployment.log`

## 🎯 Key Features

### Bayesian LSTM Model
- **Uncertainty Quantification**: Confidence intervals on predictions
- **Monte Carlo Dropout**: Robust uncertainty estimation
- **Adaptive Learning**: Retrains every 24 hours with new data

### Intelligent Trading
- **Paper Trading Only**: No real money at risk
- **Risk-Graded Positions**: Position size scales with confidence
- **Dynamic Stop Losses**: Based on model uncertainty
- **Market Anomaly Detection**: Pauses trading during unusual conditions

### Advanced Monitoring
- **Real-Time Alerts**: 10+ alert types for comprehensive monitoring
- **Performance Tracking**: Win rate, drawdown, Sharpe ratio
- **System Health**: Memory, CPU, API status monitoring
- **Parameter Drift Detection**: Alerts when parameters need adjustment

## 📊 Expected Performance

Based on backtesting with 180 days of historical data:

- **Target Win Rate**: 40-50% (conservative but profitable)
- **Expected Drawdown**: 8-12% (well within 15% safety limit)
- **Average Confidence**: 42% (models trade selectively)
- **Trading Frequency**: 1-3 trades per day (not overtrading)

## 🛡️ Safety Features

### Paper Trading Only
- **No Real Trading**: All trades are simulated
- **Read-Only API**: Binance API configured for data only
- **Virtual Capital**: $10,000 starting balance
- **Risk Management**: Multiple safety checks and limits

### Monitoring & Alerts
- **Performance Alerts**: Low win rate, high drawdown warnings
- **System Alerts**: API failures, data quality issues
- **Parameter Drift**: Automatic optimization when performance degrades
- **Graceful Shutdown**: Ctrl+C for clean exit with final summary

## 📈 Monitoring Your Deployment

### Telegram Notifications
You'll receive notifications for:
- 📊 **Trade Signals**: BUY/SELL with confidence and reasoning
- ⚙️ **Parameter Changes**: When bot optimizes strategy
- 📱 **Daily Summaries**: Performance recap at 6 PM
- 🚨 **Risk Alerts**: If performance degrades

### Log Files
Check `logs/deployment.log` for detailed information:
- Trade decisions and reasoning
- Model predictions and confidence
- System performance metrics
- Error handling and recovery

### Performance Metrics
Key metrics to watch:
- **Win Rate**: Target 40-50%
- **Total Return**: Cumulative performance
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns

## 🔧 Troubleshooting

### Common Issues

**"Missing environment variables"**
```bash
# Check your .env file has actual values (not placeholders)
cat .env
```

**"Binance API connection failed"**
- Verify API key and secret in .env
- Check API key has "Enable Reading" permission
- Ensure IP restrictions allow your current IP

**"Telegram bot connection failed"**
- Verify bot token in .env
- Send a message to your bot first
- Check chat ID is correct number (not username)

**"TensorFlow not available"**
- On Apple Silicon: `pip install tensorflow-macos tensorflow-metal`
- On Intel/AMD: `pip install tensorflow`
- Bot will use fallback predictions if TensorFlow unavailable

### Getting Help

1. **Check Logs**: `tail -f logs/deployment.log`
2. **Test Configuration**: `python test_deployment_readiness.py`
3. **Review Setup**: See `DEPLOYMENT_GUIDE.md` for detailed instructions

## 🎉 Success Indicators

Your deployment is working correctly when you see:

✅ **Initial Setup**
- All dependency tests pass
- Binance API connects successfully
- Telegram bot sends test message
- Configuration validates

✅ **During Operation**
- Regular Telegram notifications
- Trades executed with reasoning
- Performance metrics updating
- No critical errors in logs

✅ **Daily Operations**
- Daily summary at 6 PM
- Parameter adjustments when needed
- Alerts for performance issues
- Consistent data updates

## 📋 7-Day Deployment Checklist

- [ ] Dependencies installed (`python install_dependencies.py`)
- [ ] API keys configured in `.env`
- [ ] Configuration test passes (`python test_deployment_readiness.py`)
- [ ] Deployment started (`python deploy_paper_trading.py`)
- [ ] First Telegram notification received
- [ ] Monitor daily for 7 days
- [ ] Review final performance summary

**Estimated Setup Time**: 5-10 minutes  
**Deployment Duration**: 7 days continuous  
**Monitoring**: Automatic via Telegram + manual log review  

---

🎯 **Goal**: Validate the trading bot's performance over 7 days with real market conditions but no financial risk.

📱 **Stay Connected**: All important events will be sent to your Telegram chat.

🛡️ **Safety First**: Paper trading only - no real money involved.