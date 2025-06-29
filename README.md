# Bayesian Crypto Trading Bot

A sophisticated cryptocurrency trading bot using Bayesian LSTM with Monte Carlo Dropout for Bitcoin price prediction and automated trading execution via Telegram interface.

## 🚨 **Known Issues & Workarounds**

**Before starting development, check [ERRATA.md](ERRATA.md) for known issues and workarounds.**

Key current issues:
- Task Master AI dependency resolution (cosmetic, non-blocking)
- API key requirements for full Task Master functionality (workarounds available)

## 🎯 **Project Overview**

This bot implements advanced machine learning techniques for cryptocurrency trading:

- **Bayesian LSTM Model**: Uncertainty quantification for price predictions
- **Monte Carlo Dropout**: Confidence estimation for trading decisions
- **Intelligent Capital Allocation**: Risk-adjusted position sizing
- **Automated Trade Execution**: Dynamic stop-loss and take-profit management
- **Telegram Interface**: Real-time notifications and manual controls

## 📊 **Success Metrics**

- **ROI Target**: Positive returns over 6-month evaluation period
- **Risk Management**: Maximum drawdown <15%
- **Prediction Accuracy**: >55% directional accuracy
- **System Uptime**: >99% availability during market hours

## 🏗️ **Project Structure**

```
telegram_bot/
├── src/                    # Core application modules
│   ├── data/              # Data collection and processing
│   ├── models/            # ML models and training
│   ├── trading/           # Trading logic and execution
│   ├── telegram/          # Bot interface and notifications
│   └── utils/             # Configuration and utilities
├── .taskmaster/           # Task Master AI integration
├── memory_bank/           # Development context and modes
├── config/                # Configuration files
├── tests/                 # Test suites
├── docs/                  # Documentation
└── ERRATA.md             # 🚨 Known issues and workarounds
```

## 🚀 **Quick Start**

### Prerequisites
```bash
python 3.9+
pip install -r requirements.txt
```

### Task Management
This project uses Task Master AI for intelligent task management:

```bash
# View all tasks
task-master list

# Get next task to work on
task-master next

# View specific task details
task-master show TASK-002
```

**Note**: If Task Master shows dependency resolution errors, refer to [ERRATA.md](ERRATA.md) for workarounds.

## 🛠️ **Development Modes**

The project supports 5 development modes via the memory bank system:

- **VAN**: Vanilla development approach
- **PLAN**: Strategic planning and architecture
- **CREATIVE**: Innovative problem-solving
- **IMPLEMENT**: Focused implementation
- **QA**: Quality assurance and testing

See `memory_bank/modes/` for specific rules and guidelines.

## 📋 **Current Development Status**

- ✅ **Architecture Setup**: Complete project structure with Task Master AI integration
- ✅ **Task Planning**: 9 comprehensive tasks with 84 hours estimated effort
- 🔄 **Ready for TASK-002**: Configuration System implementation

## 🔧 **Configuration**

Configuration is managed through YAML files in the `config/` directory:

- `default.yaml`: Base configuration
- Environment-specific overrides supported
- Secure credential management via environment variables

## 🧪 **Testing Strategy**

- **Unit Tests**: Individual component testing
- **Integration Tests**: System interaction testing
- **Backtests**: Historical trading performance validation
- **Paper Trading**: Live market simulation

## 📚 **Documentation**

- `bayesian_crypto_bot_prd.md`: Complete Product Requirements Document (479 lines)
- `memory_bank/`: Development context and system patterns
- `docs/`: API documentation and trading strategies
- `CHANGELOG.md`: Version history and changes
- `ERRATA.md`: Known issues and workarounds

## 🔐 **Security**

- Environment variables for API keys and credentials
- No hardcoded secrets in codebase
- Secure database configuration
- Rate limiting for API calls

## 🤝 **Contributing**

1. Check [ERRATA.md](ERRATA.md) for known issues
2. Review current tasks in Task Master AI
3. Follow development mode guidelines
4. Update changelog for significant changes

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**For troubleshooting and known issues, always check [ERRATA.md](ERRATA.md) first.** 