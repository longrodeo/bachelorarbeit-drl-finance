# Deep Reinforcement Learning for Systematic Trading Strategies

**Bachelor Thesis Project | Technische Hochschule Ingolstadt**  
*Expected Completion: February 2026*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-Latest-green.svg)](https://stable-baselines3.readthedocs.io/)


## 🎯 Project Overview

This project develops and evaluates Deep Reinforcement Learning (DRL) agents for algorithmic portfolio management across multiple asset classes. The system implements state-of-the-art backtesting methodologies to address overfitting and ensure robust performance validation in live trading scenarios.

### Key Features

- **Multi-Asset Portfolio Management**: ETFs, Cryptocurrencies with dynamic cash allocation
- **Advanced Backtesting**: Combinatorial Purged Cross-Validation (CPCV) and Walk-Forward Analysis
- **Realistic Market Simulation**: Transaction costs, slippage modeling, and interest-bearing cash positions
- **Professional Data Pipeline**: Tiingo API for market data, Federal Reserve Economic Data (FRED) for interest rates
- **Production-Ready Architecture**: Modular design with comprehensive testing suite (pytest)

## 🏗️ Project Structure
```
bachelorarbeit-drl-finance/
├── config/          # Configuration files for training, data sources, and hyperparameters
├── data/            # Market data storage and preprocessing pipelines
├── notebooks/       # Jupyter notebooks for analysis, visualization, and experimentation
├── results/         # Training results, performance metrics, and TensorBoard logs
├── src/             # Core source code
│   ├── agents/      # DRL agent implementations (PPO, etc.)
│   ├── envs/        # Custom Gym trading environments with market dynamics
│   ├── backtesting/ # CPCV and Walk-Forward validation implementations
│   └── utils/       # Helper functions, data processing, and feature engineering
├── tests/           # Unit and integration tests (pytest)
├── environment.yml  # Conda environment specification
├── requirements.txt # Python dependencies
├── Makefile         # Build automation and common tasks
└── README.md        # This file
```

## 🔬 Methodology

### Reinforcement Learning Framework

**Algorithms**: Proximal Policy Optimization (PPO) implemented via Stable-Baselines3
- State-of-the-art policy gradient method
- Stable training with clipped objective function
- Efficient sample utilization

**State Space**:
- Historical price data and technical indicators
- Portfolio positions and allocations
- Cash holdings and available liquidity
- Federal Reserve interest rate data

**Action Space**: Continuous allocation weights across assets and cash

**Reward Function**: Risk-adjusted returns with transaction cost penalties and realistic execution constraints

### Backtesting & Validation

#### Combinatorial Purged Cross-Validation (CPCV)
- Prevents information leakage in time-series financial data
- Purges observations whose labels overlap with the test set
- Ensures unbiased out-of-sample performance estimation
- Implementation based on López de Prado (2018), *Advances in Financial Machine Learning*

#### Walk-Forward Analysis
- Rolling window training and testing approach
- Simulates real-world deployment conditions where models are retrained periodically
- Validates strategy robustness across different market regimes
- Adapts to market regime changes and evolving dynamics

#### Multi-Scenario Stress Testing
- Bull market, bear market, and high-volatility conditions
- Robustness validation across macroeconomic scenarios
- Performance consistency evaluation

## 📊 Technical Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | Stable-Baselines3, PyTorch, TensorFlow |
| **Data Sources** | Tiingo API (market data), FRED API (macroeconomic data) |
| **Backtesting** | Custom CPCV implementation, NumPy vectorized operations |
| **Monitoring** | TensorBoard for training visualization and metrics tracking |
| **Development** | Git/GitHub, pytest, Makefile automation |
| **Environment** | Python 3.8+, Conda/pip package management |

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/longrodeo/bachelorarbeit-drl-finance.git
cd bachelorarbeit-drl-finance

# Option 1: Create Conda environment (recommended)
conda env create -f environment.yml
conda activate drl-trading

# Option 2: Use pip
pip install -r requirements.txt
```

### Basic Usage

When I completed my bachelor thesis I will integrate a good workflow from data download to run the model
in the moment only the data download and adjustment process is down with one klick.


## 📈 Key Implementation Details

### Custom Gym Trading Environment
- Implements OpenAI Gym interface for compatibility with Stable-Baselines3
- Realistic order execution with transaction costs
- Interest accrual on cash positions using Federal Reserve rates
- Portfolio rebalancing mechanics

### Data Pipeline
- Automated data fetching from Tiingo API
- Federal Reserve interest rate integration
- Feature engineering for technical indicators
- Data validation and quality checks

### Risk Management
- Position sizing based on risk tolerance
- Maximum drawdown constraints
- Portfolio concentration limits
- Dynamic cash allocation

## 🎓 Academic Context

This project is part of a Bachelor thesis at **Technische Hochschule Ingolstadt** (expected completion: **February 2026**), investigating the application of Deep Reinforcement Learning to systematic trading strategies with emphasis on robust validation methodologies that prevent overfitting.

**Research Focus**:
- Addressing backtest overfitting in DRL-based trading systems
- Comparative analysis of cross-validation approaches for financial time-series
- Multi-asset portfolio optimization under realistic market frictions
- Integration of macroeconomic indicators (interest rates) in trading decisions


**Institution**: Technische Hochschule Ingolstadt, Faculty of Wirtschaftsingenieurwesen

## 📝 Current Status

🚧 **Work in Progress** - Active development for Bachelor thesis completion

**Completed**:
- ✅ Custom Gym trading environment with multi-asset support
- ✅ Stable-Baselines3 PPO integration
- ✅ CPCV and Walk-Forward Analysis implementation
- ✅ Data pipeline with Tiingo and FRED APIs
- ✅ Transaction cost and cash interest modeling
- ✅ TensorBoard monitoring integration

**In Progress**:
- 🔄 Final hyperparameter optimization
- 🔄 Comprehensive performance evaluation across market regimes
- 🔄 Thesis documentation and analysis

## 🔮 Future Extensions

Beyond the Bachelor thesis scope, potential extensions include:

**Trading Capabilities**:
- Short selling and leverage
- Options and derivatives trading
- Multi-currency forex with currency risk hedging
- Enhanced transaction cost modeling (market impact, realistic bid-ask spreads)

**Algorithmic Improvements**:
- Ensemble methods combining multiple DRL algorithms
- Meta-learning for rapid strategy adaptation
- Integration of alternative data sources (sentiment, fundamental data)

**Deployment**:
- Live trading integration with broker APIs
- Real-time strategy monitoring dashboard
- Automated risk management and position limits

## 📚 Key References

This implementation draws from established research and best practices:

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. - CPCV methodology
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley. - Walk-Forward Analysis
- Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

## ⚠️ Disclaimer

This is an academic research project for educational purposes only.

- **Not financial advice**: Nothing in this repository constitutes financial, investment, legal, or tax advice.
- **No trading recommendations**: Past performance does not guarantee future results.
- **Research only**: This code is for academic research and learning purposes.
- **Use at own risk**: Any use of this code for live trading is entirely at your own risk.

## 📧 Contact

**Lukas Lang**  
Student, Wirtschaftsingenieurwesen  
Technische Hochschule Ingolstadt

- 📧 Email: [lul2768@thi.de]
- 💼 LinkedIn: [linkedin.com/in/]
- 🐙 GitHub: [@longrodeo](https://github.com/longrode)
---

**Star ⭐ this repository if you find it useful for your own research!**

*Last Updated: December 2025*
