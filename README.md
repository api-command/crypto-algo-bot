# 🚀 Crypto Algorithmic Trading Bot with AI Sentiment Analysis

![Trading Bot Dashboard](docs/dashboard_screenshot.png)

High-frequency cryptocurrency trading system combining real-time market data with news sentiment analysis for alpha generation.

**Key Features:**
- 🚀 Sub-100ms trade execution latency
- 🤖 AI-powered news sentiment scoring (Hugging Face Transformers)
- 💹 Multi-exchange support (Coinbase Pro, Binance)
- 📊 Technical + fundamental + sentiment signal fusion
- 📈 QuantConnect backtesting integration
- 🔔 Multi-channel alerting (Slack, Telegram, Email)
- 📊 Real-time Grafana dashboards

## Technology Stack

- **Core:** Python 3.11
- **APIs:** CCXT Pro, Alpha Vantage, Hugging Face Inference API
- **ML:** PyTorch 2.0, Transformers, Scikit-Learn
- **Data:** SQLite + Parquet + Redis
- **Infra:** Docker, Prometheus/Grafana, Kubernetes
- **Backtesting:** QuantConnect, Backtrader

## System Architecture

```mermaid
graph TD
    A[Exchange APIs] --> B[Market Data Feed]
    C[News APIs] --> D[Sentinel Agent]
    B --> E[Signal Orchestrator]
    D --> E
    E --> F[Execution Engine]
    F --> G[Exchange APIs]
    H[Risk Manager] --> F
    G --> I[Trade Memory]
    I --> H
    J[Monitoring] --> K[Grafana]