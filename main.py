"""
Crypto Algorithmic Trading Bot with AI Sentiment Analysis
FastAPI Server for API endpoints and web interface
"""

import os
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Import our trading bot components
from src.core.execution_engine import ExecutionEngine
from src.core.risk_manager import RiskManager
from src.ml.signal_orchestrator import SignalOrchestrator
from src.data.market_feeds import MarketDataFeed
from src.data.sentinel_agent import SentinelAgent
from src.utils.logger import get_logger
from src.utils.config import Config

# Setup logging
logger = get_logger('main_server')

# Global instances
market_feed = None
sentinel_agent = None
signal_orchestrator = None
execution_engine = None
risk_manager = None
trading_active = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    global market_feed, sentinel_agent, signal_orchestrator, execution_engine, risk_manager
    
    logger.info("🚀 Starting Crypto Trading Bot...")
    
    # Initialize components
    try:
        # Initialize components
        config = Config()
        
        # Initialize market data feed
        market_feed = MarketDataFeed()
        
        # Initialize sentiment analysis
        sentinel_agent = SentinelAgent()
        
        # Initialize signal orchestrator
        signal_orchestrator = SignalOrchestrator()
        
        # Initialize risk manager
        risk_manager = RiskManager()
        
        # Initialize execution engine (sandbox mode by default)
        execution_engine = ExecutionEngine(
            exchange_name='coinbase_pro' if os.getenv('COINBASE_SANDBOX') == 'true' else 'binance'
        )
        
        logger.info("✅ All components initialized successfully")
        
        # Start background tasks
        if os.getenv('AUTO_START_TRADING', 'false').lower() == 'true':
            asyncio.create_task(start_trading_loop())
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Trading Bot...")
    global trading_active
    trading_active = False
    
    if market_feed:
        await market_feed.stop()
    if sentinel_agent:
        await sentinel_agent.stop()

# Create FastAPI app with lifespan
app = FastAPI(
    title="Crypto Trading Bot API",
    description="High-frequency cryptocurrency trading system with AI sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/")
async def home():
    """Main dashboard"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Trading Bot Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .card { background: #2d2d2d; padding: 20px; margin: 20px 0; border-radius: 8px; border: 1px solid #444; }
            .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
            .status.active { background: #0d5d2d; }
            .status.inactive { background: #5d1a1a; }
            .button { background: #007acc; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            .button:hover { background: #005fa3; }
            .danger { background: #dc3545; }
            .danger:hover { background: #c82333; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        </style>
        <script>
            async function fetchStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    document.getElementById('status-content').innerHTML = JSON.stringify(data, null, 2);
                } catch (e) {
                    document.getElementById('status-content').innerHTML = 'Error: ' + e.message;
                }
            }
            
            async function startTrading() {
                try {
                    const response = await fetch('/api/trading/start', {method: 'POST'});
                    const data = await response.json();
                    alert(data.message);
                    fetchStatus();
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }
            
            async function stopTrading() {
                try {
                    const response = await fetch('/api/trading/stop', {method: 'POST'});
                    const data = await response.json();
                    alert(data.message);
                    fetchStatus();
                } catch (e) {
                    alert('Error: ' + e.message);
                }
            }
            
            // Auto-refresh status every 5 seconds
            setInterval(fetchStatus, 5000);
            
            // Load initial status
            window.onload = fetchStatus;
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Crypto Trading Bot Dashboard</h1>
                <p>High-frequency cryptocurrency trading with AI sentiment analysis</p>
            </div>
            
            <div class="card">
                <h2>🎮 Controls</h2>
                <button class="button" onclick="startTrading()">▶️ Start Trading</button>
                <button class="button danger" onclick="stopTrading()">⏹️ Stop Trading</button>
                <button class="button" onclick="fetchStatus()">🔄 Refresh Status</button>
                <button class="button" onclick="window.open('/docs', '_blank')">📖 API Docs</button>
            </div>
            
            <div class="card">
                <h2>📊 System Status</h2>
                <pre id="status-content" style="background: #1a1a1a; padding: 15px; border-radius: 4px; overflow: auto;">
Loading...
                </pre>
            </div>
            
            <div class="card">
                <h2>🔑 Setup Instructions</h2>
                <ol>
                    <li><strong>Get Alpha Vantage API Key:</strong> Visit <a href="https://www.alphavantage.co/support/#api-key" target="_blank">Alpha Vantage</a> (Free tier: 25 requests/day)</li>
                    <li><strong>Get Hugging Face Token:</strong> Visit <a href="https://huggingface.co/settings/tokens" target="_blank">Hugging Face</a> (Free tier available)</li>
                    <li><strong>Get Exchange API Keys:</strong>
                        <ul>
                            <li><a href="https://pro.coinbase.com/profile/api" target="_blank">Coinbase Pro</a> (Sandbox available)</li>
                            <li><a href="https://www.binance.com/en/my/settings/api-management" target="_blank">Binance</a> (Testnet available)</li>
                        </ul>
                    </li>
                    <li><strong>Update .env file</strong> with your API keys</li>
                    <li><strong>Start trading</strong> in sandbox mode first!</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/api/status")
async def get_status():
    """Get system status"""
    global trading_active, market_feed, sentinel_agent, signal_orchestrator, execution_engine, risk_manager
    
    status = {
        "trading_active": trading_active,
        "paper_trading": os.getenv('PAPER_TRADING', 'true').lower() == 'true',
        "coinbase_sandbox": os.getenv('COINBASE_SANDBOX', 'true').lower() == 'true',
        "binance_testnet": os.getenv('BINANCE_TESTNET', 'true').lower() == 'true',
        "components": {
            "market_feed": market_feed is not None,
            "sentiment_agent": sentinel_agent is not None,
            "signal_orchestrator": signal_orchestrator is not None,
            "execution_engine": execution_engine is not None,
            "risk_manager": risk_manager is not None,
        },
        "api_keys": {
            "alpha_vantage": os.getenv('ALPHA_VANTAGE_API_KEY', '').startswith('demo'),
            "hugging_face": os.getenv('HUGGING_FACE_API_KEY', '').startswith('hf_demo'),
            "coinbase": os.getenv('COINBASE_PRO_API_KEY', '') != 'your_coinbase_pro_api_key',
            "binance": os.getenv('BINANCE_API_KEY', '') != 'your_binance_api_key',
        },
        "environment": {
            "max_daily_loss": os.getenv('MAX_DAILY_LOSS', '-0.05'),
            "max_position_risk": os.getenv('MAX_POSITION_RISK', '0.1'),
            "log_level": os.getenv('LOG_LEVEL', 'INFO'),
        }
    }
    
    return status

@app.post("/api/trading/start")
async def start_trading(background_tasks: BackgroundTasks):
    """Start the trading system"""
    global trading_active
    
    if trading_active:
        raise HTTPException(status_code=400, detail="Trading is already active")
    
    # Start trading in background
    background_tasks.add_task(start_trading_loop)
    
    return {"message": "Trading started successfully", "status": "active"}

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop the trading system"""
    global trading_active
    
    if not trading_active:
        raise HTTPException(status_code=400, detail="Trading is not active")
    
    trading_active = False
    logger.info("Trading stopped by user request")
    
    return {"message": "Trading stopped successfully", "status": "inactive"}

@app.get("/api/signals/{symbol}")
async def get_signals(symbol: str):
    """Get trading signals for a symbol"""
    if not signal_orchestrator:
        raise HTTPException(status_code=503, detail="Signal orchestrator not initialized")
    
    try:
        signal = await signal_orchestrator.generate_trading_signal(symbol)
        return signal
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a symbol"""
    if not market_feed:
        raise HTTPException(status_code=503, detail="Market feed not initialized")
    
    try:
        orderbook = market_feed.get_orderbook(symbol)
        ohlcv = market_feed.get_ohlcv(symbol)
        
        return {
            "symbol": symbol,
            "orderbook": orderbook,
            "ohlcv": ohlcv[-10:] if ohlcv else [],  # Last 10 candles
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio status"""
    if not execution_engine:
        raise HTTPException(status_code=503, detail="Execution engine not initialized")
    
    # This would typically return real portfolio data
    # For now, return mock data
    return {
        "cash_balance": 10000.0,
        "positions": [
            {"symbol": "BTC/USD", "size": 0.1, "value": 4500.0},
            {"symbol": "ETH/USD", "size": 2.5, "value": 5000.0},
        ],
        "total_value": 19500.0,
        "pnl_daily": 500.0,
        "pnl_percent": 2.63
    }

async def start_trading_loop():
    """Main trading loop"""
    global trading_active, market_feed, sentinel_agent, signal_orchestrator, execution_engine, risk_manager
    
    trading_active = True
    logger.info("🎯 Starting trading loop...")
    
    # Start market data feeds
    if market_feed:
        market_tasks = await market_feed.start()
    
    # Start news sentiment monitoring
    if sentinel_agent:
        asyncio.create_task(sentinel_agent.start())
    
    # Main trading loop
    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    
    while trading_active:
        try:
            for symbol in symbols:
                if not trading_active:
                    break
                
                # Generate trading signal
                if signal_orchestrator:
                    signal = await signal_orchestrator.generate_trading_signal(symbol)
                    
                    # Check risk management
                    if risk_manager:
                        risk_manager.update_risk_profile()
                        position_size = risk_manager.get_position_size(symbol)
                        
                        # Generate trade recommendation
                        current_position = 0  # Get from portfolio
                        recommendation = signal_orchestrator.generate_trade_recommendation(signal, current_position)
                        
                        logger.info(f"Signal for {symbol}: {signal['final_score']:.3f}, Action: {recommendation['action']}")
                        
                        # Execute trade (only in real trading mode and with actual API keys)
                        if (not os.getenv('PAPER_TRADING', 'true').lower() == 'true' and 
                            recommendation['action'] in ['buy', 'sell'] and
                            execution_engine):
                            
                            # This would execute the actual trade
                            logger.info(f"Would execute {recommendation['action']} for {symbol} with size {recommendation['size']}")
            
            # Wait before next iteration
            await asyncio.sleep(30)  # 30 second intervals
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(10)  # Wait before retrying
    
    logger.info("🛑 Trading loop stopped")

if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", 8001))
    
    logger.info(f"🌐 Starting server on port {port}")
    logger.info(f"📊 Dashboard: http://localhost:{port}")
    logger.info(f"📖 API Docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("DEBUG_MODE", "True") == "True" else False,
        log_level="info"
    )