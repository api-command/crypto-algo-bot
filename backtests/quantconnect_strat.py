from QuantConnect import *
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.Custom import *

class SentimentAlphaStrategy(QCAlgorithm):
    def Initialize(self):
        # Core configuration
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        self.symbol = self.AddCrypto("BTCUSD", Resolution.Minute).Symbol
        
        # Sentiment data pipeline
        self.AddData(AlphaVantageNews, "AV_NEWS", Resolution.Minute)
        self.sentiment_model = self.LoadMLModel("sentiment_model.pkl")
        
        # Execution parameters
        self.slippage_model = ConstantSlippageModel(0.0005)  # 5bps slippage
        self.SetBenchmark("BTCUSD")

    def OnData(self, data):
        if "AV_NEWS" not in data or not data["AV_NEWS"].IsValue: 
            return
            
        # Process news events
        news = data["AV_NEWS"]
        sentiment_score = self.sentiment_model.Predict(news.Text)
        
        # Generate trading signal
        current_position = self.Portfolio[self.symbol].Quantity
        signal = self.GenerateSignal(sentiment_score, current_position)
        
        # Execute with risk management
        if signal != 0:
            self.Order(self.symbol, signal * self.CalcOrderSize())

    def GenerateSignal(self, sentiment, position):
        """-1=Short, 0=Hold, 1=Long"""
        if sentiment > 0.8 and position <= 0:
            return 1  # Strong buy signal
        elif sentiment < 0.2 and position >= 0:
            return -1  # Strong sell signal
        return 0

    def CalcOrderSize(self):
        # Risk-managed position sizing
        volatility = self.STD(self.symbol, 30, Resolution.Daily)
        return self.Portfolio.TotalPortfolioValue * 0.01 / volatility

    def OnOrderEvent(self, orderEvent):
        # Latency monitoring
        latency = (self.Time - orderEvent.UtcTime).total_seconds()
        self.Log(f"Order {orderEvent.OrderId} executed with {latency*1000:.2f}ms latency")