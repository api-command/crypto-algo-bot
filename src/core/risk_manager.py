def load_risk_profile(profile_name):
    config_path = f"config/risk_profiles/{profile_name}.toml"
    return toml.load(config_path)

class RiskManager:
    def __init__(self):
        self.current_profile = "default"
        
    def detect_volatility_regime(self):
        # Calculate 5-minute volatility
        returns = self.data['close'].pct_change().dropna()
        volatility = returns.rolling(window=30).std() * np.sqrt(365*24*12)  # Annualized
        return volatility.iloc[-1]
    
    def update_risk_profile(self):
        volatility = self.detect_volatility_regime()
        
        if volatility > 0.8:  # >80% annualized volatility
            self.current_profile = "high_volatility"
        elif volatility > 0.5:
            self.current_profile = "medium_volatility"
        else:
            self.current_profile = "low_volatility"
            
        self.profile_config = load_risk_profile(self.current_profile)
        
    def get_position_size(self, symbol):
        base_size = self.capital * self.profile_config['position_sizing']['per_trade_risk']
        volatility_adjusted = base_size / self.get_volatility(symbol)
        return min(volatility_adjusted, 
                  self.profile_config['position_sizing']['max_position_size'])