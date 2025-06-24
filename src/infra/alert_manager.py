import aiohttp
import asyncio
import json
import time
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config_loader import config_loader
from src.infra.telemetry import telemetry

logger = get_logger('alert_manager')

class AlertManager:
    def __init__(self):
        self.config = config_loader.load_toml('config/bot_params.toml').get('alerts', {})
        self.channels = self.config.get('channels', ['log'])
        self.alert_levels = self.config.get('levels', ['CRITICAL'])
        self.session = None
        self.kill_switch_triggered = False
        self.last_alert_time = {}
        self.cooldown_periods = {
            'INFO': 60,      # 1 minute
            'WARNING': 300,  # 5 minutes
            'CRITICAL': 0    # No cooldown
        }
    
    async def init_session(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """
        Send alert to configured channels
        :param alert_type: Short code for alert type (e.g., LATENCY_SPIKE, LIQUIDITY_CRISIS)
        :param message: Detailed alert message
        :param severity: DEBUG, INFO, WARNING, CRITICAL
        """
        # Check if we're in cooldown period for this alert type
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        cooldown = self.cooldown_periods.get(severity, 60)
        
        if current_time - last_time < cooldown:
            logger.debug(f"Alert {alert_type} suppressed due to cooldown")
            return
        
        # Update last alert time
        self.last_alert_time[alert_type] = current_time
        
        # Format alert payload
        payload = {
            "type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Record telemetry
        telemetry.incr('alerts', tags={'type': alert_type, 'severity': severity})
        
        # Send to all configured channels
        tasks = []
        if 'slack' in self.channels:
            tasks.append(self._send_slack(payload))
        if 'telegram' in self.channels:
            tasks.append(self._send_telegram(payload))
        if 'email' in self.channels:
            tasks.append(self._send_email(payload))
        if 'log' in self.channels:
            tasks.append(self._log_alert(payload))
        
        # Run all notifications concurrently
        await asyncio.gather(*tasks)
    
    async def _send_slack(self, payload):
        """Send alert to Slack webhook"""
        webhook_url = self.config.get('slack_webhook')
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        try:
            async with self.session.post(
                webhook_url,
                json={
                    "text": f":warning: *{payload['type']}* ({payload['severity']})\n{payload['message']}",
                    "username": "Trading Bot Alerts"
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"Slack alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Slack alert error: {e}")
    
    async def _send_telegram(self, payload):
        """Send alert via Telegram bot"""
        bot_token = self.config.get('telegram_token')
        chat_id = self.config.get('telegram_chat_id')
        if not bot_token or not chat_id:
            logger.warning("Telegram credentials not configured")
            return
        
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            async with self.session.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": f"*{payload['type']}* ({payload['severity']})\n{payload['message']}",
                    "parse_mode": "Markdown"
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"Telegram alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
    
    async def _send_email(self, payload):
        """Send alert via email (using SendGrid)"""
        api_key = self.config.get('sendgrid_api_key')
        sender_email = self.config.get('email_sender')
        receiver_email = self.config.get('email_receiver')
        if not api_key or not sender_email or not receiver_email:
            logger.warning("Email credentials not configured")
            return
        
        try:
            url = "https://api.sendgrid.com/v3/mail/send"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "personalizations": [{"to": [{"email": receiver_email}]}],
                "from": {"email": sender_email},
                "subject": f"{payload['severity']} Alert: {payload['type']}",
                "content": [{"type": "text/plain", "value": payload['message']}]
            }
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status >= 300:
                    logger.error(f"Email alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Email alert error: {e}")
    
    async def _log_alert(self, payload):
        """Log alert to file and console"""
        if payload['severity'] == 'CRITICAL':
            logger.critical(f"{payload['type']}: {payload['message']}")
        elif payload['severity'] == 'WARNING':
            logger.warning(f"{payload['type']}: {payload['message']}")
        else:
            logger.info(f"{payload['type']}: {payload['message']}")
    
    async def critical_kill_switch(self, reason: str):
        """Activate kill switch and send critical alerts"""
        if self.kill_switch_triggered:
            return  # Already triggered
        
        self.kill_switch_triggered = True
        
        # 1. Send critical alert
        await self.send_alert(
            "SYSTEM_SHUTDOWN", 
            f"Kill switch activated: {reason}",
            severity="CRITICAL"
        )
        
        # 2. Trigger kill switch in trading system
        # (Implementation depends on your trading system)
        # Example: await trading_system.emergency_shutdown()
        
        # 3. Optional: System shutdown
        # os._exit(1)  # Immediate shutdown

# Singleton for easy access
alert_manager = AlertManager()