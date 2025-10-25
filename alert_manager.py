from typing import List, Dict
from api_manager import APIManager
from datetime import datetime


class AlertManager:
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
    
    def send_swing_alert(self, opportunities: List[Dict]):
        """Send swing trading alert"""
        if not opportunities:
            return

        message = "<b>🎯 Swing Trading Opportunities</b>\n\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"📊 Found {len(opportunities)} opportunities\n\n"

        for i, opp in enumerate(opportunities[:5], 1):  # Top 5
            message += f"<b>{i}. {opp['symbol']}</b>\n"
            message += f"   💰 Price: ₹{opp['current_price']:.2f}\n"
            message += f"   📈 Entry: ₹{opp['entry']:.2f}\n"
            message += f"   🎯 Target: ₹{opp['target']:.2f}\n"
            message += f"   🛑 Stop Loss: ₹{opp['stop_loss']:.2f}\n"
            message += f"   📊 R:R = 1:{opp['risk_reward']:.2f}\n"
            message += f"   ⭐ Score: {opp['score']:.0f}/100\n"
            
            if opp.get('rsi'):
                message += f"   📉 RSI: {opp['rsi']:.1f}\n"
            if opp.get('adx'):
                message += f"   💪 ADX: {opp['adx']:.1f}\n"

            message += "\n"

        self.api.send_telegram_alert(message)

    def send_intraday_alert(self, opportunities: List[Dict]):
        """Send intraday trading alert"""
        if not opportunities:
            return

        message = "<b>⚡ Intraday Trading Opportunities</b>\n\n"
        message += f"⏰ {datetime.now().strftime('%H:%M:%S')}\n"
        message += f"📊 Found {len(opportunities)} opportunities\n\n"
        
        for i, opp in enumerate(opportunities[:3], 1):  # Top 3
            message += f"<b>{i}. {opp['symbol']}</b>\n"
            message += f"   💰 Price: ₹{opp['current_price']:.2f}\n"
            message += f"   📈 Entry: ₹{opp['entry']:.2f}\n"
            message += f"   🎯 Target: ₹{opp['target']:.2f}\n"
            message += f"   🛑 SL: ₹{opp['stop_loss']:.2f}\n"
            message += f"   📊 R:R = 1:{opp['risk_reward']:.2f}\n"
            message += f"   ⭐ Score: {opp['score']:.0f}/100\n"
            message += f"   📦 Volume: {opp['volume_ratio']:.1f}x avg\n"
            message += f"   🌡️ Market: {opp['market_health']}\n\n"
        
        self.api.send_telegram_alert(message)
    
    def send_market_update(self, market_health: Dict):
        """Send market health update"""
        message = "📊 <b>Market Health Update</b>\n\n"
        message += f"⏰ {datetime.now().strftime('%H:%M:%S')}\n\n"
        message += f"🌡️ Market Health: <b>{market_health['health'].upper()}</b>\n"
        message += f"📈 Score: {market_health['score']:.1f}/100\n\n"
        message += f"📊 Advance/Decline:\n"
        message += f"   ✅ Advancing: {market_health['advancing']}\n"
        message += f"   ❌ Declining: {market_health['declining']}\n"
        message += f"   ➖ Total: {market_health['total']}\n"
        message += f"   📊 A/D Ratio: {market_health['ad_ratio']:.1f}%\n\n"
        message += f"🏢 Sectors:\n"
        message += f"   ✅ Positive: {market_health['positive_sectors']}/{market_health['total_sectors']}\n"
        
        self.api.send_telegram_alert(message)
