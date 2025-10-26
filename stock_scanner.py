from typing import List, Dict
from api_manager import APIManager
from technical_analyzer import TechnicalAnalyzer
from market_analyzer import MarketAnalyzer
from config import TRADING_CONFIG
from logger import logger


class StockScanner:
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
        self.technical = TechnicalAnalyzer(api_manager)
        self.market = MarketAnalyzer(api_manager)
    
    def scan_for_swing_trades(self, watchlist: List[str]) -> List[Dict]:
        """Scan for swing trading opportunities"""
        config = TRADING_CONFIG["swing"]
        timeframe = config["timeframe"]
        min_score = config["min_score"]
        
        # TEMPORARY: Lower threshold for testing
        min_score = 50  # Reduced from 70 to 50
        
        opportunities = []
        errors = []
        
        for symbol in watchlist:
            try:
                logger.debug(f"Scanning {symbol} for swing trades...")

                # Technical analysis
                analysis = self.technical.analyze_stock(symbol, timeframe)

                if not analysis:
                    logger.warning(f"No analysis data for {symbol}")
                    errors.append(f"{symbol}: No analysis data")
                    continue

                logger.debug(f"{symbol} - Score: {analysis['score']:.0f}/100")

                if analysis["score"] < min_score:
                    logger.debug(f"{symbol} - Score {analysis['score']:.0f} below threshold {min_score}")
                    continue
                
                # Support/Resistance levels
                sr_data = self.api.fetch_support_resistance([symbol], timeframe)
                if sr_data and "data" in sr_data:
                    sr_key = f"NSE_{symbol.replace('-EQ', '')}"
                    sr_levels = sr_data["data"].get(sr_key, {})

                    if not sr_levels:
                        logger.warning(f"No S/R levels for {symbol}")
                        continue

                    analysis["support_resistance"] = sr_levels

                    # Calculate risk/reward
                    current_price = analysis["current_price"]
                    stop_loss = sr_levels.get("s1", current_price * 0.95)
                    target = sr_levels.get("r1", current_price * 1.05)

                    risk = current_price - stop_loss
                    reward = target - current_price

                    if risk > 0:
                        rr_ratio = reward / risk
                        logger.debug(f"{symbol} - Entry: ₹{current_price:.2f}, Target: ₹{target:.2f}, SL: ₹{stop_loss:.2f}, R:R: 1:{rr_ratio:.2f}")

                        # TEMPORARY: Lower R:R threshold for testing
                        min_rr = 1.0  # Reduced from 2.0 to 1.0

                        if rr_ratio >= min_rr:
                            analysis["entry"] = current_price
                            analysis["stop_loss"] = stop_loss
                            analysis["target"] = target
                            analysis["risk_reward"] = round(rr_ratio, 2)
                            opportunities.append(analysis)
                            logger.info(f"✅ {symbol} added - Score: {analysis['score']}, R:R: 1:{rr_ratio:.2f}")
                        else:
                            logger.debug(f"{symbol} - R:R {rr_ratio:.2f} below threshold {min_rr}")
                    else:
                        logger.warning(f"{symbol} - Invalid risk calculation (risk={risk})")
                else:
                    logger.warning(f"No S/R data available for {symbol}")
                    
            except Exception as e:
                error_msg = f"{symbol}: {str(e)}"
                logger.error(f"Error scanning {symbol}: {str(e)}")
                errors.append(error_msg)
                continue

        logger.info(f"Swing scan complete - Total: {len(watchlist)}, Opportunities: {len(opportunities)}, Errors: {len(errors)}")

        if errors:
            logger.warning(f"Errors encountered during scan: {errors[:5]}")

        return opportunities
    
    def scan_for_intraday_trades(self, watchlist: List[str]) -> List[Dict]:
        """Scan for intraday trading opportunities"""
        config = TRADING_CONFIG["intraday"]
        timeframe = config["timeframe"]
        min_score = config["min_score"]
        
        opportunities = []
        
        # Check market health first
        market_health = self.market.get_market_health()
        if market_health["health"] == "bearish":
            print("⚠️  Market is bearish, reducing intraday opportunities")
            min_score += 10  # Increase threshold in bearish market
        
        for symbol in watchlist:
            print(f"🔍 Scanning {symbol} for intraday trades...")
            
            # Technical analysis
            analysis = self.technical.analyze_stock(symbol, timeframe)
            if not analysis or analysis["score"] < min_score:
                continue
            
            # Check for high volume
            if analysis["volume_ratio"] < 1.2:
                continue  # Skip low volume stocks for intraday
            
            # Support/Resistance levels
            sr_data = self.api.fetch_support_resistance([symbol], timeframe)
            if sr_data and "data" in sr_data:
                sr_key = f"NSE_{symbol.replace('-EQ', '')}"
                sr_levels = sr_data["data"].get(sr_key, {})
                analysis["support_resistance"] = sr_levels
                
                # Calculate intraday risk/reward
                current_price = analysis["current_price"]
                stop_loss = sr_levels.get("s1", current_price * 0.98)
                target = sr_levels.get("r1", current_price * 1.02)
                
                risk = current_price - stop_loss
                reward = target - current_price
                
                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio >= config["risk_reward_ratio"]:
                        analysis["entry"] = current_price
                        analysis["stop_loss"] = stop_loss
                        analysis["target"] = target
                        analysis["risk_reward"] = round(rr_ratio, 2)
                        analysis["market_health"] = market_health["health"]
                        opportunities.append(analysis)
        
        return sorted(opportunities, key=lambda x: x["score"], reverse=True)
