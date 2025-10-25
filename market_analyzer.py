from typing import Dict, List, Optional
from api_manager import APIManager

class MarketAnalyzer:
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
    
    def get_market_health(self) -> Dict:
        """Calculate overall market health score"""
        breadth = self.api.fetch_market_breadth()
        sector_perf = self.api.fetch_sector_performance()
        
        if not breadth or not sector_perf:
            return {
                "health": "unknown",
                "score": 0,
                "ad_ratio": 0,
                "advancing": 0,
                "declining": 0,
                "total": 0,
                "sector_score": 0,
                "positive_sectors": 0,
                "total_sectors": 0
            }
        
        # Calculate advance-decline ratio
        total = breadth.get("breadth", {}).get("total", 0)
        advancing = breadth.get("breadth", {}).get("advancing", 0)
        declining = breadth.get("breadth", {}).get("declining", 0)
        
        if total == 0:
            ad_ratio = 0
        else:
            ad_ratio = (advancing / total) * 100
        
        # Calculate sector momentum
        sectors = sector_perf.get("data", [])
        positive_sectors = sum(1 for s in sectors if s.get("changePercent", 0) > 0)
        sector_score = (positive_sectors / len(sectors) * 100) if sectors else 0
        
        # Overall market score
        market_score = (ad_ratio * 0.6) + (sector_score * 0.4)
        
        # Determine market health
        if market_score >= 70:
            health = "bullish"
        elif market_score >= 40:
            health = "neutral"
        else:
            health = "bearish"
        
        return {
            "health": health,
            "score": round(market_score, 2),
            "ad_ratio": round(ad_ratio, 2),
            "advancing": advancing,
            "declining": declining,
            "total": total,
            "sector_score": round(sector_score, 2),
            "positive_sectors": positive_sectors,
            "total_sectors": len(sectors)
        }
    
    def get_bullish_sectors(self, min_change: float = 0.5) -> List[Dict]:
        """Get sectors with positive momentum"""
        breadth = self.api.fetch_market_breadth()
        
        if not breadth:
            return []
        
        industries = breadth.get("industry", [])
        bullish = [
            {
                "name": ind["Industry"],
                "change": ind["avgChange"],
                "advancing": ind["advancing"],
                "total": ind["total"]
            }
            for ind in industries
            if ind.get("avgChange", 0) >= min_change
        ]
        
        return sorted(bullish, key=lambda x: x["change"], reverse=True)
    
    def get_trending_indices(self) -> List[Dict]:
        """Get top trending sector indices"""
        sector_perf = self.api.fetch_sector_performance()
        
        if not sector_perf:
            return []
        
        indices = sector_perf.get("data", [])
        trending = [
            {
                "index": idx["sector_index"],
                "momentum": idx.get("momentum", 0),
                "change_percent": idx.get("changePercent", 0),
                "price": idx.get("price", 0)
            }
            for idx in indices
        ]
        
        return sorted(trending, key=lambda x: x["momentum"], reverse=True)
