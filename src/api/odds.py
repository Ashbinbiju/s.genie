"""
Bookmaker Odds Client

Fetches betting odds from two sources:
1. Historical: football-data.co.uk CSVs (for training data)
2. Live: the-odds-api.com (for inference)

All odds are converted to implied probabilities with margin removal.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from io import StringIO


# League-average fallback values (2023-24 PL averages)
LEAGUE_DEFAULTS = {
    'win_prob': 0.33,
    'draw_prob': 0.33,
    'loss_prob': 0.33,
    'team_implied_goals': 1.35,
    'opponent_implied_goals': 1.35,
    'clean_sheet_prob': 0.30,
    'anytime_goal_scorer_prob': 0.10,  # ~10% for an average starter
}

# Position-based scoring rates (goals per 90 given team scores)
# Used to derive anytime_goal_scorer_prob from team_implied_goals
POSITION_GOAL_SHARE = {
    'FWD': 0.32,
    'MID': 0.22,
    'DEF': 0.06,
    'GK':  0.001,
    'GKP': 0.001,
}

# Normalize football-data.co.uk names → Vaastav/FPL names
TEAM_NAME_NORMALIZE = {
    'Man United': 'Man Utd',
    'Tottenham':  'Spurs',
    'Sheffield United': 'Sheffield Utd',
    'Leeds':      'Leeds',
    'Leicester':  'Leicester',
    'Southampton': 'Southampton',
    'Ipswich':    'Ipswich',
    'Sunderland': 'Sunderland',
}


class OddsClient:
    def __init__(self, cache_dir="data/cache", raw_dir="data/raw"):
        self.cache_dir = cache_dir
        self.raw_dir = raw_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(raw_dir, "odds"), exist_ok=True)
        
        # the-odds-api.com key (optional)
        self.api_key = os.environ.get("ODDS_API_KEY", "")
        
    # ---------------------------------------------------------------
    # 1. Historical odds from football-data.co.uk
    # ---------------------------------------------------------------
    def download_historical_odds(self, seasons=None):
        """Download PL match odds CSVs from football-data.co.uk."""
        if seasons is None:
            seasons = {
                "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
                "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
                "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
            }
        
        for season, url in seasons.items():
            out_path = os.path.join(self.raw_dir, "odds", f"pl_odds_{season}.csv")
            if os.path.exists(out_path):
                print(f"Historical odds for {season} already cached.")
                continue
                
            print(f"Downloading odds for {season}...")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(resp.text)
                print(f"  Saved to {out_path}")
            except Exception as e:
                print(f"  Failed to download {season}: {e}")
                
    def load_historical_odds(self, season):
        """Load historical match odds and convert to per-team implied probs."""
        path = os.path.join(self.raw_dir, "odds", f"pl_odds_{season}.csv")
        if not os.path.exists(path):
            return None
            
        df = pd.read_csv(path, encoding='latin-1')
        
        # Use market average odds (AvgH, AvgD, AvgA) or Bet365 as fallback
        h_col = 'AvgH' if 'AvgH' in df.columns else 'B365H'
        d_col = 'AvgD' if 'AvgD' in df.columns else 'B365D'
        a_col = 'AvgA' if 'AvgA' in df.columns else 'B365A'
        
        # Over/Under 2.5 goals
        over_col = 'Avg>2.5' if 'Avg>2.5' in df.columns else 'B365>2.5'
        under_col = 'Avg<2.5' if 'Avg<2.5' in df.columns else 'B365<2.5'
        
        rows = []
        for _, match in df.iterrows():
            try:
                home_team = str(match.get('HomeTeam', ''))
                away_team = str(match.get('AwayTeam', ''))
                home_team = TEAM_NAME_NORMALIZE.get(home_team, home_team)
                away_team = TEAM_NAME_NORMALIZE.get(away_team, away_team)
                date = str(match.get('Date', ''))
                
                h_odds = float(match.get(h_col, 0))
                d_odds = float(match.get(d_col, 0))
                a_odds = float(match.get(a_col, 0))
                
                if h_odds <= 1 or d_odds <= 1 or a_odds <= 1:
                    continue
                    
                # Convert to implied probs and remove margin
                raw_h = 1 / h_odds
                raw_d = 1 / d_odds
                raw_a = 1 / a_odds
                margin = raw_h + raw_d + raw_a
                
                win_h = raw_h / margin
                draw = raw_d / margin
                win_a = raw_a / margin
                
                # Implied goals from over/under 2.5
                over_odds = float(match.get(over_col, 0)) if pd.notna(match.get(over_col)) else 0
                under_odds = float(match.get(under_col, 0)) if pd.notna(match.get(under_col)) else 0
                
                if over_odds > 1 and under_odds > 1:
                    raw_over = 1 / over_odds
                    raw_under = 1 / under_odds
                    margin_ou = raw_over + raw_under
                    over_prob = raw_over / margin_ou
                    # Approximate total implied goals from over 2.5 probability
                    # Using a Poisson approximation: P(>2.5) ≈ 1 - P(0) - P(1) - P(2)
                    # Rough mapping: over_prob -> total goals
                    total_goals = 1.5 + over_prob * 2.5  # Linear approx
                else:
                    total_goals = 2.7  # PL average
                
                # Split total goals by win probability ratio
                # Higher win prob → more goals for that team
                home_goals = total_goals * (win_h / (win_h + win_a + 0.001)) * 1.8
                away_goals = total_goals * (win_a / (win_h + win_a + 0.001)) * 1.8
                
                # Clean sheet probabilities (Poisson approximation)
                home_cs = np.exp(-away_goals)  # P(away scores 0)
                away_cs = np.exp(-home_goals)  # P(home scores 0)
                
                # Home team row
                rows.append({
                    'date': date,
                    'team_name': home_team,
                    'opponent_name': away_team,
                    'is_home': True,
                    'win_prob': win_h,
                    'draw_prob': draw,
                    'loss_prob': win_a,
                    'team_implied_goals': home_goals,
                    'opponent_implied_goals': away_goals,
                    'clean_sheet_prob': home_cs,
                })
                
                # Away team row
                rows.append({
                    'date': date,
                    'team_name': away_team,
                    'opponent_name': home_team,
                    'is_home': False,
                    'win_prob': win_a,
                    'draw_prob': draw,
                    'loss_prob': win_h,
                    'team_implied_goals': away_goals,
                    'opponent_implied_goals': home_goals,
                    'clean_sheet_prob': away_cs,
                })
            except (ValueError, TypeError):
                continue
                
        return pd.DataFrame(rows)
    
    # ---------------------------------------------------------------
    # 2. Live odds from the-odds-api.com
    # ---------------------------------------------------------------
    def fetch_live_odds(self):
        """Fetch current PL odds from the-odds-api.com."""
        cache_path = os.path.join(self.cache_dir, "live_odds.json")
        
        # Use cache if less than 6 hours old
        if os.path.exists(cache_path):
            age_hours = (os.path.getmtime(cache_path) - os.path.getctime(cache_path)) / 3600
            import time
            age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_hours < 6:
                with open(cache_path, 'r') as f:
                    return json.load(f)
        
        if not self.api_key:
            print("No ODDS_API_KEY set. Using fallback defaults.")
            return None
            
        url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'uk',
            'markets': 'h2h,totals',
            'oddsFormat': 'decimal',
        }
        
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            print(f"Fetched live odds for {len(data)} matches.")
            return data
        except Exception as e:
            print(f"Failed to fetch live odds: {e}")
            return None
    
    def parse_live_odds(self, data):
        """Parse the-odds-api response into per-team odds features."""
        if not data:
            return {}
            
        team_odds = {}
        
        for event in data:
            home_team = event['home_team']
            away_team = event['away_team']
            
            h2h_odds = {'home': [], 'draw': [], 'away': []}
            totals_odds = {'over': [], 'under': []}
            
            for bk in event.get('bookmakers', []):
                for market in bk.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                h2h_odds['home'].append(outcome['price'])
                            elif outcome['name'] == away_team:
                                h2h_odds['away'].append(outcome['price'])
                            elif outcome['name'] == 'Draw':
                                h2h_odds['draw'].append(outcome['price'])
                    elif market['key'] == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                totals_odds['over'].append(outcome['price'])
                            elif outcome['name'] == 'Under':
                                totals_odds['under'].append(outcome['price'])
            
            # Average across bookmakers
            avg = lambda lst: sum(lst) / len(lst) if lst else 0
            h_avg = avg(h2h_odds['home'])
            d_avg = avg(h2h_odds['draw'])
            a_avg = avg(h2h_odds['away'])
            
            if h_avg <= 1 or d_avg <= 1 or a_avg <= 1:
                continue
                
            # Normalize
            raw_h, raw_d, raw_a = 1/h_avg, 1/d_avg, 1/a_avg
            margin = raw_h + raw_d + raw_a
            win_h, draw, win_a = raw_h/margin, raw_d/margin, raw_a/margin
            
            # Total goals
            over_avg = avg(totals_odds['over'])
            under_avg = avg(totals_odds['under'])
            if over_avg > 1 and under_avg > 1:
                over_prob = (1/over_avg) / (1/over_avg + 1/under_avg)
                total_goals = 1.5 + over_prob * 2.5
            else:
                total_goals = 2.7
                
            home_goals = total_goals * (win_h / (win_h + win_a + 0.001)) * 1.8
            away_goals = total_goals * (win_a / (win_h + win_a + 0.001)) * 1.8
            home_cs = np.exp(-away_goals)
            away_cs = np.exp(-home_goals)
            
            team_odds[home_team] = {
                'win_prob': win_h, 'draw_prob': draw, 'loss_prob': win_a,
                'team_implied_goals': home_goals, 'opponent_implied_goals': away_goals,
                'clean_sheet_prob': home_cs,
            }
            team_odds[away_team] = {
                'win_prob': win_a, 'draw_prob': draw, 'loss_prob': win_h,
                'team_implied_goals': away_goals, 'opponent_implied_goals': home_goals,
                'clean_sheet_prob': away_cs,
            }
            
        return team_odds
    
    # ---------------------------------------------------------------
    # 3. Helper: derive anytime_goal_scorer_prob
    # ---------------------------------------------------------------
    @staticmethod
    def compute_anytime_scorer_prob(team_implied_goals, position):
        """
        Approximate anytime goal scorer probability from team implied goals
        and the player's position.
        
        P(player scores) ≈ 1 - e^(-lambda * position_share)
        where lambda = team_implied_goals
        """
        share = POSITION_GOAL_SHARE.get(position, 0.10)
        player_lambda = team_implied_goals * share
        return 1 - np.exp(-player_lambda)
    
    # ---------------------------------------------------------------
    # 4. Get current GW odds features per team
    # ---------------------------------------------------------------
    def get_current_odds(self):
        """Get odds features for the current GW. Returns dict keyed by team name."""
        data = self.fetch_live_odds()
        if data:
            return self.parse_live_odds(data)
        return {}


if __name__ == "__main__":
    client = OddsClient()
    client.download_historical_odds()
    
    # Test historical
    df = client.load_historical_odds("2023-24")
    if df is not None:
        print(f"\nLoaded {len(df)} historical odds rows for 2023-24")
        print(df.head(10).to_string(index=False))
    
    # Test live
    odds = client.get_current_odds()
    if odds:
        print(f"\nLive odds for {len(odds)} teams")
        for team, o in list(odds.items())[:4]:
            print(f"  {team}: win={o['win_prob']:.2f}, goals={o['team_implied_goals']:.2f}, cs={o['clean_sheet_prob']:.2f}")
    else:
        print("\nNo live odds available (set ODDS_API_KEY env var).")
