from src.optimization.chips import ChipStrategy

def test_chips():
    # Mock History: Used Wildcard in GW3
    mock_history = {
        'chips': [
            {'name': 'wildcard', 'event': 3, 'time': '2024-08-01'},
            {'name': 'bboost', 'event': 15, 'time': '2024-12-01'}
        ]
    }
    
    team_id = 12345
    print("\n--- Testing Chip Strategy ---")
    strategy = ChipStrategy(team_id, mock_history)
    print(f"Used Chips: {strategy.used_chips}")
    
    # Mock Squad (irrelevant for WC availability logic, but needed for method sig)
    import pandas as pd
    mock_df = pd.DataFrame({'predicted_points': [5.0]*15, 'web_name': ['Player']*15})
    
    # Test 1: Current GW = 19 (Should be USED/Unavailable)
    print("\n[Test 1] Current GW = 19")
    recs_19 = strategy.analyze(mock_df, mock_df, current_gw=19)
    for r in recs_19:
        if r['chip'] == 'Wildcard':
            print(f"GW19 Wildcard Status: {r['recommendation']} ({r['reason']})")
            
    # Test 2: Current GW = 20 (Should be AVAILABLE - WC2)
    print("\n[Test 2] Current GW = 20")
    recs_20 = strategy.analyze(mock_df, mock_df, current_gw=20)
    for r in recs_20:
        if r['chip'] == 'Wildcard':
            print(f"GW20 Wildcard Status: {r['recommendation']} ({r['reason']})")

    # Test 3: Free Hit Check
    print("\n[Test 3] Free Hit Status")
    # Not used in history
    for r in recs_20:
        if r['chip'] == 'Free Hit':
             print(f"Free Hit Status: {r['recommendation']} ({r['reason']})")

if __name__ == "__main__":
    test_chips()
