import pandas as pd

def select_starting_xi(team_df):
    """
    Selects the best starting XI from a squad of 15.
    Rules:
    - 1 GK
    - Min 3 DEF, Max 5 DEF
    - Min 2 MID, Max 5 MID
    - Min 1 FWD, Max 3 FWD
    - Total 11 players
    - Captain: Highest predicted points
    - Vice: Second highest
    """
    team_df = team_df.copy()
    
    # Sort by predicted points descending
    team_df = team_df.sort_values('predicted_points', ascending=False)
    
    gks = team_df[team_df['element_type'] == 1]
    defs = team_df[team_df['element_type'] == 2]
    mids = team_df[team_df['element_type'] == 3]
    fwds = team_df[team_df['element_type'] == 4]
    
    # Must Haves - Collection of (index, row) is hard, let's collect indices
    starter_idxs = []
    bench_idxs = []
    
    # 1. Pick best GK
    starter_idxs.append(gks.index[0])
    if len(gks) > 1:
        bench_idxs.append(gks.index[1])
        
    # 2. Pick min requirements (3 DEF, 2 MID, 1 FWD)
    starter_idxs.extend(defs.iloc[:3].index.tolist())
    starter_idxs.extend(mids.iloc[:2].index.tolist())
    starter_idxs.extend(fwds.iloc[:1].index.tolist())
    
    # Remaining pool for the last 4 spots
    pool_df = pd.concat([
        defs.iloc[3:],
        mids.iloc[2:],
        fwds.iloc[1:]
    ])
    
    # Sort pool by points
    pool_df = pool_df.sort_values('predicted_points', ascending=False)
    
    # Fill remaining 4 spots
    n_def = 3
    n_mid = 2
    n_fwd = 1
    
    for idx, p in pool_df.iterrows():
        if len(starter_idxs) == 11:
            bench_idxs.append(idx)
            continue
            
        added = False
        ptype = p['element_type']
        
        if ptype == 2 and n_def < 5:
            starter_idxs.append(idx)
            n_def += 1
            added = True
        elif ptype == 3 and n_mid < 5:
            starter_idxs.append(idx)
            n_mid += 1
            added = True
        elif ptype == 4 and n_fwd < 3:
            starter_idxs.append(idx)
            n_fwd += 1
            added = True
            
        if not added:
            bench_idxs.append(idx)
            
    # Create DataFrames
    df_starters = team_df.loc[starter_idxs]
    df_bench = team_df.loc[bench_idxs]
    
    # Sort Bench order: GK always last or first?
    # Usually FPL Bench is: GK, then Points order.
    # We'll just sort by points desc for now
    if not df_bench.empty:
         df_bench = df_bench.sort_values('predicted_points', ascending=False)
    
    return df_starters, df_bench
