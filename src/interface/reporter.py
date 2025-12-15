import pandas as pd
import datetime

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        
    def generate(self, gw, team_df, transfers=None, captain=None):
        """Generates a text report for the Gameweek."""
        total_xp = team_df['predicted_points'].sum()
        
        # Calculate Team Rating (simple normalization 0-100 based on realistic max 100 pts)
        rating = min(int((total_xp / 80) * 100), 100)
        
        lines = []
        lines.append(f"GW{gw} FPL AI REPORT")
        lines.append("-" * 20)
        lines.append(f"Team Rating: {rating}/100")
        lines.append(f"Expected Points: {total_xp:.1f}")
        lines.append("")
        
        if transfers:
            lines.append("Transfers:")
            for t_out, t_in in transfers:
                lines.append(f"OUT: {t_out}")
                lines.append(f"IN: {t_in}")
            lines.append("")
        
        if captain:
            lines.append(f"Captain: {captain}")
            # Find VC (highest points not captain)
            vc = team_df[team_df['web_name'] != captain].sort_values('predicted_points', ascending=False).iloc[0]['web_name']
            lines.append(f"Vice: {vc}")
            lines.append("")
            
        lines.append("Starting XI:")
        starters = team_df[team_df['element_type'] != 0] # Assuming filtering happened before
        # Sort by position
        for _, player in team_df.sort_values('element_type').iterrows():
            lines.append(f"{player['web_name']} ({player['predicted_points']:.1f})")
            
        report_content = "\n".join(lines)
        
        # Save
        filename = f"{self.output_dir}/gw{gw}_report.txt"
        with open(filename, "w") as f:
            f.write(report_content)
        
        print("\n" + report_content)
        return report_content

