import os


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir

    def generate(self, gw, team_df, transfers=None, captain=None, starters=None):
        """
        Generates a text report for the Gameweek.

        Pass `starters` to report the actual XI; without it the whole squad is listed
        and labelled as such, rather than mislabelling 15 players as a starting XI.
        """
        total_xp = team_df['predicted_points'].sum()
        rating = min(int((total_xp / 80) * 100), 100)

        lines = [
            f"GW{gw} FPL AI REPORT",
            "-" * 20,
            f"Team Rating: {rating}/100",
            f"Expected Points: {total_xp:.1f}",
            "",
        ]

        if transfers:
            lines.append("Transfers:")
            for t_out, t_in in transfers:
                lines.append(f"OUT: {t_out}")
                lines.append(f"IN: {t_in}")
            lines.append("")

        if captain:
            lines.append(f"Captain: {captain}")
            others = team_df[team_df['web_name'] != captain]
            if not others.empty:
                vc = others.sort_values('predicted_points', ascending=False).iloc[0]['web_name']
                lines.append(f"Vice: {vc}")
            lines.append("")

        if starters is not None and not starters.empty:
            lines.append("Starting XI:")
            listing = starters
        else:
            lines.append(f"Squad ({len(team_df)} players):")
            listing = team_df

        for _, player in listing.sort_values('element_type').iterrows():
            lines.append(f"{player['web_name']} ({player['predicted_points']:.1f})")

        report_content = "\n".join(lines)

        # The output directory is gitignored and absent on a fresh clone; create it.
        os.makedirs(self.output_dir, exist_ok=True)
        filename = os.path.join(self.output_dir, f"gw{gw}_report.txt")

        # Explicit UTF-8: player names contain accented characters (Højlund, Sánchez)
        # which raise UnicodeEncodeError under the Windows default codepage.
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)

        print("\n" + report_content)
        return report_content
