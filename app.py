import sys
import pandas as pd
import pickle
import re
import os

# Load model and datasets
model = pickle.load(open("stacked_fantasy_model.pkl", "rb"))
df_main = pd.read_csv("CricMasters_2.0.csv")
df_main.rename(columns={'Player Type': 'Player_Type'}, inplace=True)
stadium_df = pd.read_csv("stadium_data.csv")

role_map = {
    "WICKETKEEPER": "WK", "BATTER": "BAT",
    "ALLROUNDER": "ALL", "BOWLER": "BOWL"
}

num_columns = [
    "Runs_Scored", "Batting_Average", "Batting_Strike_Rate", "Balls_Faced", "Centuries",
    "Half_Centuries", "Fours", "Sixes", "Matches_Bowled",
    "Wickets_Taken", "Bowling_Average", "Economy_Rate", "Bowling_Strike_Rate",
    "Four_Wicket_Hauls", "Five_Wicket_Hauls", "Bat_Consistency","Catches_Taken",
    "Stumpings", "Credits", "Consistency_Score","Bowl_Consistency"
]

def main():
    if len(sys.argv) < 2:
        print("❌ Match number not provided")
        sys.exit(1)

    match_no = int(sys.argv[1])

    # Use Docker path if it exists, else fallback to local Downloads
    if os.path.exists("/app/data/SquadPlayerNames_IndianT20League.xlsx"):
        input_file = "/app/data/SquadPlayerNames_IndianT20League.xlsx"
    else:
        input_file = os.path.expanduser("~/Downloads/SquadPlayerNames_IndianT20League.xlsx")

    if not os.path.exists(input_file):
        print("❌ Input file not found")
        sys.exit(1)

    try:
        df_playing_22 = pd.read_excel(input_file, sheet_name=f"Match_{match_no}")

        df_playing_22 = df_playing_22[df_playing_22["IsPlaying"] == "PLAYING"]

        pitch_type = stadium_df[stadium_df["match_no"] == match_no]["Best_for"].values[0] \
            if match_no in stadium_df["match_no"].values else "Balanced"

        match_players = df_playing_22.merge(df_main, on="Player Name", how="left")

        match_players["Credits"] = pd.to_numeric(
            match_players.get("Credits_x", match_players.get("Credits", 0)),
            errors='coerce'
        ).fillna(0)
        match_players["Player_Type"] = match_players["Player_Type"].str.upper().map(role_map).fillna(match_players["Player_Type"])
        match_players["Team"] = match_players.get("Team_x", match_players.get("Team", "Unknown"))

        for col in num_columns:
            match_players[col] = pd.to_numeric(match_players[col], errors='coerce').fillna(0)

        match_players["Predicted_Fantasy_Points"] = model.predict(match_players[num_columns])

        if pitch_type == "Bowling":
            match_players.loc[match_players["Player_Type"].isin(["BOWL", "ALL"]), "Consistency_Score"] *= 1.05
        elif pitch_type == "Batting":
            match_players.loc[match_players["Player_Type"].isin(["BAT", "ALL", "WK"]), "Consistency_Score"] *= 1.05
        elif pitch_type == "Balanced":
            match_players["Consistency_Score"] *= 1.02

        match_players = match_players.sort_values(by=["Predicted_Fantasy_Points", "Consistency_Score","lineupOrder"], ascending=False)
        match_players = match_players.drop_duplicates(subset=["Player Name", "Team"])

        selected_df = pd.DataFrame()
        total_credits = 0
        team_counts = {}
        role_counts = {"BAT": 0, "BOWL": 0, "ALL": 0, "WK": 0}

        for _, row in match_players.iterrows():
            if len(selected_df) >= 10:
                break
            if row["Credits"] + total_credits > 90:
                continue
            if team_counts.get(row["Team"], 0) >= 7:
                continue
            if row["Player_Type"] == "BOWL" and role_counts["BOWL"] >= 4:
                continue
            if row["Player_Type"] == "BAT" and role_counts["BAT"] >= 4:
                continue
            selected_df = pd.concat([selected_df, pd.DataFrame([row])])
            total_credits += row["Credits"]
            team_counts[row["Team"]] = team_counts.get(row["Team"], 0) + 1
            role_counts[row["Player_Type"]] += 1

        # Check if both teams have at least one BAT and one ALL, if not, enforce the constraint.
        team_a = selected_df[selected_df["Team"] == selected_df.iloc[0]["Team"]]
        team_b = selected_df[selected_df["Team"] != selected_df.iloc[0]["Team"]]

        team_a_bat = team_a[team_a["Player_Type"] == "BAT"]
        team_a_all = team_a[team_a["Player_Type"] == "ALL"]
        team_b_bat = team_b[team_b["Player_Type"] == "BAT"]
        team_b_all = team_b[team_b["Player_Type"] == "ALL"]

        # If team A or team B doesn't have at least one BAT and one ALL, add the missing player.
        if len(team_a_bat) == 0 or len(team_a_all) == 0:
            missing_type = "BAT" if len(team_a_bat) == 0 else "ALL"
            missing_player = match_players[
                (match_players["Player_Type"] == missing_type) & 
                (~match_players["Player Name"].isin(selected_df["Player Name"]))
            ]
            selected_df = pd.concat([selected_df, missing_player.iloc[0:1]])

        if len(team_b_bat) == 0 or len(team_b_all) == 0:
            missing_type = "BAT" if len(team_b_bat) == 0 else "ALL"
            missing_player = match_players[
                (match_players["Player_Type"] == missing_type) & 
                (~match_players["Player Name"].isin(selected_df["Player Name"]))
            ]
            selected_df = pd.concat([selected_df, missing_player.iloc[0:1]])

        for _, row in match_players.iterrows():
            if len(selected_df) == 11:
                break
            if row["Player Name"] in selected_df["Player Name"].values:
                continue
            if selected_df["Credits"].sum() + row["Credits"] > 100:
                continue
            if team_counts.get(row["Team"], 0) >= 7:
                continue
            if row["Player_Type"] == "BOWL" and role_counts["BOWL"] >= 4:
                continue
            if row["Player_Type"] == "BAT" and role_counts["BAT"] >= 4:
                continue
            selected_df = pd.concat([selected_df, pd.DataFrame([row])])
            team_counts[row["Team"]] = team_counts.get(row["Team"], 0) + 1
            role_counts[row["Player_Type"]] += 1

        if role_counts["BAT"] < 4:
            bats_needed = 4 - role_counts["BAT"]
            bats = match_players[
                (match_players["Player_Type"] == "BAT") & 
                (~match_players["Player Name"].isin(selected_df["Player Name"]))
            ]
            for _, bat in bats.iterrows():
                if bats_needed == 0 or len(selected_df) >= 11:
                    break
                if bat["Credits"] + selected_df["Credits"].sum() <= 100:
                    selected_df = pd.concat([selected_df, pd.DataFrame([bat])])
                    role_counts["BAT"] += 1
                    team_counts[bat["Team"]] = team_counts.get(bat["Team"], 0) + 1
                    bats_needed -= 1

        while len(selected_df) > 11 or selected_df["Credits"].sum() > 100:
            selected_df = selected_df.sort_values(by=["Consistency_Score", "Credits"], ascending=[True, False])
            row = selected_df.iloc[0]
            role_counts[row["Player_Type"]] -= 1
            team_counts[row["Team"]] -= 1
            selected_df = selected_df.iloc[1:]

        remaining = match_players[~match_players["Player Name"].isin(selected_df["Player Name"])]
        for _, row in remaining.iterrows():
            if len(selected_df) == 11:
                break
            if selected_df["Credits"].sum() + row["Credits"] <= 100:
                selected_df = pd.concat([selected_df, pd.DataFrame([row])])
                role_counts[row["Player_Type"]] += 1
                team_counts[row["Team"]] = team_counts.get(row["Team"], 0) + 1

        if len(selected_df) != 11:
            print("❌ Could not finalize 11 players.")
            sys.exit(1)

        selected_df["CV_Score"] = (
            0.7 * selected_df["Consistency_Score"] +
            0.3 * (1 - selected_df["lineupOrder"] / selected_df["lineupOrder"].max())  # earlier in lineup preferred
        )

        candidates = selected_df[selected_df["Player_Type"].isin(["BAT", "ALL", "WK"])]
        candidates = candidates.sort_values(by="CV_Score", ascending=False)
        if len(candidates) >= 2:
            c_player = candidates.iloc[0]["Player Name"]
            vc_player = candidates.iloc[1]["Player Name"]
            print(f"Captain: {c_player}, Vice-Captain: {vc_player}")

        selected_df.to_csv("Fantasy_Team.csv", index=False)
        print("✅ Fantasy Team selected and saved to 'Fantasy_Team.csv'")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
