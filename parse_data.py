import os
import pandas as pd
from bs4 import BeautifulSoup

SCORE_DIRECTORY = "data/scores"
box_scores_files = os.listdir(SCORE_DIRECTORY)
box_scores_files = [os.path.join(SCORE_DIRECTORY, file) for file in box_scores_files if file.endswith(".html")]

def parse_html_box_score(box_score_file):
    with open(box_score_file) as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content)
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup

def read_season_information(soup):
    navigation = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in navigation.find_all('a')]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

def read_line_score_data(soup):
    line_score_data = pd.read_html(str(soup), attrs={'id': 'line_score'})[0]
    column_names = list(line_score_data.columns)
    column_names[0] = "team"
    column_names[-1] = "total"
    line_score_data.columns = column_names
    line_score_data = line_score_data[["team", "total"]]
    return line_score_data

def read_team_stats(soup, team, stat):
    df = pd.read_html(str(soup), attrs={'id': f'box-{team}-game-{stat}'}, index_col=0)[0]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

games_data = []
base_columns = None

for box_score_file in box_scores_files:
    soup_box_score = parse_html_box_score(box_score_file)
    line_score_data = read_line_score_data(soup_box_score)
    teams_list = list(line_score_data["team"])

    game_summaries = []
    for team_name in teams_list:
        basic_stats = read_team_stats(soup_box_score, team_name, "basic")
        advanced_stats = read_team_stats(soup_box_score, team_name, "advanced")

        total_stats = pd.concat([basic_stats.iloc[-1, :], advanced_stats.iloc[-1, :]])
        total_stats.index = total_stats.index.str.lower()

        max_stats = pd.concat([basic_stats.iloc[:-1].max(), advanced_stats.iloc[:-1].max()])
        max_stats.index = max_stats.index.str.lower() + "_max"

        team_summary = pd.concat([total_stats, max_stats])

        if base_columns is None:
            base_columns = list(team_summary.index.drop_duplicates(keep="first"))
            base_columns = [col for col in base_columns if "bpm" not in col]

        team_summary = team_summary[base_columns]

        game_summaries.append(team_summary)
    
    game_summary_data = pd.concat(game_summaries, axis=1).T
    game_data = pd.concat([game_summary_data, line_score_data], axis=1)

    game_data["home"] = [0, 1]

    game_opponent = game_data.iloc[::-1].reset_index()
    game_opponent.columns += "_opp"

    full_game_data = pd.concat([game_data, game_opponent], axis=1)
    full_game_data["season"] = read_season_information(soup_box_score)
    
    full_game_data["date"] = os.path.basename(box_score_file)[:8]
    full_game_data["date"] = pd.to_datetime(full_game_data["date"], format="%Y%m%d")
    
    full_game_data["won"] = full_game_data["total"] > full_game_data["total_opp"]
    games_data.append(full_game_data)
    
    if len(games_data) % 100 == 0:
        print(f"{len(games_data)} / {len(box_scores_files)}")
