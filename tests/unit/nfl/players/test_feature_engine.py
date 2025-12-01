import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure project root is on path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from winsight_predictions.nfl.players.features import FeatureEngine


def make_prior_games(target_col: str = 'passing_yards') -> pd.DataFrame:
    base_date = datetime(2025, 9, 1)
    rows = []
    for i in range(12):
        rows.append({
            'game_date': base_date + timedelta(days=i*7),
            'is_home': 1 if i % 2 == 0 else 0,
            target_col: 100 + i * 10,
            'attempted_passes': 30 + i,
            'completed_passes': 20 + i,
            'epa': 0.5 + i * 0.1,
            'epa_added': 1.0 + i * 0.2,
            'rush_attempts': 10 + i,
            'rush_yards': 40 + i * 5,
            'times_pass_target': 8 + i,
            'receptions': 5 + i,
            'big_play_count_10': i % 3,
            'big_play_count_20': i % 4,
            'big_play_count_30': i % 5,
            'big_play_count_40': i % 6,
            'big_play_count_50': i % 7,
            'year': 2025,
            'off_pct': 0.5 + (i * 0.02),
        })
    return pd.DataFrame(rows)


def test_rolling_target_stats_basic():
    prior = make_prior_games('passing_yards')
    row = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1})
    fe = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row, position='QB')
    stats = fe.rolling_target_stats
    assert stats['overall_avg_passing_yards'] == prior['passing_yards'].mean()
    assert stats['last5_avg_passing_yards'] == prior['passing_yards'].tail(5).mean()
    assert stats['last10_avg_passing_yards'] == prior['passing_yards'].tail(10).mean()
    assert stats['last_game_passing_yards'] == prior['passing_yards'].iloc[-1]


def test_dependent_stats_use_predictions_over_history():
    prior = make_prior_games('passing_yards')
    row = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1})
    preds = {'attempted_passes': 42, 'completed_passes': 35}
    fe = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row, position='QB', predicted_features=preds)
    deps = fe.dependent_stats
    assert deps['dep_attempted_passes_for_passing_yards'] == 42
    assert deps['dep_completed_passes_for_passing_yards'] == 35


def test_weather_features_parsing_and_defaults():
    prior = make_prior_games('passing_yards')
    row = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1, 'weather': '65|55|10'})
    fe = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row, position='QB')
    wf = fe.weather_features
    assert wf == {'temperature': 65.0, 'humidity': 55.0, 'wind_speed': 10.0}

    row2 = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1})
    fe2 = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row2, position='QB')
    wf2 = fe2.weather_features
    assert wf2 == {'temperature': 72, 'humidity': 50, 'wind_speed': 0}


def test_days_rest_computation():
    prior = make_prior_games('passing_yards')
    date = prior['game_date'].max() + timedelta(days=6)
    row = pd.Series({'game_date': date, 'is_home': 1})
    fe = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row, position='QB')
    dr = fe.days_rest_feature
    assert dr['days_rest'] == 6


def test_over_under_parsing_various_sources():
    prior = make_prior_games('passing_yards')
    row_ou = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1, 'ou': 47.5})
    fe_ou = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row_ou, position='QB')
    assert fe_ou.over_under_feature['over_under'] == 47.5

    row_ou_str = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1, 'over_under': '47.5 total'})
    fe_ou_str = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row_ou_str, position='QB')
    assert fe_ou_str.over_under_feature['over_under'] == 47.5


def test_game_targets_with_src_df_and_without():
    prior = make_prior_games('passing_yards')
    row = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1, 'abbr': 'ABC'})
    src_df = pd.DataFrame([{
        'pred_home_points': 24.0,
        'pred_away_points': 20.0,
        'pred_home_total_yards': 350.0,
        'pred_away_total_yards': 320.0,
        'pred_home_pass_yards': 250.0,
        'pred_away_pass_yards': 220.0,
        'pred_home_rush_yards': 100.0,
        'pred_away_rush_yards': 110.0,
        'pred_home_pass_attempts': 32.0,
        'pred_away_pass_attempts': 28.0,
        'pred_home_rush_attempts': 26.0,
        'pred_away_rush_attempts': 24.0,
        'pred_home_win': 0.6,
        'pred_away_win': 0.4,
    }])
    fe = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row, position='QB', game_src_df=src_df)
    gtf = fe.game_targets_features
    assert gtf['team_game_points'] == 24.0
    assert gtf['opp_game_points'] == 20.0
    assert gtf['team_game_win'] == 0.6
    assert gtf['opp_game_win'] == 0.4

    # Fallback to row values when src_df is empty
    row_fb = pd.Series({
        'game_date': prior['game_date'].max(), 'is_home': 0, 'abbr': 'ABC',
        'team_game_points': 21.0, 'opp_game_points': 27.0, 'team_game_win': 0.3, 'opp_game_win': 0.7,
        'team_game_total_yards': 300.0, 'opp_game_total_yards': 360.0,
        'team_game_pass_yards': 200.0, 'opp_game_pass_yards': 240.0,
        'team_game_rush_yards': 110.0, 'opp_game_rush_yards': 120.0,
        'team_game_pass_attempts': 28.0, 'opp_game_pass_attempts': 31.0,
        'team_game_rush_attempts': 24.0, 'opp_game_rush_attempts': 25.0,
    })
    fe_fb = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row_fb, position='QB', game_src_df=pd.DataFrame())
    gtf_fb = fe_fb.game_targets_features
    assert gtf_fb['team_game_points'] == 21.0
    assert gtf_fb['opp_game_points'] == 27.0
    assert gtf_fb['team_game_win'] == 0.3
    assert gtf_fb['opp_game_win'] == 0.7


def test_memoization_of_properties():
    prior = make_prior_games('passing_yards')
    row = pd.Series({'game_date': prior['game_date'].max(), 'is_home': 1, 'abbr': 'ABC'})
    fe = FeatureEngine(prior_games=prior, target_name='passing_yards', row=row, position='QB')
    a = fe.rolling_target_stats
    b = fe.rolling_target_stats
    assert a is b  # cached dict instance


def test_load_features_combines_all_parts_minimally():
    prior = make_prior_games('passing_yards')
    row = pd.Series({
        'game_date': prior['game_date'].max() + timedelta(days=3),
        'is_home': 1,
        'abbr': 'ABC',
        'away_abbr': 'XYZ',
        'weather': '70|60|5',
        'ou': 45.0,
        'spread': -3.5,
        'starter': 1,
        'week': 5,
        'last_week': 4,
        'year': 2025,
    })

    # Minimal external sources empty to validate graceful defaults
    fe = FeatureEngine(
        prior_games=prior,
        target_name='passing_yards',
        row=row,
        position='QB',
        player_data=pd.DataFrame(),
        player_data_big_plays=pd.DataFrame(columns=['game_date','home_abbr','away_abbr','abbr','pos','key']),
        standings=pd.DataFrame(columns=['abbr', 'week', 'year']),
        team_ranks={},
        player_group_ranks={},
        advanced_stat_cols={},
        big_play_stat_columns=[],
    )

    features = fe.load_features()
    # Spot-check a few expected keys are present from different parts
    assert 'overall_avg_passing_yards' in features
    assert 'dep_attempted_passes_for_passing_yards' in features  # may be NaN
    assert 'temperature' in features
    assert 'days_rest' in features
    assert 'over_under' in features
    assert 'spread' in features and 'is_favorite' in features
    assert 'starter' in features
    assert 'team_game_points' in features and 'opp_game_points' in features
