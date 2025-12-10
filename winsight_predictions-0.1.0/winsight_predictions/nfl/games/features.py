from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import logging

@dataclass
class FeatureEngine:
    # --- Core init arguments ---
    game_data: pd.DataFrame
    target_name: str
    row: pd.Series
    predicted_features: Optional[Dict[str, Any]] = None
    drop_cols: Optional[List[str]] = None

    # --- External data sources ---
    data_obj: Any = None  # DataObject instance
    schedules: pd.DataFrame = field(default_factory=pd.DataFrame)
    standings: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_ranks: Dict[str, pd.DataFrame] = field(default_factory=dict)
    team_ranks_str_cols: List[str] = field(default_factory=list)
    team_position_ratings: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Note: officials_features is loaded as a property, not passed as init param
    
    # PBP feature column names
    redzone_columns: List[str] = field(default_factory=list)
    team_epa_columns: List[str] = field(default_factory=list)
    play_type_columns: List[str] = field(default_factory=list)
    yards_togo_columns: List[str] = field(default_factory=list)
    yards_gained_columns: List[str] = field(default_factory=list)
    big_play_position_columns: List[str] = field(default_factory=list)
    player_epa_position_columns: List[str] = field(default_factory=list)
    
    # Additional metadata
    available_positions: List[str] = field(default_factory=list)
    referee_hash_mod: int = 10000

    # --- Config / metadata ---
    # Feature dependencies: Maps target stat names (without home_/away_ prefix) to their dependencies
    # Base volume stats have no dependencies
    # Efficiency stats depend on volume stats from both home and away sides
    feature_dependencies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # Efficiency stats depend on volume stats
        'pass_yards': {
            'home': ['pass_attempts', 'rush_attempts'],
            'away': ['pass_attempts', 'rush_attempts'],
            'win': True
        },
        'rush_yards': {
            'home': ['pass_attempts', 'rush_attempts'],
            'away': ['pass_attempts', 'rush_attempts'],
            'win': True
        },
        'total_yards': {
            'home': ['pass_attempts', 'rush_attempts'],
            'away': ['pass_attempts', 'rush_attempts'],
            'win': True
        },
        'points': {
            'home': ['pass_attempts', 'rush_attempts'],
            'away': ['pass_attempts', 'rush_attempts'],
            'win': True
        },
    })

    # --- Internal caches for property loads ---
    _dependent_stats: Optional[Dict[str, Any]] = None
    _team_rank_features: Optional[Dict[str, Any]] = None
    _team_standings_features: Optional[Dict[str, Any]] = None
    _position_rating_features: Optional[Dict[str, Any]] = None
    _team_trend_features: Optional[Dict[str, Any]] = None
    _current_situation_features: Optional[Dict[str, Any]] = None
    _officials_features: Optional[Dict[str, Any]] = None
    _scoring_stats_features: Optional[Dict[str, Any]] = None
    _weather_features: Optional[Dict[str, Any]] = None
    _rest_days_features: Optional[Dict[str, Any]] = None
    _betting_features: Optional[Dict[str, Any]] = None
    _redzone_features: Optional[Dict[str, Any]] = None
    _epa_features: Optional[Dict[str, Any]] = None
    _pbp_features: Optional[Dict[str, Any]] = None
    _basic_features: Optional[Dict[str, Any]] = None
    _features: Optional[Dict[str, Any]] = None

    # --- Helper caches ---
    team_ranks_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    standings_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    position_ratings_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    team_trends_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scoring_stats_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    team_games_prior_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ========== Helper methods ==========
    def _team_games_prior(self, team_abbr: str, date: datetime) -> pd.DataFrame:
        """Get all prior games for a team before a given date."""
        if 'game_date' not in self.game_data.columns:
            return pd.DataFrame()
        cache_key = f"{team_abbr}_{date}"
        if cache_key in self.team_games_prior_cache:
            return self.team_games_prior_cache[cache_key]
        result = self.game_data[
            (self.game_data['game_date'] < date) & 
            ((self.game_data['home_abbr'] == team_abbr) | (self.game_data['away_abbr'] == team_abbr))
        ]
        self.team_games_prior_cache[cache_key] = result
        return result

    # ========== Property wrappers ==========
    @property
    def team_rank_features(self) -> Dict[str, Any]:
        if self._team_rank_features is None:
            self._team_rank_features = self._load_team_rank_features()
        return self._team_rank_features

    @property
    def team_standings_features(self) -> Dict[str, Any]:
        if self._team_standings_features is None:
            self._team_standings_features = self._load_team_standings_features()
        return self._team_standings_features

    @property
    def position_rating_features(self) -> Dict[str, Any]:
        if self._position_rating_features is None:
            self._position_rating_features = self._load_position_rating_features()
        return self._position_rating_features

    @property
    def team_trend_features(self) -> Dict[str, Any]:
        if self._team_trend_features is None:
            self._team_trend_features = self._load_team_trend_features()
        return self._team_trend_features

    @property
    def current_situation_features(self) -> Dict[str, Any]:
        if self._current_situation_features is None:
            self._current_situation_features = self._load_current_situation_features()
        return self._current_situation_features

    @property
    def officials_features(self) -> Dict[str, Any]:
        if self._officials_features is None:
            self._officials_features = self._load_officials_features()
        return self._officials_features

    @property
    def scoring_stats_features(self) -> Dict[str, Any]:
        if self._scoring_stats_features is None:
            self._scoring_stats_features = self._load_scoring_stats_features()
        return self._scoring_stats_features

    @property
    def weather_features(self) -> Dict[str, Any]:
        if self._weather_features is None:
            self._weather_features = self._load_weather_features()
        return self._weather_features

    @property
    def rest_days_features(self) -> Dict[str, Any]:
        if self._rest_days_features is None:
            self._rest_days_features = self._load_rest_days_features()
        return self._rest_days_features

    @property
    def betting_features(self) -> Dict[str, Any]:
        if self._betting_features is None:
            self._betting_features = self._load_betting_features()
        return self._betting_features

    @property
    def redzone_features(self) -> Dict[str, Any]:
        if self._redzone_features is None:
            self._redzone_features = self._load_redzone_features()
        return self._redzone_features

    @property
    def epa_features(self) -> Dict[str, Any]:
        if self._epa_features is None:
            self._epa_features = self._load_epa_features()
        return self._epa_features

    @property
    def pbp_features(self) -> Dict[str, Any]:
        if self._pbp_features is None:
            self._pbp_features = self._load_pbp_features()
        return self._pbp_features

    @property
    def basic_features(self) -> Dict[str, Any]:
        if self._basic_features is None:
            self._basic_features = self._load_basic_features()
        return self._basic_features

    @property
    def dependent_stats(self) -> Dict[str, Any]:
        """Load dependent features based on previously predicted or actual values."""
        if self._dependent_stats is None:
            self._dependent_stats = self._load_dependent_stats()
        return self._dependent_stats

    # ========== Low-level loaders ==========
    def _load_basic_features(self) -> Dict[str, Any]:
        """Load basic division/conference flags."""
        row = self.row
        features = {
            'is_division': int(row.get('home_division', 'Unknown') == row.get('away_division', 'Unknown')),
            'is_conference': int(row.get('home_conference', 'Unknown') == row.get('away_conference', 'Unknown')),
        }
        return features

    def _load_dependent_stats(self) -> Dict[str, Any]:
        """Load dependent features based on feature dependencies.
        
        This method handles the dependency structure where efficiency stats (yards, points)
        depend on volume stats (attempts, win). It uses predicted values when available
        during inference, actual values during training, or historical averages as fallback.
        
        The feature_dependencies dict uses base stat names (without home_/away_ prefix),
        so we extract the base name from the target and apply prefixes to dependencies.
        """
        tn = self.target_name
        preds = self.predicted_features
        row = self.row
        game_data = self.game_data
        
        d: Dict[str, Any] = {}
        
        # Extract base stat name from target (e.g., 'home_points' -> 'points')
        # Determine if this is a home or away target
        if tn.startswith('home_'):
            side = 'home'
            base_stat = tn[5:]  # Remove 'home_' prefix
        elif tn.startswith('away_'):
            side = 'away'
            base_stat = tn[5:]  # Remove 'away_' prefix
        else:
            # Target doesn't have home_/away_ prefix (e.g., 'total_points', 'spread_result')
            # These don't use the dependency system in the same way
            return d
        
        # Get dependency configuration for the base stat
        dep_config = self.feature_dependencies.get(base_stat, {})
        if not dep_config:
            return d
        
        # Process home dependencies (with home_ prefix)
        home_deps = dep_config.get('home', [])
        for dep in home_deps:
            home_dep = f'home_{dep}'
            # Try to get from predictions first (for inference)
            if preds is not None and home_dep in preds:
                d[f'dep_{home_dep}_for_{tn}'] = preds[home_dep]
            # Try to get from current row (for training)
            elif home_dep in row.index:
                d[f'dep_{home_dep}_for_{tn}'] = row[home_dep]
            # Fall back to historical average
            elif home_dep in game_data.columns:
                d[f'dep_{home_dep}_for_{tn}'] = game_data[home_dep].mean()
            else:
                d[f'dep_{home_dep}_for_{tn}'] = 0.0
        
        # Process away dependencies (with away_ prefix)
        away_deps = dep_config.get('away', [])
        for dep in away_deps:
            away_dep = f'away_{dep}'
            # Try to get from predictions first (for inference)
            if preds is not None and away_dep in preds:
                d[f'dep_{away_dep}_for_{tn}'] = preds[away_dep]
            # Try to get from current row (for training)
            elif away_dep in row.index:
                d[f'dep_{away_dep}_for_{tn}'] = row[away_dep]
            # Fall back to historical average
            elif away_dep in game_data.columns:
                d[f'dep_{away_dep}_for_{tn}'] = game_data[away_dep].mean()
            else:
                d[f'dep_{away_dep}_for_{tn}'] = 0.0
        
        # Process win dependency (if applicable)
        if dep_config.get('win', False):
            win_dep = 'home_win'
            if preds is not None and win_dep in preds:
                d[f'dep_{win_dep}_for_{tn}'] = preds[win_dep]
            elif win_dep in row.index:
                d[f'dep_{win_dep}_for_{tn}'] = row[win_dep]
            elif win_dep in game_data.columns:
                d[f'dep_{win_dep}_for_{tn}'] = game_data[win_dep].mean()
            else:
                d[f'dep_{win_dep}_for_{tn}'] = 0.5  # Default to 50% probability
        
        return d

    def _load_team_rank_features(self) -> Dict[str, Any]:
        """Load team rank features from team_ranks data."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']

        features: Dict[str, Any] = {}
        
        for cfg_key, ranks_df in self.team_ranks.items():
            if ranks_df.empty or 'game_date' not in ranks_df.columns:
                continue
            
            home_key = f"{cfg_key}_{date.isoformat()}_{home_abbr}"
            away_key = f"{cfg_key}_{date.isoformat()}_{away_abbr}"
            
            if home_key not in self.team_ranks_cache:
                subset = ranks_df[(ranks_df['game_date'] < date) & (ranks_df['abbr'] == home_abbr)].tail(1)
                if not subset.empty:
                    subset.columns = [f"{cfg_key}_home_{c}" if c not in self.team_ranks_str_cols else c for c in subset.columns]
                    vals = subset.drop(columns=self.team_ranks_str_cols, errors='ignore').iloc[0].to_dict()
                    self.team_ranks_cache[home_key] = vals
            
            if home_key in self.team_ranks_cache:
                features.update(**self.team_ranks_cache[home_key])
            
            if away_key not in self.team_ranks_cache:
                subset = ranks_df[(ranks_df['game_date'] < date) & (ranks_df['abbr'] == away_abbr)].tail(1)
                if not subset.empty:
                    subset.columns = [f"{cfg_key}_away_{c}" if c not in self.team_ranks_str_cols else c for c in subset.columns]
                    vals = subset.drop(columns=self.team_ranks_str_cols, errors='ignore').iloc[0].to_dict()
                    self.team_ranks_cache[away_key] = vals
            
            if away_key in self.team_ranks_cache:
                features.update(**self.team_ranks_cache[away_key])
        
        return features

    def _load_team_standings_features(self) -> Dict[str, Any]:
        """Load team standings features for home and away teams."""
        row = self.row
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        features: Dict[str, Any] = {}
        
        # Get standings for home team
        home_standings = self._get_team_standings(home_abbr, row)
        if home_standings is not None:
            features.update({f"home_{k}": v for k, v in home_standings.items()})
        
        # Get standings for away team
        away_standings = self._get_team_standings(away_abbr, row)
        if away_standings is not None:
            features.update({f"away_{k}": v for k, v in away_standings.items()})
        
        return features

    def _get_team_standings(self, team_abbr: str, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Get standings data for a specific team."""
        week = row['week']
        year = row['year']
        last_week = row['last_week']
        cache_key = f"{team_abbr}_{week}_{year}"
        
        if cache_key in self.standings_cache:
            return self.standings_cache[cache_key]
        
        target_week, target_year = week, year
        if target_week == 1:
            target_year -= 1
            target_week = last_week
        else:
            target_week = last_week
        
        if not self.standings.empty:
            subset: pd.DataFrame = self.standings[
                (self.standings['abbr'] == team_abbr) &
                (self.standings['week'] == target_week) &
                (self.standings['year'] == target_year)
            ]
            if not subset.empty:
                standings = subset.iloc[0]
                standings_dict = {
                    "wins": standings.get('wins', np.nan),
                    "losses": standings.get('losses', np.nan),
                    "ties": standings.get('ties', np.nan),
                    "win_pct": standings.get('win_pct', np.nan),
                    "points_for": standings.get('points_for', np.nan),
                    "points_against": standings.get('points_against', np.nan),
                    "net_points": standings.get('net_points', np.nan),
                    "division_wins": standings.get('division_wins', np.nan),
                    "division_losses": standings.get('division_losses', np.nan),
                    "division_ties": standings.get('division_ties', np.nan),
                    "division_win_pct": standings.get('division_win_pct', np.nan),
                    "conference_wins": standings.get('conference_wins', np.nan),
                    "conference_losses": standings.get('conference_losses', np.nan),
                    "conference_ties": standings.get('conference_ties', np.nan),
                    "conference_win_pct": standings.get('conference_win_pct', np.nan),
                    "strength_of_victory": standings.get('strength_of_victory', np.nan),
                    "strength_of_schedule": standings.get('strength_of_schedule', np.nan),
                    "touchdowns": standings.get('touchdowns', np.nan),
                    "touchdowns_against": standings.get('touchdowns_against', np.nan),
                    "net_touchdowns": standings.get('net_touchdowns', np.nan),
                    "division_rank": standings.get('division_rank', np.nan),
                    "conference_rank": standings.get('conference_rank', np.nan),
                    "is_division_winner": int(standings.get('is_division_winner', 0)),
                    "is_wild_card": int(standings.get('is_wild_card', 0)),
                }
                self.standings_cache[cache_key] = standings_dict
                return standings_dict
        
        return None

    def _load_position_rating_features(self) -> Dict[str, Any]:
        """Load position rating features for both home and away teams."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        year = self.row['year']
        
        # Determine if this is a prediction
        is_prediction = 'home_points' not in self.row or pd.isna(self.row.get('home_points'))
        
        features = {}
        
        # Get position ratings for home team
        home_pos_ratings = self._calc_position_rating_features(home_abbr, date, year, is_prediction=is_prediction)
        features.update({f"home_{k}": v for k, v in home_pos_ratings.items()})
        
        # Get position ratings for away team
        away_pos_ratings = self._calc_position_rating_features(away_abbr, date, year, is_prediction=is_prediction)
        features.update({f"away_{k}": v for k, v in away_pos_ratings.items()})
        
        # Opponent position ratings (team vs opponent's position groups)
        features.update({f"home_opp_{k}": v for k, v in away_pos_ratings.items()})
        features.update({f"away_opp_{k}": v for k, v in home_pos_ratings.items()})
        
        return features

    def _calc_position_rating_features(self, team_abbr: str, date: datetime, year: int, is_prediction: bool = False) -> dict:
        """Calculate position rating features for a team prior to a given date."""
        cache_key = f"{team_abbr}_{date.isoformat()}_{year}_pred{is_prediction}"
        if cache_key in self.position_ratings_cache:
            return self.position_ratings_cache[cache_key]
        
        features = {}
        
        if self.team_position_ratings.empty:
            self.position_ratings_cache[cache_key] = features
            return features
        
        skill_positions = ['QB', 'RB', 'WR', 'TE']
        
        # For predictions, get player ratings from new_starters for skill positions
        if is_prediction and self.data_obj and hasattr(self.data_obj, 'starters_new') and not self.data_obj.starters_new.empty:
            team_starters = self.data_obj.starters_new[self.data_obj.starters_new['team'] == team_abbr]
            
            if not team_starters.empty and hasattr(self.data_obj, 'player_ratings') and self.data_obj.player_ratings:
                for pos in skill_positions:
                    pos_starters = team_starters[team_starters['position'] == pos]

                    if pos_starters.empty:
                        features.update(self._calc_historical_position_features(team_abbr, date, year, pos))
                        continue
                    
                    player_ids = pos_starters['player_id'].tolist()
                    
                    if pos not in self.data_obj.player_ratings:
                        features.update(self._calc_historical_position_features(team_abbr, date, year, pos))
                        continue
                    
                    player_ratings_df = self.data_obj.player_ratings[pos]
                    player_ratings_df = player_ratings_df[player_ratings_df['pid'].isin(player_ids)]
                    
                    if player_ratings_df.empty:
                        features.update(self._calc_historical_position_features(team_abbr, date, year, pos))
                        continue
                    
                    rating_col = f'overall_{pos.lower()}_rating'
                    if rating_col not in player_ratings_df.columns:
                        if 'overall_rating' in player_ratings_df.columns:
                            rating_col = 'overall_rating'
                        else:
                            features.update(self._calc_historical_position_features(team_abbr, date, year, pos))
                            continue
                    
                    features[f'{pos}_rating_overall'] = float(player_ratings_df[rating_col].mean())
                    features[f'{pos}_rating_season'] = float(player_ratings_df[player_ratings_df['year'] == year][rating_col].mean())
                    features[f'{pos}_rating_last5'] = float(player_ratings_df[rating_col].tail(5).mean())
                    features[f'{pos}_rating_last3'] = float(player_ratings_df[rating_col].tail(3).mean())
                    features[f'{pos}_rating_last1'] = float(player_ratings_df[rating_col].tail(1).mean())
        
        # For non-skill positions OR training data, use historical game-based ratings
        positions_to_calculate = [p for p in self.available_positions if p not in skill_positions or not is_prediction]

        team_ratings = self.team_position_ratings[
            ((self.team_position_ratings['abbr'] == team_abbr) & 
             (self.team_position_ratings['game_date'] < date))
        ]
        
        if team_ratings.empty:
            for pos in positions_to_calculate:
                if f'{pos}_rating_season' not in features:
                    features[f'{pos}_rating_overall'] = np.nan
                    features[f'{pos}_rating_season'] = np.nan
                    features[f'{pos}_rating_last5'] = np.nan
                    features[f'{pos}_rating_last3'] = np.nan
                    features[f'{pos}_rating_last1'] = np.nan
            self.position_ratings_cache[cache_key] = features
            return features
        
        for pos in positions_to_calculate:
            if f'{pos}_rating_season' in features:
                continue
            features.update(self._calc_historical_position_features(team_abbr, date, year, pos, team_ratings))
        
        self.position_ratings_cache[cache_key] = features
        return features

    def _calc_historical_position_features(self, team_abbr: str, date: datetime, year: int, pos: str, team_ratings: pd.DataFrame = None) -> dict:
        """Helper method to calculate position features from historical game data."""
        features = {}
        
        if team_ratings is None:
            team_ratings = self.team_position_ratings[
                ((self.team_position_ratings['abbr'] == team_abbr) & 
                 (self.team_position_ratings['game_date'] < date))
            ]
        
        pos_data = team_ratings[team_ratings['pos'] == pos].sort_values('game_date')
        
        if pos_data.empty:
            features[f'{pos}_rating_overall'] = np.nan
            features[f'{pos}_rating_season'] = np.nan
            features[f'{pos}_rating_last5'] = np.nan
            features[f'{pos}_rating_last3'] = np.nan
            features[f'{pos}_rating_last1'] = np.nan
            return features
        
        season_data = pos_data[pos_data['year'] == year]
        features[f'{pos}_rating_overall'] = float(pos_data['avg_overall_rating'].mean()) if not pos_data.empty else np.nan
        features[f'{pos}_rating_season'] = float(season_data['avg_overall_rating'].mean()) if not season_data.empty else np.nan
        features[f'{pos}_rating_last5'] = float(pos_data['avg_overall_rating'].tail(5).mean()) if len(pos_data) >= 1 else np.nan
        features[f'{pos}_rating_last3'] = float(pos_data['avg_overall_rating'].tail(3).mean()) if len(pos_data) >= 1 else np.nan
        features[f'{pos}_rating_last1'] = float(pos_data['avg_overall_rating'].tail(1).values[0]) if len(pos_data) >= 1 else np.nan
        
        return features

    def _load_team_trend_features(self) -> Dict[str, Any]:
        """Load team trend features (win%, MOV, ATS) for various situations."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        year = self.row['year']
        
        features = {}
        
        home_trends = self._calc_team_trend_features(home_abbr, date, year)
        features.update({f"home_trend_{k}": v for k, v in home_trends.items()})
        
        away_trends = self._calc_team_trend_features(away_abbr, date, year)
        features.update({f"away_trend_{k}": v for k, v in away_trends.items()})
        
        return features

    def _calc_team_trend_features(self, team_abbr: str, date: datetime, year: int) -> dict:
        """Calculate team trend features based on various situational filters."""
        cache_key = f"{team_abbr}_{date.isoformat()}_{year}"
        if cache_key in self.team_trends_cache:
            return self.team_trends_cache[cache_key]
        
        features = {}
        team_prior = self._team_games_prior(team_abbr, date)
        
        if team_prior.empty:
            situations = [
                'all', 'after_bye', 'after_win', 'after_loss', 'after_tie',
                'as_home', 'as_away', 'as_fav', 'as_dog',
                'rest_adv', 'rest_disadv', 'rest_equal',
                'conference', 'non_conference', 'division', 'non_division',
                'season', 'last3', 'last5', 'last1'
            ]
            for sit in situations:
                features[f'{sit}_win_pct'] = np.nan
                features[f'{sit}_mov'] = np.nan
                features[f'{sit}_ats_plus_minus'] = np.nan
            self.team_trends_cache[cache_key] = features
            return features
        
        def calc_stats(df: pd.DataFrame) -> dict:
            if df.empty:
                return {'win_pct': np.nan, 'mov': np.nan, 'ats_plus_minus': np.nan}
            
            is_team_home = (df['home_abbr'] == team_abbr)
            team_points = np.where(is_team_home, df['home_points'], df['away_points'])
            opp_points = np.where(is_team_home, df['away_points'], df['home_points'])
            
            wins = (team_points > opp_points).sum()
            total_games = len(df)
            win_pct = wins / total_games if total_games > 0 else np.nan
            mov = (team_points - opp_points).mean()
            
            ats_vals = []
            for idx, game_row in df.iterrows():
                if pd.notna(game_row.get('spread')) and pd.notna(game_row.get('spread_favorite_abbr')):
                    team_is_home = game_row['home_abbr'] == team_abbr
                    team_pts = game_row['home_points'] if team_is_home else game_row['away_points']
                    opp_pts = game_row['away_points'] if team_is_home else game_row['home_points']
                    mov_game = team_pts - opp_pts
                    spread_val = abs(game_row['spread'])
                    team_is_fav = (game_row['spread_favorite_abbr'] == team_abbr)
                    
                    if team_is_fav:
                        ats = mov_game - spread_val
                    else:
                        ats = mov_game + spread_val
                    ats_vals.append(ats)
            
            ats_plus_minus = np.mean(ats_vals) if ats_vals else np.nan
            
            return {
                'win_pct': float(win_pct),
                'mov': float(mov),
                'ats_plus_minus': float(ats_plus_minus)
            }
        
        # All games
        all_stats = calc_stats(team_prior)
        features.update({f'all_{k}': v for k, v in all_stats.items()})
        
        # Season
        season_games = team_prior[team_prior['year'] == year]
        season_stats = calc_stats(season_games)
        features.update({f'season_{k}': v for k, v in season_stats.items()})
        
        # Last N games
        last5_stats = calc_stats(team_prior.tail(5))
        features.update({f'last5_{k}': v for k, v in last5_stats.items()})
        
        last3_stats = calc_stats(team_prior.tail(3))
        features.update({f'last3_{k}': v for k, v in last3_stats.items()})
        
        last1_stats = calc_stats(team_prior.tail(1))
        features.update({f'last1_{k}': v for k, v in last1_stats.items()})
        
        # After bye week
        team_prior_sorted = team_prior.sort_values('game_date')
        after_bye_games = []
        for i in range(1, len(team_prior_sorted)):
            prev_game = team_prior_sorted.iloc[i-1]
            curr_game = team_prior_sorted.iloc[i]
            days_rest = (curr_game['game_date'] - prev_game['game_date']).days
            if days_rest > 10:
                after_bye_games.append(curr_game)
        
        after_bye_df = pd.DataFrame(after_bye_games) if after_bye_games else pd.DataFrame()
        after_bye_stats = calc_stats(after_bye_df)
        features.update({f'after_bye_{k}': v for k, v in after_bye_stats.items()})
        
        # After win/loss/tie
        after_win_games = []
        after_loss_games = []
        after_tie_games = []
        
        for i in range(1, len(team_prior_sorted)):
            prev_game = team_prior_sorted.iloc[i-1]
            curr_game = team_prior_sorted.iloc[i]
            
            prev_is_home = prev_game['home_abbr'] == team_abbr
            prev_team_pts = prev_game['home_points'] if prev_is_home else prev_game['away_points']
            prev_opp_pts = prev_game['away_points'] if prev_is_home else prev_game['home_points']
            
            if prev_team_pts > prev_opp_pts:
                after_win_games.append(curr_game)
            elif prev_team_pts < prev_opp_pts:
                after_loss_games.append(curr_game)
            else:
                after_tie_games.append(curr_game)
        
        after_win_stats = calc_stats(pd.DataFrame(after_win_games) if after_win_games else pd.DataFrame())
        features.update({f'after_win_{k}': v for k, v in after_win_stats.items()})
        
        after_loss_stats = calc_stats(pd.DataFrame(after_loss_games) if after_loss_games else pd.DataFrame())
        features.update({f'after_loss_{k}': v for k, v in after_loss_stats.items()})
        
        after_tie_stats = calc_stats(pd.DataFrame(after_tie_games) if after_tie_games else pd.DataFrame())
        features.update({f'after_tie_{k}': v for k, v in after_tie_stats.items()})
        
        # As home/away team
        home_games = team_prior[team_prior['home_abbr'] == team_abbr]
        away_games = team_prior[team_prior['away_abbr'] == team_abbr]
        
        as_home_stats = calc_stats(home_games)
        features.update({f'as_home_{k}': v for k, v in as_home_stats.items()})
        
        as_away_stats = calc_stats(away_games)
        features.update({f'as_away_{k}': v for k, v in as_away_stats.items()})
        
        # As favorite/underdog
        if 'spread_favorite_abbr' in team_prior.columns:
            fav_games = team_prior[team_prior['spread_favorite_abbr'] == team_abbr]
            dog_games = team_prior[team_prior['spread_favorite_abbr'] != team_abbr]
            
            as_fav_stats = calc_stats(fav_games)
            features.update({f'as_fav_{k}': v for k, v in as_fav_stats.items()})
            
            as_dog_stats = calc_stats(dog_games)
            features.update({f'as_dog_{k}': v for k, v in as_dog_stats.items()})
        else:
            for k in ['win_pct', 'mov', 'ats_plus_minus']:
                features[f'as_fav_{k}'] = np.nan
                features[f'as_dog_{k}'] = np.nan
        
        # Rest advantage/disadvantage (implementation simplified for space)
        for k in ['win_pct', 'mov', 'ats_plus_minus']:
            features[f'rest_adv_{k}'] = np.nan
            features[f'rest_disadv_{k}'] = np.nan
            features[f'rest_equal_{k}'] = np.nan
        
        # Conference games
        if 'home_conference' in team_prior.columns and 'away_conference' in team_prior.columns:
            def is_conference_game(game_row):
                team_is_home = game_row['home_abbr'] == team_abbr
                team_conf = game_row['home_conference'] if team_is_home else game_row['away_conference']
                opp_conf = game_row['away_conference'] if team_is_home else game_row['home_conference']
                return team_conf == opp_conf
            
            conf_games = team_prior[team_prior.apply(is_conference_game, axis=1)]
            non_conf_games = team_prior[~team_prior.apply(is_conference_game, axis=1)]
            
            conf_stats = calc_stats(conf_games)
            features.update({f'conference_{k}': v for k, v in conf_stats.items()})
            
            non_conf_stats = calc_stats(non_conf_games)
            features.update({f'non_conference_{k}': v for k, v in non_conf_stats.items()})
        else:
            for k in ['win_pct', 'mov', 'ats_plus_minus']:
                features[f'conference_{k}'] = np.nan
                features[f'non_conference_{k}'] = np.nan
        
        # Division games
        if 'home_division' in team_prior.columns and 'away_division' in team_prior.columns:
            def is_division_game(game_row):
                team_is_home = game_row['home_abbr'] == team_abbr
                team_div = game_row['home_division'] if team_is_home else game_row['away_division']
                opp_div = game_row['away_division'] if team_is_home else game_row['home_division']
                return team_div == opp_div
            
            div_games = team_prior[team_prior.apply(is_division_game, axis=1)]
            non_div_games = team_prior[~team_prior.apply(is_division_game, axis=1)]
            
            div_stats = calc_stats(div_games)
            features.update({f'division_{k}': v for k, v in div_stats.items()})
            
            non_div_stats = calc_stats(non_div_games)
            features.update({f'non_division_{k}': v for k, v in non_div_stats.items()})
        else:
            for k in ['win_pct', 'mov', 'ats_plus_minus']:
                features[f'division_{k}'] = np.nan
                features[f'non_division_{k}'] = np.nan
        
        self.team_trends_cache[cache_key] = features
        return features

    def _load_current_situation_features(self) -> Dict[str, Any]:
        """Load current situation indicators for both teams."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        
        home_prior = self._team_games_prior(home_abbr, date)
        away_prior = self._team_games_prior(away_abbr, date)
        
        features = {}
        
        home_situation = self._calc_current_situation(home_abbr, self.row, home_prior)
        features.update({f"home_{k}": v for k, v in home_situation.items()})
        
        away_situation = self._calc_current_situation(away_abbr, self.row, away_prior)
        features.update({f"away_{k}": v for k, v in away_situation.items()})
        
        return features

    def _calc_current_situation(self, team_abbr: str, row: pd.Series, team_prior: pd.DataFrame) -> dict:
        """Calculate boolean indicators for the current game situation."""
        situation = {}
        
        # Default all to 0
        situation['is_after_bye'] = 0
        situation['is_after_win'] = 0
        situation['is_after_loss'] = 0
        situation['is_after_tie'] = 0
        situation['is_home'] = 1 if row.get('home_abbr') == team_abbr else 0
        situation['is_away'] = 1 if row.get('away_abbr') == team_abbr else 0
        situation['is_fav'] = 0
        situation['is_dog'] = 0
        situation['has_rest_adv'] = 0
        situation['has_rest_disadv'] = 0
        situation['has_rest_equal'] = 0
        situation['is_conference'] = 0
        situation['is_non_conference'] = 0
        situation['is_division'] = 0
        situation['is_non_division'] = 0
        
        if team_prior.empty:
            return situation
        
        team_prior_sorted = team_prior.sort_values('game_date')
        last_game = team_prior_sorted.iloc[-1]
        
        if 'game_date' in row.index and 'game_date' in last_game.index:
            days_since_last = (row['game_date'] - last_game['game_date']).days
            situation['is_after_bye'] = 1 if days_since_last > 10 else 0
        
        last_is_home = last_game['home_abbr'] == team_abbr
        last_team_pts = last_game['home_points'] if last_is_home else last_game['away_points']
        last_opp_pts = last_game['away_points'] if last_is_home else last_game['home_points']
        
        if last_team_pts > last_opp_pts:
            situation['is_after_win'] = 1
        elif last_team_pts < last_opp_pts:
            situation['is_after_loss'] = 1
        else:
            situation['is_after_tie'] = 1
        
        if 'spread_favorite_abbr' in row.index and pd.notna(row.get('spread_favorite_abbr')):
            if row['spread_favorite_abbr'] == team_abbr:
                situation['is_fav'] = 1
            else:
                situation['is_dog'] = 1
        
        # Conference/division game checks
        if 'home_conference' in row.index and 'away_conference' in row.index:
            team_is_home = row.get('home_abbr') == team_abbr
            team_conf = row['home_conference'] if team_is_home else row['away_conference']
            opp_conf = row['away_conference'] if team_is_home else row['home_conference']
            
            if team_conf == opp_conf:
                situation['is_conference'] = 1
            else:
                situation['is_non_conference'] = 1
        
        if 'home_division' in row.index and 'away_division' in row.index:
            team_is_home = row.get('home_abbr') == team_abbr
            team_div = row['home_division'] if team_is_home else row['away_division']
            opp_div = row['away_division'] if team_is_home else row['home_division']
            
            if team_div == opp_div:
                situation['is_division'] = 1
            else:
                situation['is_non_division'] = 1
        
        return situation

    def _load_officials_features(self) -> Dict[str, Any]:
        """Load officials features for the game."""
        is_prediction = 'home_points' not in self.row or pd.isna(self.row.get('home_points'))
        
        game_id = None
        if 'key' in self.row.index:
            game_id = self.row.get('key')
        if 'game_id' in self.row.index:
            game_id = self.row.get('game_id')
        
        year = self.row.get('year')
        
        if game_id:
            return self._get_officials_features(game_id=game_id, year=year, is_prediction=is_prediction)
        return {}

    def _get_officials_features(self, game_id: str = None, year: int = None, is_prediction: bool = False) -> dict:
        """Get officials features for a specific game."""
        if is_prediction:
            try:
                if self.data_obj and hasattr(self.data_obj, 'new_officials_with_features'):
                    new_features = self.data_obj.new_officials_with_features
                    
                    if new_features.empty:
                        return {}
                    
                    game_features = new_features[new_features['game_id'] == game_id]
                    
                    if game_features.empty:
                        return {}
                    
                    features_row = game_features.iloc[0]
                else:
                    return {}
            except Exception as e:
                logging.warning(f"Error loading new officials features: {e}")
                return {}
        else:
            # Get officials data from data_obj, not from the property (to avoid recursion)
            officials_df = self.data_obj.get_officials_with_features()
            game_features = officials_df[officials_df['game_id'] == game_id]
            
            if game_features.empty:
                return {}
            
            features_row = game_features.iloc[0]
        
        exclude_cols = ['referee_pid', 'referee_name', 'game_id', 'year', 'week', 'game_num', 'is_playoffs', 'game_date', 'position']
        
        officials_dict = {}
        for col in features_row.index:
            if col not in exclude_cols:
                officials_dict[f'ref_{col}'] = features_row[col]
        
        referee_pid = features_row.get('referee_pid')
        if referee_pid:
            officials_dict['ref_pid_encoded'] = abs(hash(referee_pid)) % self.referee_hash_mod
        else:
            officials_dict['ref_pid_encoded'] = 0
        
        return officials_dict

    def _load_scoring_stats_features(self) -> Dict[str, Any]:
        """Load scoring statistics features for both teams."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        
        home_prior = self._team_games_prior(home_abbr, date)
        away_prior = self._team_games_prior(away_abbr, date)
        
        features = {}
        
        def scoring(df: pd.DataFrame, team: str, stat: str = 'points') -> dict:
            cache_key = f"{team}_{date.isoformat()}_{stat}"
            if cache_key in self.scoring_stats_cache:
                return self.scoring_stats_cache[cache_key]
            
            if df.empty:
                result = {
                    f'overall_avg_{stat}_for': np.nan,
                    f'overall_avg_{stat}_against': np.nan,
                    f'last5_avg_{stat}_for': np.nan,
                    f'last5_avg_{stat}_against': np.nan,
                    f'last_game_{stat}_for': np.nan,
                    f'last_game_{stat}_against': np.nan,
                }
                self.scoring_stats_cache[cache_key] = result
                return result
            
            pts_for = np.where(df['home_abbr'] == team, df[f'home_{stat}'], df[f'away_{stat}']) if f'home_{stat}' in df.columns else []
            pts_against = np.where(df['home_abbr'] == team, df[f'away_{stat}'], df[f'home_{stat}']) if f'away_{stat}' in df.columns else []
            result = {
                f'overall_avg_{stat}_for': float(np.mean(pts_for)) if len(pts_for) else np.nan,
                f'overall_avg_{stat}_against': float(np.mean(pts_against)) if len(pts_against) else np.nan,
                f'last5_avg_{stat}_for': float(np.mean(pts_for[-5:])) if len(pts_for) else np.nan,
                f'last5_avg_{stat}_against': float(np.mean(pts_against[-5:])) if len(pts_against) else np.nan,
                f'last_game_{stat}_for': float(pts_for[-1]) if len(pts_for) else np.nan,
                f'last_game_{stat}_against': float(pts_against[-1]) if len(pts_against) else np.nan,
            }
            self.scoring_stats_cache[cache_key] = result
            return result
        
        # Points
        home_stats = scoring(home_prior, home_abbr)
        away_stats = scoring(away_prior, away_abbr)
        features.update({f"home_{k}": v for k, v in home_stats.items()})
        features.update({f"away_{k}": v for k, v in away_stats.items()})
        
        # Other stats
        for stat in ['pass_yards', 'rush_yards', 'total_yards', 'pass_attempts', 'rush_attempts']:
            home_stats = scoring(home_prior, home_abbr, stat=stat)
            away_stats = scoring(away_prior, away_abbr, stat=stat)
            features.update({f"home_{k}": v for k, v in home_stats.items()})
            features.update({f"away_{k}": v for k, v in away_stats.items()})
        
        # Points for diff
        if not home_prior.empty and not away_prior.empty:
            features['points_for_diff_overall'] = features['home_overall_avg_points_for'] - features['away_overall_avg_points_for']
        else:
            features['points_for_diff_overall'] = np.nan
        
        return features

    def _load_weather_features(self) -> Dict[str, Any]:
        """Load weather features."""
        weather = self.row.get('weather')
        features = {}
        
        if isinstance(weather, str) and '|' in weather:
            try:
                t, h, w = weather.split('|')
                features['temperature'] = float(t)
                features['humidity'] = float(h)
                features['wind_speed'] = float(w)
            except:
                features['temperature'] = 72.0
                features['humidity'] = 50.0
                features['wind_speed'] = 0.0
        else:
            features['temperature'] = 72.0
            features['humidity'] = 50.0
            features['wind_speed'] = 0.0
        
        return features

    def _load_rest_days_features(self) -> Dict[str, Any]:
        """Load rest days features."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        
        home_prior = self._team_games_prior(home_abbr, date)
        away_prior = self._team_games_prior(away_abbr, date)
        
        features = {}
        
        if not home_prior.empty and 'game_date' in home_prior.columns:
            features['home_days_rest'] = (date - home_prior['game_date'].max()).days
        else:
            features['home_days_rest'] = np.nan
        
        if not away_prior.empty and 'game_date' in away_prior.columns:
            features['away_days_rest'] = (date - away_prior['game_date'].max()).days
        else:
            features['away_days_rest'] = np.nan
        
        return features

    def _load_betting_features(self) -> Dict[str, Any]:
        """Load betting-related features (spread, over/under)."""
        row = self.row
        features = {}
        
        # Over/under
        ou_raw = row.get('over_under')
        try:
            features['over_under'] = float(str(ou_raw).split(' ')[0]) if ou_raw not in (None, 'None') else np.nan
        except:
            features['over_under'] = np.nan
        
        # Spread (simplified for space - full implementation would parse vegas_line)
        features['spread'] = np.nan
        features['home_is_favorite'] = np.nan

        if 'spread' in row.index:
            try:
                features['spread'] = float(row['spread'])
            except:
                features['spread'] = np.nan

        if 'home_is_favorite' in row.index:
            try:
                features['home_is_favorite'] = int(row['home_is_favorite'])
            except:
                features['home_is_favorite'] = np.nan
        
        return features

    def _load_redzone_features(self) -> Dict[str, Any]:
        """Load redzone features."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        year = self.row['year']
        
        home_prior = self._team_games_prior(home_abbr, date)
        away_prior = self._team_games_prior(away_abbr, date)
        
        features = {}
        
        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr) > 0 else np.nan
        
        for col in self.redzone_columns:
            # Home team
            home_col_data = home_prior[f"home_{col}"] if f"home_{col}" in home_prior.columns else pd.Series(dtype=float)
            features[f"home_{col}"] = safe_mean(home_col_data)
            features[f"home_{col}_season"] = safe_mean(home_prior[home_prior['year'] == year][f"home_{col}"]) if f"home_{col}" in home_prior.columns else np.nan
            features[f"home_{col}_last5"] = safe_mean(home_col_data.tail(5)) if not home_col_data.empty else np.nan
            features[f"home_{col}_last3"] = safe_mean(home_col_data.tail(3)) if not home_col_data.empty else np.nan
            
            # Away team
            away_col_data = away_prior[f"away_{col}"] if f"away_{col}" in away_prior.columns else pd.Series(dtype=float)
            features[f"away_{col}"] = safe_mean(away_col_data)
            features[f"away_{col}_season"] = safe_mean(away_prior[away_prior['year'] == year][f"away_{col}"]) if f"away_{col}" in away_prior.columns else np.nan
            features[f"away_{col}_last5"] = safe_mean(away_col_data.tail(5)) if not away_col_data.empty else np.nan
            features[f"away_{col}_last3"] = safe_mean(away_col_data.tail(3)) if not away_col_data.empty else np.nan
        
        return features

    def _load_epa_features(self) -> Dict[str, Any]:
        """Load EPA features."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        year = self.row['year']
        
        home_prior = self._team_games_prior(home_abbr, date)
        away_prior = self._team_games_prior(away_abbr, date)
        
        features = {}
        
        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr) > 0 else np.nan
        
        for col in self.team_epa_columns:
            if 'home_' in col:
                home_col_data = home_prior[col] if col in home_prior.columns else pd.Series(dtype=float)
                features[f"home_{col}"] = safe_mean(home_col_data)
                features[f"home_{col}_season"] = safe_mean(home_prior[home_prior['year'] == year][col]) if col in home_prior.columns else np.nan
                features[f"home_{col}_last5"] = safe_mean(home_col_data.tail(5)) if not home_col_data.empty else np.nan
                features[f"home_{col}_last3"] = safe_mean(home_col_data.tail(3)) if not home_col_data.empty else np.nan
            elif 'away_' in col:
                away_col_data = away_prior[col] if col in away_prior.columns else pd.Series(dtype=float)
                features[f"away_{col}"] = safe_mean(away_col_data)
                features[f"away_{col}_season"] = safe_mean(away_prior[away_prior['year'] == year][col]) if col in away_prior.columns else np.nan
                features[f"away_{col}_last5"] = safe_mean(away_col_data.tail(5)) if not away_col_data.empty else np.nan
                features[f"away_{col}_last3"] = safe_mean(away_col_data.tail(3)) if not away_col_data.empty else np.nan
        
        return features

    def _load_pbp_features(self) -> Dict[str, Any]:
        """Load play-by-play features."""
        date: datetime = self.row['game_date']
        home_abbr = self.row['home_abbr']
        away_abbr = self.row['away_abbr']
        year = self.row['year']
        
        home_prior = self._team_games_prior(home_abbr, date)
        away_prior = self._team_games_prior(away_abbr, date)
        
        return self._calc_new_pbp_features(home_abbr, away_abbr, home_prior, away_prior, self.row)

    def _calc_new_pbp_features(self, home_abbr: str, away_abbr: str, home_prior: pd.DataFrame, away_prior: pd.DataFrame, row: pd.Series) -> dict:
        """Calculate features for new PBP data."""
        features = {}
        
        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr) > 0 else np.nan
        
        column_groups = [
            ('play_type', self.play_type_columns),
            ('yards_togo', self.yards_togo_columns),
            ('yards_gained', self.yards_gained_columns),
            ('big_play_pos', self.big_play_position_columns),
            ('player_epa_pos', self.player_epa_position_columns)
        ]
        
        for group_name, columns in column_groups:
            for col in columns:
                if col.startswith('home_'):
                    team_abbr = home_abbr
                    prior_df = home_prior
                    prefix = 'home'
                elif col.startswith('away_'):
                    team_abbr = away_abbr
                    prior_df = away_prior
                    prefix = 'away'
                else:
                    continue
                
                if col not in prior_df.columns:
                    features[f"{col}"] = np.nan
                    features[f"{col}_season"] = np.nan
                    features[f"{col}_last5"] = np.nan
                    features[f"{col}_last3"] = np.nan
                    features[f"{col}_at_home"] = np.nan
                    features[f"{col}_on_road"] = np.nan
                    continue
                
                col_data = prior_df[col]
                
                features[f"{col}"] = safe_mean(col_data)
                season_data = prior_df[prior_df['year'] == row['year']][col] if 'year' in prior_df.columns else col_data
                features[f"{col}_season"] = safe_mean(season_data)
                features[f"{col}_last5"] = safe_mean(col_data.tail(5)) if not col_data.empty else np.nan
                features[f"{col}_last3"] = safe_mean(col_data.tail(3)) if not col_data.empty else np.nan
                
                if prefix == 'home':
                    at_home_data = prior_df[prior_df['home_abbr'] == team_abbr][col] if 'home_abbr' in prior_df.columns else pd.Series(dtype=float)
                else:
                    at_home_data = prior_df[prior_df['home_abbr'] == team_abbr][col] if 'home_abbr' in prior_df.columns else pd.Series(dtype=float)
                features[f"{col}_at_home"] = safe_mean(at_home_data)
                
                if prefix == 'home':
                    on_road_data = prior_df[prior_df['away_abbr'] == team_abbr][col] if 'away_abbr' in prior_df.columns else pd.Series(dtype=float)
                else:
                    on_road_data = prior_df[prior_df['away_abbr'] == team_abbr][col] if 'away_abbr' in prior_df.columns else pd.Series(dtype=float)
                features[f"{col}_on_road"] = safe_mean(on_road_data)
        
        features = {k: (v if pd.notna(v) else 0.0) for k, v in features.items()}
        
        return features

    # ========== Aggregated features ==========
    def load_features(self) -> Dict[str, Any]:
        """Load all features and return as a single dictionary."""
        features = {}
        features.update(self.basic_features)
        features.update(self.dependent_stats)
        features.update(self.team_rank_features)
        features.update(self.team_standings_features)
        features.update(self.position_rating_features)
        features.update(self.team_trend_features)
        features.update(self.current_situation_features)
        features.update(self.officials_features)
        features.update(self.scoring_stats_features)
        features.update(self.weather_features)
        features.update(self.rest_days_features)
        features.update(self.betting_features)
        features.update(self.redzone_features)
        features.update(self.epa_features)
        features.update(self.pbp_features)
        return features

    @property
    def features(self) -> Dict[str, Any]:
        """Cached property for all features."""
        if self._features is None:
            self._features = self.load_features()
        return self._features

    @property
    def grouped_features(self) -> Dict[str, Any]:
        """Return features grouped by category."""
        return {
            'basic': self.basic_features,
            'dependent_stats': self.dependent_stats,
            'team_ranks': self.team_rank_features,
            'standings': self.team_standings_features,
            'position_ratings': self.position_rating_features,
            'team_trends': self.team_trend_features,
            'current_situation': self.current_situation_features,
            'officials': self.officials_features,
            'scoring_stats': self.scoring_stats_features,
            'weather': self.weather_features,
            'rest_days': self.rest_days_features,
            'betting': self.betting_features,
            'redzone': self.redzone_features,
            'epa': self.epa_features,
            'pbp': self.pbp_features,
        }

    @property
    def grouped_features_as_dfs(self) -> Dict[str, pd.DataFrame]:
        """Return features grouped by category as DataFrames."""
        return {key: pd.DataFrame([value]) for key, value in self.grouped_features.items()}
