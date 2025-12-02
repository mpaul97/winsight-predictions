from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.typing import DataFrameGroupBy

@dataclass
class FeatureEngine:
	# --- Core init arguments (former calculate_features parameters) ---
	prior_games: DataFrame
	target_name: str
	row: pd.Series
	position: str
	predicted_features: Optional[Dict[str, Any]] = None
	drop_cols: Optional[List[str]] = None
	game_src_df: Optional[DataFrame] = None

	# --- External data sources ---
	player_data: DataFrame = field(default_factory=pd.DataFrame)
	player_data_big_plays: DataFrame = field(default_factory=pd.DataFrame)
	standings: DataFrame = field(default_factory=pd.DataFrame)
	team_ranks: Dict[str, DataFrame] = field(default_factory=dict)
	player_group_ranks: Dict[str, DataFrame] = field(default_factory=dict)
	advanced_stat_cols: Dict[str, List[str]] = field(default_factory=dict)
	big_play_stat_columns: List[str] = field(default_factory=list)

	# --- Config / metadata ---
	feature_dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
		'completed_passes': ['attempted_passes'],
		'passing_yards': ['attempted_passes', 'completed_passes'],
		'passing_touchdowns': ['attempted_passes', 'completed_passes'],
		'interceptions_thrown': ['attempted_passes', 'completed_passes'],
		'rush_yards': ['rush_attempts'],
		'rush_touchdowns': ['rush_attempts'],
		'receptions': ['times_pass_target'],
		'receiving_yards': ['times_pass_target', 'receptions'],
		'receiving_touchdowns': ['times_pass_target', 'receptions'],
		'fantasy_points': [],
		'over_under_completed_passes_22+': ['attempted_passes'],
		'over_under_passing_yards_250+': ['attempted_passes', 'completed_passes'],
		'over_under_passing_touchdowns_2+': ['attempted_passes', 'completed_passes'],
		'over_under_interceptions_thrown_1+': ['attempted_passes', 'completed_passes'],
		'over_under_rush_yards_60+': ['rush_attempts'],
		'over_under_rush_touchdowns_1+': ['rush_attempts'],
		'over_under_receptions_5+': ['times_pass_target'],
		'over_under_receiving_yards_60+': ['times_pass_target', 'receptions'],
		'over_under_receiving_touchdowns_1+': ['times_pass_target', 'receptions'],
		'over_under_rush_yards_&_receiving_yards_100+': ['rush_attempts', 'times_pass_target', 'receptions'],
		'over_under_rush_touchdowns_&_receiving_touchdowns_1+': ['rush_attempts', 'times_pass_target', 'receptions'],
	})
	fantasy_dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
		'QB': ['attempted_passes', 'completed_passes', 'passing_yards', 'passing_touchdowns',
		       'interceptions_thrown', 'rush_attempts', 'rush_yards', 'rush_touchdowns'],
		'RB': ['rush_attempts', 'rush_yards', 'rush_touchdowns', 'times_pass_target',
		       'receptions', 'receiving_yards', 'receiving_touchdowns'],
		'WR': ['times_pass_target', 'receptions', 'receiving_yards', 'receiving_touchdowns',
		       'rush_attempts', 'rush_yards'],
		'TE': ['times_pass_target', 'receptions', 'receiving_yards', 'receiving_touchdowns']
	})
	advanced_stat_types: Dict[str, List[str]] = field(default_factory=lambda: {
		'QB': ['passing', 'rushing'],
		'RB': ['rushing', 'receiving'],
		'WR': ['rushing', 'receiving'],
		'TE': ['receiving']
	})

	# --- Internal caches for property loads ---
	_rolling_target_stats: Optional[Dict[str, Any]] = None
	_dependent_stats: Optional[Dict[str, Any]] = None
	_fantasy_dependent_stats: Optional[Dict[str, Any]] = None
	_last_game_numeric_stats: Optional[Dict[str, Any]] = None
	_similar_player_last_game: Optional[Dict[str, Any]] = None
	_epa_features: Optional[Dict[str, Any]] = None
	_home_away_splits: Optional[Dict[str, Any]] = None
	_team_ranks_features: Optional[Dict[str, Any]] = None
	_player_group_ranks_features: Optional[Dict[str, Any]] = None
	_advanced_stats_summaries: Optional[Dict[str, Any]] = None
	_weather_features: Optional[Dict[str, Any]] = None
	_days_rest_feature: Optional[Dict[str, Any]] = None
	_over_under_feature: Optional[Dict[str, Any]] = None
	_spread_and_favorite_feature: Optional[Dict[str, Any]] = None
	_starter_flag_feature: Optional[Dict[str, Any]] = None
	_game_targets_features: Optional[Dict[str, Any]] = None
	_standings_features: Optional[Dict[str, Any]] = None
	_big_plays_player_features: Optional[Dict[str, Any]] = None
	_big_plays_opponent_features: Optional[Dict[str, Any]] = None
	_features: Optional[Dict[str, Any]] = None

	# --- Helper caches ---
	team_ranks_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	player_group_ranks_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	standings_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	opp_big_plays_cache: Dict[str, DataFrame] = field(default_factory=dict)
	last_game_sim_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

	# ========== Low-level loaders ==========
	def _load_last_game_sim_player(self) -> Dict[str, Any]:
		prior_games = self.prior_games
		date: datetime = self.row['game_date']
		is_home = self.row['is_home']
		opp_abbr = self.row['away_abbr'] if is_home == 1 else self.row.get('home_abbr', self.row.get('opp_abbr'))
		position = self.position
		drop_cols = self.drop_cols or []
		key = f"{date}_{opp_abbr}_{position}"
		if key in self.last_game_sim_cache:
			return self.last_game_sim_cache[key]
		try:
			opp_last_data: DataFrameGroupBy = self.player_data[
				(self.player_data['game_date'] < date) &
				(self.player_data['abbr'] == opp_abbr) &
				(self.player_data['pos'] == position)
			].groupby('key')
			last_group_key = sorted(opp_last_data.groups.keys())[-1]
			last_group_df: DataFrame = opp_last_data.get_group(last_group_key)
			last_group_df = last_group_df.select_dtypes(include=float)
			median_off_pct = prior_games['off_pct'].median()
			last_group_df['diff'] = abs(last_group_df['off_pct'] - median_off_pct)
			last_group_df = last_group_df.loc[last_group_df['diff'].idxmin()] \
				.to_frame().transpose().reset_index(drop=True) \
				.drop(drop_cols, axis=1, errors='ignore')
			result = last_group_df.add_prefix("last_game_sim_").to_dict(orient='records')[0]
		except Exception:
			result = {}
		self.last_game_sim_cache[key] = result
		return result

	def _load_team_ranks_features(self) -> Dict[str, Any]:
		date: datetime = self.row['game_date']
		is_home = self.row['is_home']
		abbr = self.row['abbr']
		opp_abbr = self.row['away_abbr'] if is_home == 1 else self.row.get('home_abbr', self.row.get('opp_abbr'))
		features: Dict[str, Any] = {}
		for cfg_key, all_ranks in self.team_ranks.items():
			team_cache_key = f"{cfg_key}_{date.isoformat()}_{abbr}"
			if team_cache_key not in self.team_ranks_cache:
				df = all_ranks[(all_ranks['abbr'] == abbr) & (all_ranks['game_date'] <= date)].sort_values('game_date')
				self.team_ranks_cache[team_cache_key] = df.iloc[-1].to_dict() if not df.empty else {}
			for col, val in self.team_ranks_cache.get(team_cache_key, {}).items():
				features[f'team_rank_{cfg_key}_{col}'] = val
			opp_cache_key = f"{cfg_key}_{date.isoformat()}_{opp_abbr}"
			if opp_cache_key not in self.team_ranks_cache:
				df = all_ranks[(all_ranks['abbr'] == opp_abbr) & (all_ranks['game_date'] <= date)].sort_values('game_date')
				self.team_ranks_cache[opp_cache_key] = df.iloc[-1].to_dict() if not df.empty else {}
			for col, val in self.team_ranks_cache.get(opp_cache_key, {}).items():
				features[f'opp_rank_{cfg_key}_{col}'] = val
		return features

	def _load_player_group_ranks_features(self) -> Dict[str, Any]:
		date: datetime = self.row['game_date']
		is_home = self.row['is_home']
		abbr = self.row['abbr']
		opp_abbr = self.row['away_abbr'] if is_home == 1 else self.row.get('home_abbr', self.row.get('opp_abbr'))
		features: Dict[str, Any] = {}
		for cfg_key, all_ranks in self.player_group_ranks.items():
			if all_ranks.empty or 'game_date' not in all_ranks.columns:
				continue
			team_cache_key = f"{cfg_key}_{date.isoformat()}_{abbr}"
			if team_cache_key not in self.player_group_ranks_cache:
				df = all_ranks[(all_ranks['abbr'] == abbr) & (all_ranks['game_date'] <= date)].sort_values('game_date')
				self.player_group_ranks_cache[team_cache_key] = df.iloc[-1].to_dict() if not df.empty else {}
			for col, val in self.player_group_ranks_cache.get(team_cache_key, {}).items():
				features[f'group_rank_{cfg_key}_{col}'] = val
			opp_cache_key = f"{cfg_key}_{date.isoformat()}_{opp_abbr}"
			if opp_cache_key not in self.player_group_ranks_cache:
				df = all_ranks[(all_ranks['abbr'] == opp_abbr) & (all_ranks['game_date'] <= date)].sort_values('game_date')
				self.player_group_ranks_cache[opp_cache_key] = df.iloc[-1].to_dict() if not df.empty else {}
			for col, val in self.player_group_ranks_cache.get(opp_cache_key, {}).items():
				features[f'opp_group_rank_{cfg_key}_{col}'] = val
		return features

	def _load_team_standings_features(self) -> Dict[str, Any]:
		row = self.row
		abbr = row['abbr']
		is_home = row['is_home']
		opp_abbr = row['away_abbr'] if is_home == 1 else row.get('home_abbr', row.get('opp_abbr'))
		out: Dict[str, Any] = {}
		for tag in [abbr, opp_abbr]:
			current_week = row.get('week')
			current_year = row.get('year')
			target_week = row.get('last_week')
			if pd.isna(target_week) or (isinstance(target_week, (int, float)) and target_week < 1):
				target_week = (current_week or 1) - 1
				if isinstance(target_week, (int, float)) and target_week < 1:
					target_week = 1
			cache_key = f"{tag}_{target_week}_{current_year}"
			if cache_key in self.standings_cache:
				vals = self.standings_cache[cache_key]
			else:
				subset = self.standings[(self.standings['abbr'] == tag) &
										 (self.standings['week'] == target_week) &
										 (self.standings['year'] == current_year)]
				vals = subset.iloc[-1].to_dict() if not subset.empty else {}
				self.standings_cache[cache_key] = vals
			prefix = 'team_standing_' if tag == abbr else 'opp_standing_'
			out.update({f'{prefix}{k}': v for k, v in vals.items()})
		return out

	def _load_opp_big_plays_features(self) -> Dict[str, Any]:
		date: datetime = self.row['game_date']
		is_home = self.row['is_home']
		opp_abbr = self.row['away_abbr'] if is_home == 1 else self.row.get('home_abbr', self.row.get('opp_abbr'))
		pos = self.position
		cache_key = f"{opp_abbr}_{date.isoformat()}_{pos}"
		if cache_key not in self.opp_big_plays_cache:
			mask = (self.player_data_big_plays['game_date'] < date) & \
				   ((self.player_data_big_plays['home_abbr'] == opp_abbr) | (self.player_data_big_plays['away_abbr'] == opp_abbr)) & \
				   (self.player_data_big_plays['abbr'] != opp_abbr) & \
				   (self.player_data_big_plays['pos'] == pos)
			cols = ['key'] + list(self.big_play_stat_columns)
			df = self.player_data_big_plays.loc[mask, cols]
			self.opp_big_plays_cache[cache_key] = df
		else:
			df = self.opp_big_plays_cache[cache_key]
		out: Dict[str, Any] = {}
		if df is not None and not df.empty:
			last_game_vals = df[self.big_play_stat_columns].iloc[-1].to_dict()
			overall_vals = df[self.big_play_stat_columns].sum().to_dict()
			last3_vals = df[self.big_play_stat_columns].tail(3).sum().to_dict()
			last5_vals = df[self.big_play_stat_columns].tail(5).sum().to_dict()
			last10_vals = df[self.big_play_stat_columns].tail(10).sum().to_dict()
			std_vals = df[self.big_play_stat_columns].std().to_dict()
			for stat_col in self.big_play_stat_columns:
				out[f'opp_last_game_{stat_col}'] = last_game_vals.get(stat_col, 0)
				out[f'opp_overall_total_{stat_col}'] = overall_vals.get(stat_col, 0)
				out[f'opp_last3_total_{stat_col}'] = last3_vals.get(stat_col, 0)
				out[f'opp_last5_total_{stat_col}'] = last5_vals.get(stat_col, 0)
				out[f'opp_last10_total_{stat_col}'] = last10_vals.get(stat_col, 0)
				out[f'opp_std_{stat_col}'] = std_vals.get(stat_col, 0)
		else:
			for stat_col in self.big_play_stat_columns:
				out[f'opp_last_game_{stat_col}'] = 0
				out[f'opp_overall_total_{stat_col}'] = 0
				out[f'opp_last3_total_{stat_col}'] = 0
				out[f'opp_last5_total_{stat_col}'] = 0
				out[f'opp_last10_total_{stat_col}'] = 0
				out[f'opp_std_{stat_col}'] = 0
		return out

	def _load_game_targets_features(self) -> Dict[str, Any]:
		row = self.row
		is_home = row['is_home']
		abbr = row['abbr']
		src_df = self.game_src_df
		features: Dict[str, Any] = {}
		game_targets_regression = ['points', 'total_yards', 'pass_yards', 'rush_yards', 'pass_attempts', 'rush_attempts']
		game_targets_classification = ['win']
		if src_df is not None and not src_df.empty:
			preds_row = src_df.iloc[0]
			if is_home == 1:
				for t in game_targets_regression:
					features[f'team_game_{t}'] = preds_row.get(f'pred_home_{t}', np.nan)
					features[f'opp_game_{t}'] = preds_row.get(f'pred_away_{t}', np.nan)
				features['team_game_win'] = preds_row.get('pred_home_win', np.nan)
				features['opp_game_win'] = preds_row.get('pred_away_win', np.nan)
			else:
				for t in game_targets_regression:
					features[f'team_game_{t}'] = preds_row.get(f'pred_away_{t}', np.nan)
					features[f'opp_game_{t}'] = preds_row.get(f'pred_home_{t}', np.nan)
				features['team_game_win'] = preds_row.get('pred_away_win', np.nan)
				features['opp_game_win'] = preds_row.get('pred_home_win', np.nan)
		else:
			for t in game_targets_regression + game_targets_classification:
				features[f'team_game_{t}'] = row.get(f'team_game_{t}', np.nan)
				features[f'opp_game_{t}'] = row.get(f'opp_game_{t}', np.nan)
		return features

	# ========== Property wrappers ==========
	@property
	def rolling_target_stats(self) -> Dict[str, Any]:
		if self._rolling_target_stats is None:
			pg = self.prior_games; tn = self.target_name
			d = {
				f'overall_avg_{tn}': pg[tn].mean(),
				f'last5_avg_{tn}': pg[tn].tail(5).mean(),
				f'last10_avg_{tn}': pg[tn].tail(10).mean(),
				f'last_game_{tn}': pg[tn].iloc[-1]
			}
			rolling_means = pg[tn].rolling(window=10).mean().values
			for i in range(2, 7):
				d[f'last{i}_rolling_{tn}'] = rolling_means[-i] if len(rolling_means) > i else np.nan
			for n in range(1, min(10, len(pg)) + 1):
				d[f'last_nth_game_{n}_{tn}'] = pg[tn].iloc[-n]
			self._rolling_target_stats = d
		return self._rolling_target_stats

	@property
	def dependent_stats(self) -> Dict[str, Any]:
		if self._dependent_stats is None:
			pg = self.prior_games; tn = self.target_name; preds = self.predicted_features; row = self.row
			d: Dict[str, Any] = {}
			for dep in self.feature_dependencies.get(tn, []):
				if dep in pg.columns:
					if preds is not None and dep in preds:
						d[f'dep_{dep}_for_{tn}'] = preds[dep]  # Use predicted value for predictions
					elif dep in row.index:
						d[f'dep_{dep}_for_{tn}'] = row[dep]  # Use actual value for training
					else:
						d[f'dep_{dep}_for_{tn}'] = pg[dep].mean()  # Use historical average
			self._dependent_stats = d
		return self._dependent_stats

	@property
	def fantasy_dependent_stats(self) -> Dict[str, Any]:
		if self._fantasy_dependent_stats is None:
			if self.target_name != 'fantasy_points':
				self._fantasy_dependent_stats = {}
			else:
				pg = self.prior_games; pos = self.position; preds = self.predicted_features; row = self.row
				d: Dict[str, Any] = {}
				for dep in self.fantasy_dependencies.get(pos, []):
					if dep in pg.columns:
						if preds is not None and dep in preds:
							d[f'fp_dep_{dep}'] = preds[dep]  # Use predicted value for predictions
						elif dep in row.index:
							d[f'fp_dep_{dep}'] = row[dep]  # Use actual value for training
						else:
							d[f'fp_dep_{dep}'] = pg[dep].mean()  # Use historical average
				self._fantasy_dependent_stats = d
		return self._fantasy_dependent_stats

	@property
	def last_game_numeric_stats(self) -> Dict[str, Any]:
		if self._last_game_numeric_stats is None:
			pg = self.prior_games; tn = self.target_name
			d: Dict[str, Any] = {}
			for col in pg.select_dtypes(exclude=[object]).columns:
				if col == tn: continue
				try:
					d[f'last_game_{col}'] = pg[col].iloc[-1]
				except Exception:
					d[f'last_game_{col}'] = np.nan
			self._last_game_numeric_stats = d
		return self._last_game_numeric_stats

	@property
	def similar_player_last_game(self) -> Dict[str, Any]:
		if self._similar_player_last_game is None:
			self._similar_player_last_game = self._load_last_game_sim_player()
		return self._similar_player_last_game

	@property
	def epa_features(self) -> Dict[str, Any]:
		if self._epa_features is None:
			pg = self.prior_games
			d: Dict[str, Any] = {}
			for c in ['epa', 'epa_added']:
				if c in pg.columns:
					d[f'overall_avg_{c}'] = pg[c].mean()
					d[f'last5_avg_{c}'] = pg[c].tail(5).mean()
					d[f'last10_avg_{c}'] = pg[c].tail(10).mean()
			self._epa_features = d
		return self._epa_features

	@property
	def home_away_splits(self) -> Dict[str, Any]:
		if self._home_away_splits is None:
			pg = self.prior_games; tn = self.target_name; is_home = self.row['is_home']
			d: Dict[str, Any] = {'is_home': is_home}
			home_games = pg[pg['is_home'] == 1]
			away_games = pg[pg['is_home'] == 0]
			def _apply(prefix: str, df: DataFrame):
				if not df.empty:
					d[f'{prefix}_avg_{tn}'] = df[tn].mean()
					d[f'{prefix}_last5_avg_{tn}'] = df[tn].tail(5).mean()
					d[f'{prefix}_last_{tn}'] = df[tn].iloc[-1]
				else:
					d[f'{prefix}_avg_{tn}'] = np.nan
					d[f'{prefix}_last5_avg_{tn}'] = np.nan
					d[f'{prefix}_last_{tn}'] = np.nan
			_apply('home', home_games)
			_apply('away', away_games)
			self._home_away_splits = d
		return self._home_away_splits

	@property
	def team_ranks_features(self) -> Dict[str, Any]:
		if self._team_ranks_features is None:
			self._team_ranks_features = self._load_team_ranks_features()
		return self._team_ranks_features

	@property
	def player_group_ranks_features(self) -> Dict[str, Any]:
		if self._player_group_ranks_features is None:
			self._player_group_ranks_features = self._load_player_group_ranks_features()
		return self._player_group_ranks_features

	@property
	def advanced_stats_summaries(self) -> Dict[str, Any]:
		if self._advanced_stats_summaries is None:
			pg = self.prior_games; pos = self.position
			d: Dict[str, Any] = {}
			for stat_type in self.advanced_stat_types.get(pos, []):
				cols = self.advanced_stat_cols.get(stat_type, [])
				for col in cols:
					col = f"adv_{stat_type}_{col}"
					d[f'{col}_last3_mean'] = pg[col].tail(3).mean() if col in pg.columns else np.nan
					d[f'{col}_last5_mean'] = pg[col].tail(5).mean() if col in pg.columns else np.nan
					d[f'{col}_last10_mean'] = pg[col].tail(10).mean() if col in pg.columns else np.nan
					d[f'{col}_overall_mean'] = pg[col].mean() if col in pg.columns else np.nan
			self._advanced_stats_summaries = d
		return self._advanced_stats_summaries

	@property
	def weather_features(self) -> Dict[str, Any]:
		if self._weather_features is None:
			weather = self.row.get('weather')
			if isinstance(weather, str) and '|' in weather:
				parts = weather.split('|')
				d = {
					'temperature': float(parts[0]) if parts[0] else 72,
					'humidity': float(parts[1]) if parts[1] else 50,
					'wind_speed': float(parts[2]) if len(parts) > 2 and parts[2] else 0,
				}
			else:
				d = {'temperature': 72, 'humidity': 50, 'wind_speed': 0}
			self._weather_features = d
		return self._weather_features

	@property
	def days_rest_feature(self) -> Dict[str, Any]:
		if self._days_rest_feature is None:
			pg = self.prior_games; date = self.row['game_date']
			if len(pg) > 0:
				last_game_date = pg['game_date'].max()
				self._days_rest_feature = {'days_rest': (date - last_game_date).days}
			else:
				self._days_rest_feature = {'days_rest': np.nan}
		return self._days_rest_feature

	@property
	def over_under_feature(self) -> Dict[str, Any]:
		if self._over_under_feature is None:
			row = self.row
			if 'over_under' in row.index:
				try:
					self._over_under_feature = {'over_under': float(str(row['over_under']).split(' ')[0])}
				except Exception:
					self._over_under_feature = {'over_under': np.nan}
			elif 'ou' in row.index:
				self._over_under_feature = {'over_under': float(row['ou']) if isinstance(row['ou'], (float, int)) else np.nan}
			else:
				self._over_under_feature = {'over_under': np.nan}
		return self._over_under_feature

	@property
	def spread_and_favorite_feature(self) -> Dict[str, Any]:
		if self._spread_and_favorite_feature is None:
			row = self.row
			if row.get('home_is_favorite') is not None and row.get('is_home') is not None:
				is_favorite = 1 if (row['home_is_favorite'] == 1 and row['is_home'] == 1) or (row['home_is_favorite'] == 0 and row['is_home'] == 0) else 0
				self._spread_and_favorite_feature = {'spread': row.get('spread', np.nan), 'is_favorite': is_favorite}
			else:
				self._spread_and_favorite_feature = {'spread': row.get('spread', np.nan), 'is_favorite': row.get('is_favorite', np.nan)}
		return self._spread_and_favorite_feature

	@property
	def starter_flag_feature(self) -> Dict[str, Any]:
		if self._starter_flag_feature is None:
			self._starter_flag_feature = {'starter': self.row.get('starter', 0)}
		return self._starter_flag_feature

	@property
	def game_targets_features(self) -> Dict[str, Any]:
		if self._game_targets_features is None:
			self._game_targets_features = self._load_game_targets_features()
		return self._game_targets_features

	@property
	def standings_features(self) -> Dict[str, Any]:
		if self._standings_features is None:
			self._standings_features = self._load_team_standings_features()
		return self._standings_features

	@property
	def big_plays_player_features(self) -> Dict[str, Any]:
		if self._big_plays_player_features is None:
			pg = self.prior_games; row = self.row
			d: Dict[str, Any] = {}
			for col in ['big_play_count_10', 'big_play_count_20', 'big_play_count_30', 'big_play_count_40', 'big_play_count_50']:
				if col in pg.columns:
					d[f'overall_total_{col}'] = pg[col].sum()
					d[f'last5_total_{col}'] = pg[col].tail(5).sum()
					d[f'last10_total_{col}'] = pg[col].tail(10).sum()
					if 'year' in pg.columns:
						d[f'season_total_{col}'] = pg[pg['year'] == row.get('year')][col].sum()
			self._big_plays_player_features = d
		return self._big_plays_player_features

	@property
	def big_plays_opponent_features(self) -> Dict[str, Any]:
		if self._big_plays_opponent_features is None:
			self._big_plays_opponent_features = self._load_opp_big_plays_features()
		return self._big_plays_opponent_features

	# ========== Aggregated features ==========
	def load_features(self) -> Dict[str, Any]:
		if self._features is not None:
			return self._features
		parts = [
			self.rolling_target_stats,
			self.dependent_stats,
			self.fantasy_dependent_stats,
			self.last_game_numeric_stats,
			self.similar_player_last_game,
			self.epa_features,
			self.home_away_splits,
			self.team_ranks_features,
			self.player_group_ranks_features,
			self.advanced_stats_summaries,
			self.weather_features,
			self.days_rest_feature,
			self.over_under_feature,
			self.spread_and_favorite_feature,
			self.starter_flag_feature,
			self.game_targets_features,
			self.standings_features,
			self.big_plays_player_features,
			self.big_plays_opponent_features,
		]
		agg: Dict[str, Any] = {}
		for p in parts:
			agg.update(p)
		self._features = agg
		return agg

	@property
	def features(self) -> Dict[str, Any]:
		return self.load_features()

	@property
	def grouped_features(self) -> Dict[str, Dict[str, Any]]:
		"""Group features dynamically for storage or processing."""
		return {
			"rolling_stats": self.rolling_target_stats,
			"dependent_stats": self.dependent_stats,
			"fantasy_stats": self.fantasy_dependent_stats,
			"last_game_stats": self.last_game_numeric_stats,
			"similar_player_stats": self.similar_player_last_game,
			"epa_features": self.epa_features,
			"home_away_splits": self.home_away_splits,
			"team_ranks": self.team_ranks_features,
			"player_group_ranks": self.player_group_ranks_features,
			"advanced_stats": self.advanced_stats_summaries,
			"weather": self.weather_features,
			"days_rest": self.days_rest_feature,
			"over_under": self.over_under_feature,
			"spread_and_favorite": self.spread_and_favorite_feature,
			"starter_flag": self.starter_flag_feature,
			"game_targets": self.game_targets_features,
			"standings": self.standings_features,
			"big_plays_player": self.big_plays_player_features,
			"big_plays_opponent": self.big_plays_opponent_features,
		}

	@property
	def grouped_features_as_dfs(self) -> Dict[str, pd.DataFrame]:
		"""Map all grouped features to Dict[str, pd.DataFrame]."""
		grouped = self.grouped_features
		result = {}

		for key, value in grouped.items():
			if isinstance(value, pd.DataFrame):
				result[key] = value
			elif isinstance(value, dict):
				# Convert dictionary to DataFrame
				result[key] = pd.DataFrame([value])
			else:
				# Wrap scalar values or unsupported types in a DataFrame
				result[key] = pd.DataFrame([{key: value}])

		return result
