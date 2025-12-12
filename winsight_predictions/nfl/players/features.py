from __future__ import annotations

from dataclasses import dataclass, field
import logging
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
	game_predictions: Optional[DataFrame] = None

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
	_snap_counts_features: Optional[Dict[str, Any]] = None
	_features: Optional[Dict[str, Any]] = None

	# --- Helper caches ---
	team_ranks_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	player_group_ranks_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	standings_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	opp_big_plays_cache: Dict[str, DataFrame] = field(default_factory=dict)
	last_game_sim_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	
	# --- Feature computation tracking (for debugging) ---
	_cached_features_used: int = 0
	_computed_features: int = 0

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
			last_group_df = last_group_df.fillna(0.0)
			result = last_group_df.add_prefix("last_game_sim_").to_dict(orient='records')[0]
		except Exception:
			result = {}
		self.last_game_sim_cache[key] = result
		return result

	def _load_team_ranks_features(self) -> Dict[str, Any]:
		# OPTIMIZATION: Efficient caching with reduced dictionary operations
		# BEFORE: O(n*k) - filtering DataFrames multiple times per config
		# AFTER: O(k) - optimized boolean indexing and batch dictionary updates
		date: datetime = self.row['game_date']
		is_home = self.row['is_home']
		abbr = self.row['abbr']
		opp_abbr = self.row['away_abbr'] if is_home == 1 else self.row.get('home_abbr', self.row.get('opp_abbr'))
		features: Dict[str, Any] = {}
		
		for cfg_key, all_ranks in self.team_ranks.items():
			team_cache_key = f"{cfg_key}_{date.isoformat()}_{abbr}"
			if team_cache_key not in self.team_ranks_cache:
				# Vectorized boolean indexing - single pass
				mask = (all_ranks['abbr'] == abbr) & (all_ranks['game_date'] <= date)
				df = all_ranks[mask]
				if not df.empty:
					# Use idxmax to avoid sort - O(n) instead of O(n log n)
					last_idx = df['game_date'].idxmax()
					self.team_ranks_cache[team_cache_key] = all_ranks.loc[last_idx].to_dict()
				else:
					self.team_ranks_cache[team_cache_key] = {}
			
			# Batch update features dictionary
			team_data = self.team_ranks_cache.get(team_cache_key, {})
			features.update({f'team_rank_{cfg_key}_{col}': val for col, val in team_data.items()})
			
			opp_cache_key = f"{cfg_key}_{date.isoformat()}_{opp_abbr}"
			if opp_cache_key not in self.team_ranks_cache:
				mask = (all_ranks['abbr'] == opp_abbr) & (all_ranks['game_date'] <= date)
				df = all_ranks[mask]
				if not df.empty:
					last_idx = df['game_date'].idxmax()
					self.team_ranks_cache[opp_cache_key] = all_ranks.loc[last_idx].to_dict()
				else:
					self.team_ranks_cache[opp_cache_key] = {}
			
			opp_data = self.team_ranks_cache.get(opp_cache_key, {})
			features.update({f'opp_rank_{cfg_key}_{col}': val for col, val in opp_data.items()})
			
		return features

	def _load_player_group_ranks_features(self) -> Dict[str, Any]:
		# OPTIMIZATION: Use idxmax instead of sort, batch dictionary updates
		# BEFORE: O(n*k) - sort operations for each config
		# AFTER: O(k) - idxmax O(n) instead of sort O(n log n)
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
				mask = (all_ranks['abbr'] == abbr) & (all_ranks['game_date'] <= date)
				df = all_ranks[mask]
				if not df.empty:
					last_idx = df['game_date'].idxmax()
					self.player_group_ranks_cache[team_cache_key] = all_ranks.loc[last_idx].to_dict()
				else:
					self.player_group_ranks_cache[team_cache_key] = {}
			
			team_data = self.player_group_ranks_cache.get(team_cache_key, {})
			features.update({f'group_rank_{cfg_key}_{col}': val for col, val in team_data.items()})
			
			opp_cache_key = f"{cfg_key}_{date.isoformat()}_{opp_abbr}"
			if opp_cache_key not in self.player_group_ranks_cache:
				mask = (all_ranks['abbr'] == opp_abbr) & (all_ranks['game_date'] <= date)
				df = all_ranks[mask]
				if not df.empty:
					last_idx = df['game_date'].idxmax()
					self.player_group_ranks_cache[opp_cache_key] = all_ranks.loc[last_idx].to_dict()
				else:
					self.player_group_ranks_cache[opp_cache_key] = {}
			
			opp_data = self.player_group_ranks_cache.get(opp_cache_key, {})
			features.update({f'opp_group_rank_{cfg_key}_{col}': val for col, val in opp_data.items()})
			
		return features

	def _load_team_standings_features(self) -> Dict[str, Any]:
		# OPTIMIZATION: Efficient caching and batch updates
		# BEFORE: O(n) - filtering standings dataframe twice
		# AFTER: O(1) - cached results with optimized boolean indexing
		row = self.row
		abbr = row['abbr']
		is_home = row['is_home']
		opp_abbr = row['away_abbr'] if is_home == 1 else row.get('home_abbr', row.get('opp_abbr'))
		out: Dict[str, Any] = {}
		
		# Simple encodings for non-numeric columns
		conference_encoding = {'AFC': 0, 'NFC': 1}
		division_encoding = {
			'AFC East': 0, 'AFC North': 1, 'AFC South': 2, 'AFC West': 3,
			'NFC East': 4, 'NFC North': 5, 'NFC South': 6, 'NFC West': 7
		}
		
		for tag, prefix in [(abbr, 'team_standing_'), (opp_abbr, 'opp_standing_')]:
			current_week = row.get('week')
			current_year = row.get('year')
			target_week = row.get('last_week')
			
			if pd.isna(target_week) or (isinstance(target_week, (int, float)) and target_week < 1):
				target_week = (current_week or 1) - 1
				if isinstance(target_week, (int, float)) and target_week < 1:
					target_week = 1
			
			cache_key = f"{tag}_{target_week}_{current_year}"
			if cache_key not in self.standings_cache:
				# Vectorized boolean indexing
				mask = (self.standings['abbr'] == tag) & \
					   (self.standings['week'] == target_week) & \
					   (self.standings['year'] == current_year)
				subset = self.standings[mask]
				vals = subset.iloc[-1].to_dict() if not subset.empty else {}
				self.standings_cache[cache_key] = vals
			else:
				vals = self.standings_cache[cache_key]
			
			# Batch dictionary update with encodings
			for k, v in vals.items():
				feature_key = f'{prefix}{k}'
				
				# Apply encodings for non-numeric columns
				if k == 'conference':
					out[feature_key] = conference_encoding.get(v, -1) if isinstance(v, str) else v
				elif k == 'division':
					out[feature_key] = division_encoding.get(v, -1) if isinstance(v, str) else v
				elif k in ['is_division_winner', 'is_wild_card']:
					# Convert boolean strings to integers
					if isinstance(v, str):
						out[feature_key] = 1 if v.lower() == 'true' else 0
					elif isinstance(v, bool):
						out[feature_key] = int(v)
					else:
						out[feature_key] = v
				elif k in ['name', 'abbr']:
					# Skip name and abbr columns as they are identifiers, not features
					continue
				else:
					out[feature_key] = v
		
		return out

	def _load_opp_big_plays_features(self) -> Dict[str, Any]:
		# OPTIMIZATION: Vectorized aggregation instead of multiple operations
		# BEFORE: O(n*s) - multiple tail() and aggregation calls per stat column
		# AFTER: O(s) - single aggregation operation with numpy slicing
		date: datetime = self.row['game_date']
		is_home = self.row['is_home']
		opp_abbr = self.row['away_abbr'] if is_home == 1 else self.row.get('home_abbr', self.row.get('opp_abbr'))
		pos = self.position
		logging.debug(f"[Opp Big Plays Feature]: {date}, {is_home}, {opp_abbr}, {pos}")
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
		if df is not None and not df.empty and self.big_play_stat_columns:
			n_rows = len(df)
			# Vectorized: get all stats at once using numpy array operations
			stats_array = df[self.big_play_stat_columns].fillna(0.0).values
			
			# Single aggregation operations
			last_game_vals = stats_array[-1] if n_rows > 0 else np.zeros(len(self.big_play_stat_columns))
			overall_vals = stats_array.sum(axis=0)
			last3_vals = stats_array[-3:].sum(axis=0) if n_rows >= 3 else stats_array.sum(axis=0)
			last5_vals = stats_array[-5:].sum(axis=0) if n_rows >= 5 else stats_array.sum(axis=0)
			last10_vals = stats_array[-10:].sum(axis=0) if n_rows >= 10 else stats_array.sum(axis=0)
			std_vals = stats_array.std(axis=0) if n_rows > 1 else np.zeros(len(self.big_play_stat_columns))
		
			# Batch dictionary creation
			for i, stat_col in enumerate(self.big_play_stat_columns):
				out[f'opp_last_game_{stat_col}'] = last_game_vals[i]
				out[f'opp_overall_total_{stat_col}'] = overall_vals[i]
				out[f'opp_last3_total_{stat_col}'] = last3_vals[i]
				out[f'opp_last5_total_{stat_col}'] = last5_vals[i]
				out[f'opp_last10_total_{stat_col}'] = last10_vals[i]
				out[f'opp_std_{stat_col}'] = std_vals[i]
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
		src_df = self.game_predictions.copy().rename(columns={ 'game_id':'key' })
		features: Dict[str, Any] = {}
		game_targets_regression = ['points', 'total_yards', 'pass_yards', 'rush_yards', 'pass_attempts', 'rush_attempts']
		game_targets_classification = ['win']
		try:
			if src_df is not None and not src_df.empty:
				preds_row = src_df[src_df['key'] == row['key']].iloc[0].to_dict()
				if is_home == 1:
					for t in game_targets_regression:
						features[f'team_game_{t}'] = preds_row.get(f'predicted_home_{t}', preds_row.get(f'home_{t}', np.nan))
						features[f'opp_game_{t}'] = preds_row.get(f'predicted_away_{t}', preds_row.get(f'away_{t}', np.nan))
					features['team_game_win'] = preds_row.get('predicted_home_win', preds_row.get(f'home_win', np.nan))
					features['opp_game_win'] = preds_row.get('predicted_away_win', 1 - preds_row.get(f'home_win', np.nan))
				else:
					for t in game_targets_regression:
						features[f'team_game_{t}'] = preds_row.get(f'predicted_away_{t}', preds_row.get(f'away_{t}', np.nan))
						features[f'opp_game_{t}'] = preds_row.get(f'predicted_home_{t}', preds_row.get(f'home_{t}', np.nan))
					features['team_game_win'] = preds_row.get('predicted_away_win', 1 - preds_row.get(f'home_win', np.nan))
					features['opp_game_win'] = preds_row.get('predicted_home_win', preds_row.get(f'home_win', np.nan))
			else:
				for t in game_targets_regression + game_targets_classification:
					features[f'team_game_{t}'] = row.get(f'team_game_{t}', np.nan)
					features[f'opp_game_{t}'] = row.get(f'opp_game_{t}', np.nan)
		except Exception as e:
			logging.warning(f"Error loading game target features for row {row.get('key', 'unknown')}: {e}")
			for t in game_targets_regression + game_targets_classification:
				features[f'team_game_{t}'] = np.nan
				features[f'opp_game_{t}'] = np.nan
		return features

	# ========== Property wrappers ==========
	@property
	def rolling_target_stats(self) -> Dict[str, Any]:
		# OPTIMIZATION: Vectorized rolling operations
		# BEFORE: O(n*m) - multiple iterations over games with individual lookups
		# AFTER: O(n) - single pass with vectorized pandas operations
		if self._rolling_target_stats is None:
			pg = self.prior_games; tn = self.target_name
			target_values = pg[tn].fillna(0.0).values  # Single array access
			n_games = len(target_values)
			
			# Pre-compute all aggregates in one pass
			d = {
				f'overall_avg_{tn}': target_values.mean(),
				f'last5_avg_{tn}': target_values[-5:].mean() if n_games >= 5 else target_values.mean(),
				f'last10_avg_{tn}': target_values[-10:].mean() if n_games >= 10 else target_values.mean(),
				f'last_game_{tn}': target_values[-1] if n_games > 0 else np.nan
			}
			
			# Vectorized rolling computation - single operation
			rolling_means = pg[tn].fillna(0.0).rolling(window=10, min_periods=1).mean().values
			for i in range(2, 7):
				d[f'last{i}_rolling_{tn}'] = rolling_means[-i] if len(rolling_means) >= i else np.nan
			
			# Vectorized nth game lookups using negative indexing
			max_n = min(10, n_games)
			for n in range(1, max_n + 1):
				d[f'last_nth_game_{n}_{tn}'] = target_values[-n]
			
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
		# OPTIMIZATION: Vectorized column selection and single row access
		# BEFORE: O(n*c) - iterating over columns with try/except and individual iloc calls
		# AFTER: O(c) - single iloc operation on filtered dataframe
		if self._last_game_numeric_stats is None:
			pg = self.prior_games; tn = self.target_name
			if len(pg) == 0:
				self._last_game_numeric_stats = {}
				return self._last_game_numeric_stats
			
			# Vectorized: select numeric columns and exclude target in one operation
			numeric_cols = [col for col in pg.select_dtypes(exclude=[object]).columns if 'game_date' not in col]
			
			if numeric_cols:
				# Single iloc operation to get last row
				last_row = pg[numeric_cols].iloc[-1]
				# Direct dictionary creation with prefix
				d = {f'last_game_{col}': val for col, val in last_row.items()}
			else:
				d = {}

			self._last_game_numeric_stats = d
		return self._last_game_numeric_stats

	@property
	def similar_player_last_game(self) -> Dict[str, Any]:
		if self._similar_player_last_game is None:
			self._similar_player_last_game = self._load_last_game_sim_player()
		return self._similar_player_last_game

	@property
	def snap_counts_features(self) -> Dict[str, Any]:
		# OPTIMIZATION: Pre-compute values and use vectorized operations
		# BEFORE: O(n*w) - multiple tail() operations for different windows
		# AFTER: O(w) - single array access with efficient slicing
		if self._snap_counts_features is None:
			pg = self.prior_games
			row = self.row
			d: Dict[str, Any] = {}
			
			# For training: use actual snap counts
			if 'off_pct' in pg.columns:
				off_pct_values = pg['off_pct'].values  # Single array access
				n_games = len(off_pct_values)
				
				if n_games == 0:
					self._snap_counts_features = {}
					return self._snap_counts_features
				
				# Pre-compute common values using array operations
				overall_avg = off_pct_values.mean()
				last_game_val = off_pct_values[-1]
				last5_avg = off_pct_values[-5:].mean() if n_games >= 5 else overall_avg
				last3_avg = off_pct_values[-3:].mean() if n_games >= 3 else overall_avg
				
				# Historical snap count averages
				d['snap_overall_avg'] = overall_avg
				d['snap_last5_avg'] = last5_avg
				d['snap_last3_avg'] = last3_avg
				d['snap_last_game'] = last_game_val
				
				# Vectorized rolling averages - compute all at once
				for window in [3, 5, 7, 10]:
					d[f'snap_rolling_{window}'] = off_pct_values[-window:].mean() if n_games >= window else overall_avg
				
				# Weighted recent average (last game * 1.2 + last 5 avg) / 2
				d['snap_weighted_recent'] = (last_game_val * 1.2 + last5_avg) / 2.0
				
				# For prediction: calculate next game snap prediction
				if self.predicted_features is not None:  # We're predicting
					logging.debug(f"Calculating snap count prediction for player {row.get('pid', 'unknown')}")
					# Fill missing off_pct with 0 for healthy players who didn't play
					off_pct_series = pg['off_pct'].fillna(0)
					
					# Calculate prediction: average of (last 5 avg, last game * 1.2)
					last5_mean = off_pct_series.tail(5).mean()
					last_game = off_pct_series.iloc[-1] if len(off_pct_series) > 0 else 0.0
					final_pred = np.mean([last5_mean, last_game * 1.2])
					
					# Clip to valid range [0, 1]
					final_pred = np.clip(final_pred, 0.0, 1.0)
					
					# Boost for starters (if starter flag is 1 in row)
					if 'starter' in row.index and row.get('starter', 0) == 1:
						final_pred = np.clip(final_pred + 0.15, 0.0, 1.0)
					
					d['snap_current_game'] = round(final_pred, 4)
				else:
					logging.debug(f"Using actual snap count for training for player {row.get('player_name', 'unknown')}")
					# Training mode: use actual current game snap count if available
					if 'off_pct' in row.index:
						d['snap_current_game'] = row['off_pct']
			else:
				# No snap data available
				d['snap_overall_avg'] = 0.0
				d['snap_last5_avg'] = 0.0
				d['snap_last3_avg'] = 0.0
				d['snap_last_game'] = 0.0
				d['snap_weighted_recent'] = 0.0
				d['snap_current_game'] = 0.0
			
			self._snap_counts_features = d
		return self._snap_counts_features

	@property
	def epa_features(self) -> Dict[str, Any]:
		# OPTIMIZATION: Vectorized array slicing for EPA calculations
		# BEFORE: O(n*c) - multiple tail() operations per column
		# AFTER: O(c) - single array access with efficient slicing
		if self._epa_features is None:
			pg = self.prior_games
			d: Dict[str, Any] = {}
			n_games = len(pg)
			
			for c in ['epa', 'epa_added']:
				if c in pg.columns:
					values = pg[c].fillna(0.0).values
					d[f'overall_avg_{c}'] = values.mean()
					d[f'last5_avg_{c}'] = values[-5:].mean() if n_games >= 5 else values.mean()
					d[f'last10_avg_{c}'] = values[-10:].mean() if n_games >= 10 else values.mean()
			
			self._epa_features = d
		return self._epa_features

	@property
	def home_away_splits(self) -> Dict[str, Any]:
		# OPTIMIZATION: Vectorized boolean masking for home/away splits
		# BEFORE: O(n) - filtering dataframe twice with separate tail() calls
		# AFTER: O(1) - single boolean mask with array slicing
		if self._home_away_splits is None:
			pg = self.prior_games; tn = self.target_name; is_home = self.row['is_home']
			d: Dict[str, Any] = {'is_home': is_home}
			
			if tn in pg.columns and 'is_home' in pg.columns:
				# Create masks once
				home_mask = pg['is_home'] == 1
				away_mask = pg['is_home'] == 0
				
				# Get values using masks - more efficient than filtering twice
				home_values = pg.loc[home_mask, tn].values
				away_values = pg.loc[away_mask, tn].values
				
				# Home stats
				if len(home_values) > 0:
					d['home_avg_' + tn] = home_values.mean()
					d['home_last5_avg_' + tn] = home_values[-5:].mean() if len(home_values) >= 5 else home_values.mean()
					d['home_last_' + tn] = home_values[-1]
				else:
					d['home_avg_' + tn] = np.nan
					d['home_last5_avg_' + tn] = np.nan
					d['home_last_' + tn] = np.nan
				
				# Away stats
				if len(away_values) > 0:
					d['away_avg_' + tn] = away_values.mean()
					d['away_last5_avg_' + tn] = away_values[-5:].mean() if len(away_values) >= 5 else away_values.mean()
					d['away_last_' + tn] = away_values[-1]
				else:
					d['away_avg_' + tn] = np.nan
					d['away_last5_avg_' + tn] = np.nan
					d['away_last_' + tn] = np.nan
			
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
		# OPTIMIZATION: Batch column processing with vectorized operations
		# BEFORE: O(n*s*c) - multiple tail() operations per stat type per column
		# AFTER: O(s*c) - single array access with efficient slicing per column
		if self._advanced_stats_summaries is None:
			pg = self.prior_games; pos = self.position
			d: Dict[str, Any] = {}
			n_games = len(pg)
			
			for stat_type in self.advanced_stat_types.get(pos, []):
				cols = self.advanced_stat_cols.get(stat_type, [])
				for col in cols:
					col_name = f"adv_{stat_type}_{col}"
					if col_name in pg.columns:
						values = pg[col_name].values
						d[f'{col_name}_overall_mean'] = values.mean()
						d[f'{col_name}_last3_mean'] = values[-3:].mean() if n_games >= 3 else values.mean()
						d[f'{col_name}_last5_mean'] = values[-5:].mean() if n_games >= 5 else values.mean()
						d[f'{col_name}_last10_mean'] = values[-10:].mean() if n_games >= 10 else values.mean()
					else:
						d[f'{col_name}_overall_mean'] = np.nan
						d[f'{col_name}_last3_mean'] = np.nan
						d[f'{col_name}_last5_mean'] = np.nan
						d[f'{col_name}_last10_mean'] = np.nan
			
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
			spread = row.get('spread', np.nan)
			home_is_favorite = row.get('home_is_favorite', None)

			# Handle 'Pick' case
			if isinstance(spread, str) and spread.lower() == 'pick':
				spread = 0.0
				home_is_favorite = True

			if home_is_favorite is not None and row.get('is_home') is not None:
				is_favorite = 1 if (home_is_favorite and row['is_home'] == 1) or (not home_is_favorite and row['is_home'] == 0) else 0
				self._spread_and_favorite_feature = {'spread': spread, 'is_favorite': is_favorite}
			else:
				self._spread_and_favorite_feature = {'spread': spread, 'is_favorite': row.get('is_favorite', np.nan)}
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
		# OPTIMIZATION: Batch column processing with vectorized operations
		# BEFORE: O(n*c) - iterating columns with multiple tail() operations each
		# AFTER: O(c) - single pass with array slicing
		if self._big_plays_player_features is None:
			pg = self.prior_games; row = self.row
			d: Dict[str, Any] = {}
			
			big_play_cols = ['big_play_count_10', 'big_play_count_20', 'big_play_count_30', 'big_play_count_40', 'big_play_count_50']
			available_cols = [col for col in big_play_cols if col in pg.columns]
			if available_cols:
				n_games = len(pg)
				# Pre-filter for season if year column exists
				season_mask = pg['year'] == row.get('year') if 'year' in pg.columns else None
				
				# Vectorized: process all columns at once
				for col in available_cols:
					col_values = pg[col].fillna(0.0).values
					d[f'overall_total_{col}'] = col_values.sum()
					d[f'last5_total_{col}'] = col_values[-5:].sum() if n_games >= 5 else col_values.sum()
					d[f'last10_total_{col}'] = col_values[-10:].sum() if n_games >= 10 else col_values.sum()
					if season_mask is not None:
						d[f'season_total_{col}'] = pg.loc[season_mask, col].sum()
						
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
			self.snap_counts_features,
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

	def load_features_lightweight(self) -> Dict[str, Any]:
		"""Lightweight feature loading that only computes target-specific features.
		
		This method assumes that shared features have already been cached
		in the private attributes (e.g., _team_ranks_features, _weather_features).
		It only computes the target-dependent features (rolling_target_stats,
		dependent_stats, fantasy_dependent_stats, home_away_splits).
		
		Returns:
			Dictionary of all features with cached shared features and computed target-specific features
		"""
		# Target-specific features that must be computed
		target_specific = [
			self.rolling_target_stats,
			self.dependent_stats,
			self.fantasy_dependent_stats,
			self.home_away_splits,
		]
		
		# Track computed features
		self._computed_features += len(target_specific)
		
		# Shared features that should already be cached (will return from cache if set)
		shared_feature_properties = [
			('last_game_numeric_stats', self._last_game_numeric_stats),
			('similar_player_last_game', self._similar_player_last_game),
			('snap_counts_features', self._snap_counts_features),
			('epa_features', self._epa_features),
			('team_ranks_features', self._team_ranks_features),
			('player_group_ranks_features', self._player_group_ranks_features),
			('advanced_stats_summaries', self._advanced_stats_summaries),
			('weather_features', self._weather_features),
			('days_rest_feature', self._days_rest_feature),
			('over_under_feature', self._over_under_feature),
			('spread_and_favorite_feature', self._spread_and_favorite_feature),
			('starter_flag_feature', self._starter_flag_feature),
			('game_targets_features', self._game_targets_features),
			('standings_features', self._standings_features),
			('big_plays_player_features', self._big_plays_player_features),
			('big_plays_opponent_features', self._big_plays_opponent_features),
		]
		
		# Count cached vs computed shared features
		for name, cached_value in shared_feature_properties:
			if cached_value is not None:
				self._cached_features_used += 1
			else:
				self._computed_features += 1
		
		# Access properties to get values (will use cache or compute)
		shared = [
			self.last_game_numeric_stats,
			self.similar_player_last_game,
			self.snap_counts_features,
			self.epa_features,
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
		
		# Compute target-specific features
		for p in target_specific:
			agg.update(p)
		
		# Get shared features from cache (or compute if not cached)
		for p in shared:
			agg.update(p)
		
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
			"snap_counts": self.snap_counts_features,
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
