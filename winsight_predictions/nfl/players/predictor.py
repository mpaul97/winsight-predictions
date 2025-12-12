"""Lean player prediction and training interfaces.

This module intentionally removes the legacy monolithic `Predictor` class
and all duplicated feature engineering logic. Feature construction is now
exclusively delegated to `FeatureEngine` (see `features.py`). Data loading,
storage mode abstraction (local vs S3), and ancillary metadata are handled
by `DataObject` (see `data_object.py`).

Provided classes:
-----------------
1. PlayerPredictor (inference only)
   - Consumes pre-trained per-target models you supply.
   - Iteratively predicts targets in dependency order so later targets can
     incorporate earlier predictions (e.g. yards depend on attempts &
     completions; fantasy points depend on all positional stat outputs).
   - Uses a private helper to instantiate FeatureEngine consistently.

2. PlayerModelTrainer (simple training helper)
   - Builds per-target training rows from historical player games.
   - Trains a baseline model (RandomForestRegressor if available; else a
     tiny MeanRegressor). Persisted with joblib under `model_dir`.
   - Returns the fitted model object for integration into inference.

Targets & Ordering:
-------------------
Targets are predicted in grouped passes to respect dependencies. Only the
over/under targets present in FeatureEngine's `feature_dependencies` are
kept here (legacy attempted/rush_attempts over/under thresholds removed).

Extending:
----------
Add new target names to the appropriate group lists (or create a new group
if dependency layering changes) and ensure your models dict contains the
`POSITION_target` key with a `.predict(2D_array)` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List, Optional, Sequence, Callable, Tuple
import os
import logging
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import traceback
import asyncio
from asyncio import Semaphore
from copy import deepcopy
import hashlib
import boto3
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

try:
    from .features import FeatureEngine
    from .trainer import PlayerModelTrainer
    from ..data_object import DataObject
except ImportError:
    from features import FeatureEngine
    from trainer import PlayerModelTrainer
    # Get the absolute path of the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(current_dir))
    from data_object import DataObject

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# PlayerPredictor (Optimized Inference)
# ---------------------------------------------------------------------------
@dataclass
class PlayerPredictor:
    data_obj: DataObject
    min_games: int = 2
    root_dir: str = "./"
    
    # Internal caches (populated during __post_init__)
    models: Dict[str, Any] = field(default_factory=dict, init=False)
    scalers: Dict[str, Any] = field(default_factory=dict, init=False)
    model_feature_names: Dict[str, List[str]] = field(default_factory=dict, init=False)
    model_dir: str = field(default="", init=False)
    features_dir: str = field(default="", init=False)
    
    # Residual training metadata
    global_means: Dict[str, float] = field(default_factory=dict, init=False)
    player_baselines: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    use_residual_training: Dict[str, bool] = field(default_factory=dict, init=False)
    
    # Cached data
    _player_data: pd.DataFrame = field(default=None, init=False)
    _upcoming_games: pd.DataFrame = field(default=None, init=False)
    _starters: pd.DataFrame = field(default=None, init=False)
    _feature_accumulator: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    
    # Dependency ordering for predictions
    _target_order: Dict[str, int] = field(default_factory=lambda: {
        # Base volume (no dependencies)
        'attempted_passes': 0, 'rush_attempts': 0, 'times_pass_target': 0,
        # Derived volume (depends on base)
        'completed_passes': 1, 'receptions': 1,
        # Efficiency (depends on volume)
        'passing_yards': 2, 'passing_touchdowns': 2, 'interceptions_thrown': 2,
        'rush_yards': 2, 'rush_touchdowns': 2, 'receiving_yards': 2, 'receiving_touchdowns': 2,
        # Fantasy (depends on all)
        'fantasy_points': 3,
        # Over/under (depends on base predictions)
        'over_under_completed_passes_22+': 4, 'over_under_passing_yards_250+': 4,
        'over_under_passing_touchdowns_2+': 4, 'over_under_interceptions_thrown_1+': 4,
        'over_under_rush_yards_60+': 4, 'over_under_rush_touchdowns_1+': 4,
        'over_under_receptions_5+': 4, 'over_under_receiving_yards_60+': 4,
        'over_under_receiving_touchdowns_1+': 4, 
        'over_under_rush_yards_&_receiving_yards_100+': 4,
        'over_under_rush_touchdowns_&_receiving_touchdowns_1+': 4,
    }, init=False)

    _classification_targets: List[str] = field(default_factory=lambda: [
        'over_under_attempted_passes_34+',
        'over_under_completed_passes_22+',
        'over_under_passing_yards_250+',
        'over_under_passing_touchdowns_2+',
        'over_under_interceptions_thrown_1+',
        'over_under_rush_attempts_16+',
        'over_under_rush_yards_60+',
        'over_under_rush_touchdowns_1+',
        'over_under_receptions_5+',
        'over_under_receiving_yards_60+',
        'over_under_receiving_touchdowns_1+',
        'over_under_rush_yards_&_receiving_yards_100+',
        'over_under_rush_touchdowns_&_receiving_touchdowns_1+',
    ])

    def __post_init__(self):
        self.model_dir = os.path.join(self.root_dir, "models")
        self.features_dir = os.path.join(self.root_dir, "predicted_features")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

        # Load all models once
        self._load_all_models()
        
        # Cache player data once
        self._player_data = self.data_obj.player_data.copy()
        
        # Preload game data for PBP columns
        _ = self.data_obj.get_game_data_with_features()
        logging.info(f"Initialized PlayerPredictor with {len(self.models)} models")

        return
    
    def _load_model_from_s3(self, bucket: str, s3_key: str, s3_client) -> Optional[Any]:
        """Load a model file from S3.
        
        Args:
            bucket: S3 bucket name
            s3_key: S3 object key
            s3_client: boto3 S3 client
            
        Returns:
            Loaded model object or None if failed
        """
        try:
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            model_bytes = response['Body'].read()
            model_bundle = joblib.load(BytesIO(model_bytes))
            return model_bundle
        except Exception as e:
            logging.warning(f"Failed loading s3://{bucket}/{s3_key}: {e}")
            return None
    
    def _load_all_models(self) -> None:
        """Load all available model bundles from model_dir or S3."""
        count = 0
        
        if self.data_obj.storage_mode == "s3":
            # Load models from S3
            bucket_name = os.getenv("LEAGUE_PREDICTIONS_BUCKET_NAME")
            if not bucket_name:
                logging.error("LEAGUE_PREDICTIONS_BUCKET_NAME environment variable not set for S3 mode.")
                return
            
            s3_client = boto3.client('s3')
            s3_prefix = 'nfl/players/'
            
            try:
                # List all objects under the nfl/players/ prefix
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
                
                for page in pages:
                    if 'Contents' not in page:
                        continue
                    
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('_model.pkl'):
                            # Extract position and target from S3 key
                            # Expected format: nfl/players/POSITION/TARGET_model.pkl
                            parts = key.replace(s3_prefix, '').split('/')
                            if len(parts) >= 2:
                                position_dir = parts[0]
                                filename = parts[1]
                                target = filename.replace('_model.pkl', '')
                                model_key = f"{position_dir}_{target}"
                                
                                model_bundle = self._load_model_from_s3(bucket_name, key, s3_client)
                                if model_bundle:
                                    if isinstance(model_bundle, dict):
                                        self.models[model_key] = model_bundle.get('model')
                                        self.scalers[model_key] = model_bundle.get('scaler')
                                        self.model_feature_names[model_key] = model_bundle.get('feature_names', [])
                                        # Extract residual training metadata
                                        self.use_residual_training[model_key] = model_bundle.get('use_residual_training', False)
                                        if self.use_residual_training[model_key]:
                                            self.global_means[model_key] = model_bundle.get('global_mean', 0.0)
                                            self.player_baselines[model_key] = model_bundle.get('player_baselines', {})
                                        count += 1
                                    else:
                                        self.models[model_key] = model_bundle
                                        self.use_residual_training[model_key] = False
                                        count += 1
            except Exception as e:
                logging.error(f"Failed listing S3 models: {e}")
        
        else:
            # Load models from local filesystem
            if not os.path.exists(self.model_dir):
                logging.warning(f"Model directory {self.model_dir} does not exist")
                return
            
            for position_dir in os.listdir(self.model_dir):
                position_path = os.path.join(self.model_dir, position_dir)
                if not os.path.isdir(position_path):
                    continue
                
                for filename in os.listdir(position_path):
                    if filename.endswith('_model.pkl'):
                        target = filename.replace('_model.pkl', '')
                        model_key = f"{position_dir}_{target}"
                        model_path = os.path.join(position_path, filename)
                        
                        try:
                            model_bundle = joblib.load(model_path)
                            if isinstance(model_bundle, dict):
                                self.models[model_key] = model_bundle.get('model')
                                self.scalers[model_key] = model_bundle.get('scaler')
                                self.model_feature_names[model_key] = model_bundle.get('feature_names', [])
                                # Extract residual training metadata
                                self.use_residual_training[model_key] = model_bundle.get('use_residual_training', False)
                                if self.use_residual_training[model_key]:
                                    self.global_means[model_key] = model_bundle.get('global_mean', 0.0)
                                    self.player_baselines[model_key] = model_bundle.get('player_baselines', {})
                                count += 1
                            else:
                                self.models[model_key] = model_bundle
                                self.use_residual_training[model_key] = False
                                count += 1
                        except Exception as e:
                            logging.warning(f"Failed loading {model_path}: {e}")
        
        logging.info(f"Loaded {count} model bundles from {self.data_obj.storage_mode} storage")

        return
    
    def _get_upcoming_games(self) -> pd.DataFrame:
        """Lazy-load and cache upcoming games."""
        if self._upcoming_games is not None:
            return self._upcoming_games
        
        previews = self.data_obj.previews.copy()
        if previews.empty:
            return pd.DataFrame()
        
        upcoming = previews[previews['is_home'] == 1].copy()
        upcoming = upcoming.rename(columns={'abbr': 'home_abbr', 'opp_abbr': 'away_abbr'})
        
        current_week = previews['week'].mode()[0] if not previews['week'].empty else previews['week'].iloc[0]
        current_year = previews['year'].mode()[0] if not previews['year'].empty else previews['year'].iloc[0]
        
        last_week = self.data_obj.schedules[
            (self.data_obj.schedules['year'] == current_year) &
            (self.data_obj.schedules['week'] < current_week)
        ]['week'].max() if not self.data_obj.schedules.empty else 0
        
        upcoming['last_week'] = last_week
        upcoming['home_division'] = upcoming['home_abbr'].map(self.data_obj.DIVISION_MAPPINGS).fillna('Unknown')
        upcoming['home_conference'] = upcoming['home_abbr'].map(self.data_obj.CONFERENCE_MAPPINGS).fillna('Unknown')
        upcoming['away_division'] = upcoming['away_abbr'].map(self.data_obj.DIVISION_MAPPINGS).fillna('Unknown')
        upcoming['away_conference'] = upcoming['away_abbr'].map(self.data_obj.CONFERENCE_MAPPINGS).fillna('Unknown')
        upcoming['key'] = upcoming['game_id']
        
        self._upcoming_games = upcoming
        return self._upcoming_games
    
    def _predict_single_target(
        self,
        position: str,
        target: str,
        player_games: pd.DataFrame,
        upcoming_row: pd.Series,
        predictions: Dict[str, float],
        accumulate_features: bool = False
    ) -> Optional[float]:
        """Generate prediction for a single target."""
        model_key = f"{position}_{target}"
        model = self.models.get(model_key)
        
        if model is None:
            return None
        
        # Build feature engine
        fe = FeatureEngine(
            prior_games=player_games,
            target_name=target,
            row=upcoming_row,
            position=position,
            predicted_features=predictions,
            player_data=self.data_obj.player_data,
            player_data_big_plays=self.data_obj.player_data[
                ['key','game_date','home_abbr','away_abbr','abbr','pos', *self.data_obj.big_play_stat_columns]
            ] if not self.data_obj.player_data.empty else pd.DataFrame(),
            standings=self.data_obj.standings,
            team_ranks=self.data_obj.team_ranks,
            player_group_ranks=self.data_obj.player_group_ranks,
            advanced_stat_cols=self.data_obj.advanced_stat_cols,
            big_play_stat_columns=self.data_obj.big_play_stat_columns,
            game_predictions=self.data_obj.next_game_predictions
        )
        
        # Build feature matrix directly without CSV I/O
        key_cols = ['pid', 'game_date', 'abbr', 'key']
        all_features = []
        
        for group_name, group_df in fe.grouped_features_as_dfs.items():
            group_df = group_df.copy()
            # Add key columns
            group_df['pid'] = upcoming_row['pid']
            group_df['game_date'] = upcoming_row['game_date']
            group_df['abbr'] = upcoming_row['abbr']
            group_df['key'] = upcoming_row['key']
            
            # Reorder columns: key columns first, then features
            feature_cols = [col for col in group_df.columns if col not in key_cols]
            group_df = group_df[key_cols + feature_cols]
            
            # Accumulate features if requested (for saving like trainer)
            if accumulate_features:
                fe_key = f"{position}-{target}-{group_name}"
                if fe_key not in self._feature_accumulator:
                    self._feature_accumulator[fe_key] = group_df.copy()
                else:
                    self._feature_accumulator[fe_key] = pd.concat(
                        [self._feature_accumulator[fe_key], group_df], 
                        ignore_index=True
                    )
            
            # Add group_name suffix to features AFTER accumulating
            group_df.columns = [
                f"{col}_{group_name}" if col not in key_cols else col 
                for col in group_df.columns
            ]
                    
            all_features.append(group_df)
        
        # Merge all feature groups
        if not all_features:
            return None
        
        merged_df = all_features[0]
        for df in all_features[1:]:
            merged_df = merged_df.merge(df, on=key_cols, how='outer', suffixes=('', '_dup'))
            # Drop duplicate columns
            merged_df = merged_df[[c for c in merged_df.columns if not c.endswith('_dup')]]
        
        # Prepare feature matrix
        X_pred = merged_df.drop(columns=key_cols, errors='ignore')
        X_pred = X_pred.select_dtypes(exclude=[object])
        
        # Align to training features
        feature_names = self.model_feature_names.get(model_key, [])
        if feature_names:
            # Add missing columns with zeros
            for col in feature_names:
                if col not in X_pred.columns:
                    X_pred[col] = 0.0
            # Reorder to match training
            X_pred = X_pred[feature_names]
        
        X_pred = X_pred.fillna(0.0)
        
        # Scale and predict
        scaler = self.scalers.get(model_key)
        if scaler is not None:
            X_pred_scaled = scaler.transform(X_pred)
        else:
            X_pred_scaled = X_pred.values
        
        prediction = float(model.predict(X_pred_scaled)[0])
        
        # Denormalize residual prediction if model was trained with residuals
        if self.use_residual_training.get(model_key, False):
            pid = upcoming_row['pid']
            game_date = upcoming_row['game_date']
            player_baseline_dict = self.player_baselines.get(model_key, {})
            global_mean = self.global_means.get(model_key, 0.0)
            
            # Check if baseline dict uses time-varying (tuple) or static (string) keys
            if player_baseline_dict:
                sample_key = next(iter(player_baseline_dict.keys()))
                if isinstance(sample_key, tuple):
                    # Time-varying baselines (e.g., EWM): find most recent baseline for player
                    # Since upcoming game_date is new, get the last available baseline
                    player_baselines_filtered = {
                        (p, date): val 
                        for (p, date), val in player_baseline_dict.items() 
                        if p == pid
                    }
                    
                    if player_baselines_filtered:
                        # Get the most recent baseline (max date)
                        most_recent_key = max(player_baselines_filtered.keys(), key=lambda x: x[1])
                        player_baseline = player_baselines_filtered[most_recent_key]
                    else:
                        # Player not in training data, use global mean
                        player_baseline = global_mean
                else:
                    # Static baselines (e.g., career mean): use pid key
                    player_baseline = player_baseline_dict.get(pid, global_mean)
            else:
                player_baseline = global_mean

            # Denormalize: actual_prediction = residual_prediction + player_baseline
            prediction = prediction + player_baseline
            
            logging.debug(f"Denormalized residual prediction for {pid} {target}: "
                         f"residual={prediction - player_baseline:.2f}, baseline={player_baseline:.2f}, "
                         f"final={prediction:.2f}")
        
        return prediction
    
    def predict_single_player(
        self, 
        pid: str, 
        position: str, 
        target_subset: Optional[List[str]] = None,
        show: Optional[bool] = False,
        accumulate_features: bool = False
    ) -> Optional[Dict[str, float]]:
        """Predict targets for a single player.
        
        Args:
            pid: Player ID
            position: Player position (e.g., 'QB', 'RB', 'WR', 'TE')
            target_subset: Optional list of specific targets. If None, predict all available.
            
        Returns:
            Dictionary of predictions {target: value} or None if prediction fails
        """
        # Get player games
        player_games = self._player_data[
            (self._player_data['pid'] == pid) & 
            (self._player_data['pos'] == position)
        ].sort_values('game_date')
        
        if len(player_games) < self.min_games:
            logging.warning(f"Player {pid} has only {len(player_games)} games (need {self.min_games})")
            return None
        
        # Get upcoming game
        team_abbr = player_games['abbr'].iloc[-1]
        upcoming_games = self._get_upcoming_games()
        
        if upcoming_games.empty:
            logging.warning("No upcoming games available")
            return None
        
        upcoming_game = upcoming_games[
            (upcoming_games['home_abbr'] == team_abbr) | 
            (upcoming_games['away_abbr'] == team_abbr)
        ]
        
        if upcoming_game.empty:
            logging.warning(f"No upcoming game for team {team_abbr}")
            return None
        
        # Prepare upcoming row
        upcoming_row = upcoming_game.iloc[0].copy()
        upcoming_row['pid'] = pid
        upcoming_row['abbr'] = team_abbr
        upcoming_row['position'] = position
        upcoming_row['starter'] = int(pid in self.data_obj.starters_new['player_id'].values)
        
        # Check game is in future
        if upcoming_row['game_date'] <= pd.Timestamp(datetime.now().date()):
            logging.warning(f"Game date {upcoming_row['game_date']} is not in the future")
            return None
        
        # Determine targets to predict
        if target_subset:
            available_targets = [t for t in target_subset if f"{position}_{t}" in self.models]
        else:
            available_targets = [
                key.replace(f"{position}_", "") 
                for key in self.models.keys() 
                if key.startswith(f"{position}_")
            ]
        
        if not available_targets: 
            logging.debug(f"No models available for {position}")
            return None
        
        # Sort targets by dependency order
        ordered_targets = sorted(
            available_targets, 
            key=lambda t: self._target_order.get(t, 999)
        )
        
        # Generate predictions
        predictions = {}
        for target in ordered_targets:
            try:
                pred = self._predict_single_target(
                    position, target, player_games, upcoming_row, predictions, accumulate_features
                )
                if pred is not None:
                    predictions[target] = pred
            except Exception as e:
                logging.warning(f"Failed to predict {target} for {pid}: {e}")
        
        if predictions:
            predictions.update({
                'pid': pid,
                'position': position,
                'team_abbr': team_abbr,
                'player_name': player_games['player_name'].iloc[-1] if 'player_name' in player_games.columns else ''
            })
            if show:
                logging.info(f"Generated {len(predictions)-3} predictions for {pid}")
                for k, v in predictions.items():
                    if k not in ['pid', 'position', 'team_abbr', 'player_name']:
                        print(f"  {k}: {v:.2f}")
        
        return predictions if predictions else None
    
    def predict_next_players(
        self,
        positions: Optional[List[str]] = None,
        save_results: bool = True,
        save_features: bool = True
    ) -> Optional[pd.DataFrame]:
        """Predict for all starters in upcoming games.
        
        Args:
            positions: Optional list of positions to predict. If None, predict all.
            save_results: Whether to save predictions to CSV
            save_features: Whether to save feature groupings (like trainer)
            
        Returns:
            DataFrame with predictions or None if no predictions generated
        """
        # Clear feature accumulator if saving features
        if save_features:
            self._feature_accumulator.clear()
        
        starters = self.data_obj.starters_new
        if starters.empty:
            logging.warning("No starters data available")
            return None
        
        # Filter by positions if specified
        if positions:
            starters = starters[starters['position'].isin(positions)]
        
        all_predictions = []
        for _, row in tqdm(starters.iterrows(), total=len(starters), desc="Predicting players"):
            pid = row['player_id']
            position = row['position']
            
            predictions = self.predict_single_player(
                pid=pid, 
                position=position, 
                accumulate_features=save_features
            )
            if predictions:
                all_predictions.append(predictions)
        
        if not all_predictions:
            logging.warning("No predictions generated")
            return None
        
        df = pd.DataFrame(all_predictions)
        logging.info(f"Generated predictions for {len(df)} players")
        
        if save_results:
            local_predictions_dir = './player_predictions/'
            os.makedirs(local_predictions_dir, exist_ok=True)
            df = df[['pid', 'player_name', 'position', 'team_abbr'] + [col for col in df.columns if col not in ['pid', 'player_name', 'position', 'team_abbr']]]
            for position, group_df in df.groupby('position'):
                filepath = os.path.join(local_predictions_dir, f'{position}_next_player_predictions.csv')
                group_df.sort_values(by=['fantasy_points'], ascending=False).round(2).to_csv(filepath, index=False)
                logging.info(f"Saved predictions to {filepath}")
        
        # Save feature groupings if requested
        if save_features and self._feature_accumulator:
            self._save_feature_groupings()
        
        return df
    
    def _save_feature_groupings(self) -> None:
        """Save accumulated feature groupings to CSV files (trainer format)."""
        for fe_key, df in self._feature_accumulator.items():
            pos, target, group_name = fe_key.split("-")
            _dir = os.path.join(self.features_dir, pos, target)
            os.makedirs(_dir, exist_ok=True)
            filepath = os.path.join(_dir, f"{group_name}_features.csv")
            df.to_csv(filepath, index=False)
            logging.debug(f"Saved feature group: {fe_key}, Shape: {df.shape} -> {filepath}")
        
        logging.info(f"Saved {len(self._feature_accumulator)} feature groupings")
        self._feature_accumulator.clear()

if __name__ == "__main__":
    # Example usage
    data_obj = DataObject(
        league='nfl',
        storage_mode='local',
        local_root=os.path.join(sys.path[0], "..", "..", "..", "..", "sports-data-storage-copy/")
    )

    # data_obj = DataObject(
    #     storage_mode='s3',
    #     s3_bucket=os.getenv('SPORTS_DATA_BUCKET_NAME')
    # )
    
    predictor = PlayerPredictor(data_obj=data_obj)
    
    # Example 1: Predict for a single player (all targets)
    # predictor.predict_single_player(pid='ChasJa00', position='WR', show=True)
    # predictor.predict_single_player(pid='LoveJo03', position='QB', show=True)
    
    # Example 2: Predict for all starters
    predictor.predict_next_players()

