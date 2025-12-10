"""Lean game prediction and training interfaces.

This module mirrors the structure of the player predictor but for game-level
predictions. Feature construction is exclusively delegated to `FeatureEngine`
(see `features.py`). Data loading, storage mode abstraction (local vs S3), and
ancillary metadata are handled by `DataObject` (see `data_object.py`).

Provided classes:
-----------------
1. GamePredictor (inference only)
   - Consumes pre-trained per-target models you supply.
   - Iteratively predicts targets in dependency order so later targets can
     incorporate earlier predictions (e.g. total_points depends on home_points
     and away_points).
   - Uses a private helper to instantiate FeatureEngine consistently.

2. GameModelTrainer (simple training helper)
   - Builds per-target training rows from historical games.
   - Trains a baseline model (RandomForestRegressor if available; else a
     tiny MeanRegressor). Persisted with joblib under `model_dir`.
   - Returns the fitted model object for integration into inference.

Targets & Ordering:
-------------------
Targets are predicted in grouped passes to respect dependencies.

Extending:
----------
Add new target names to the appropriate group lists (or create a new group
if dependency layering changes) and ensure your models dict contains the
`target` key with a `.predict(2D_array)` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import traceback
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
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

try:
    from .features import FeatureEngine
    from ..data_object import DataObject
    from ..helpers import upload_file_to_s3
except ImportError:
    from features import FeatureEngine
    # Get the absolute path of the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(current_dir))
    from data_object import DataObject
    from helpers import upload_file_to_s3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Minimal fallback model (used when a factory isn't provided on training)
# ---------------------------------------------------------------------------
class MeanRegressor:
    """Extremely small baseline: predicts global mean of y."""
    def __init__(self):
        self.mean_: float = 0.0
    def fit(self, X: np.ndarray, y: Sequence[float]):  # type: ignore[override]
        self.mean_ = float(np.mean(y)) if len(y) else 0.0
    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


def default_model_factory() -> Any:
    """Picklable default model factory used by GameModelTrainer.

    Avoid using lambdas/closures to keep the trainer instance picklable for
    process-based parallelism on Windows (spawn).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor()
    except Exception:
        return MeanRegressor()

# ---------------------------------------------------------------------------
# GamePredictor (Inference)
# ---------------------------------------------------------------------------
@dataclass
class GamePredictor:
    data_obj: DataObject
    models: Dict[str, Any]  # key: target -> fitted model
    min_games: int = 3
    root_dir: str = "./"
    model_dir: str = field(default="")  # Will be set in __post_init__
    features_dir: str = field(default="")  # Will be set in __post_init__
    use_saved_scalers: bool = False
    scalers: Dict[str, Any] = field(default_factory=dict)  # optional per-target scaler
    model_feature_names: Dict[str, List[str]] = field(default_factory=dict)  # stored feature ordering per model

    # Target classification (matching FeatureEngine structure)
    base_volume_stats: List[str] = field(default_factory=lambda: [
        'pass_attempts', 'rush_attempts', 'win'
    ])
    efficiency_stats: List[str] = field(default_factory=lambda: [
        'pass_yards', 'rush_yards', 'total_yards', 'points'
    ])
    
    # Feature dependencies (matching FeatureEngine)
    feature_dependencies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
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

    def __post_init__(self):
        # Set directory paths based on root_dir
        if not self.model_dir:
            self.model_dir = os.path.join(self.root_dir, "models")
        if not self.features_dir:
            self.features_dir = os.path.join(self.root_dir, "predicted_features")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # CRITICAL: Force DataObject to load game data and populate PBP column lists
        # This must happen BEFORE caching column lists in _fe_common_params
        # Otherwise all PBP-related column lists will be empty []
        _ = self.data_obj.get_game_data_with_features()
        logging.info("Preloaded game data to populate PBP column lists")
        logging.info(f"PBP columns available - play_type: {len(self.data_obj.play_type_columns)}, "
                    f"redzone: {len(self.data_obj.redzone_columns)}, "
                    f"team_epa: {len(self.data_obj.team_epa_columns)}, "
                    f"big_play_pos: {len(self.data_obj.big_play_position_columns)}, "
                    f"player_epa_pos: {len(self.data_obj.player_epa_position_columns)}")
        
        # Cache team position ratings to avoid recalculation on every feature engine build
        self._team_position_ratings = self.data_obj.get_team_position_ratings()
        self._available_positions = sorted(self._team_position_ratings['pos'].unique()) if not self._team_position_ratings.empty else []
        
        # Preload new officials features to cache them before predictions
        if hasattr(self.data_obj, 'new_officials_with_features'):
            _ = self.data_obj.new_officials_with_features
            logging.info("Preloaded new officials features")

        self.features_cache = {}
        
        # Cache common FeatureEngine parameters to avoid repeated lookups
        self._fe_common_params = {
            'data_obj': self.data_obj,
            'schedules': self.data_obj.schedules,
            'standings': self.data_obj.standings,
            'team_ranks': self.data_obj.team_ranks,
            'team_ranks_str_cols': self.data_obj.team_ranks_str_cols,
            'team_position_ratings': self._team_position_ratings,
            'redzone_columns': getattr(self.data_obj, 'redzone_columns', []),
            'team_epa_columns': getattr(self.data_obj, 'team_epa_columns', []),
            'play_type_columns': getattr(self.data_obj, 'play_type_columns', []),
            'yards_togo_columns': getattr(self.data_obj, 'yards_togo_columns', []),
            'yards_gained_columns': getattr(self.data_obj, 'yards_gained_columns', []),
            'big_play_position_columns': getattr(self.data_obj, 'big_play_position_columns', []),
            'player_epa_position_columns': getattr(self.data_obj, 'player_epa_position_columns', []),
            'available_positions': self._available_positions,
        }

    def _get_prediction_order(self, available_targets: List[str]) -> List[str]:
        """Determine prediction order based on dependencies.
        
        Base volume stats (pass_attempts, rush_attempts, win) have no dependencies
        and are predicted first. Efficiency stats depend on volume stats.
        
        Args:
            available_targets: List of target names that have trained models
            
        Returns:
            Ordered list of targets to predict
        """
        base_targets = []
        efficiency_targets = []
        
        for target in available_targets:
            # Extract base stat name (remove home_/away_ prefix)
            if target.startswith('home_') or target.startswith('away_'):
                base_stat = target.split('_', 1)[1]
            else:
                base_stat = target
            
            # Classify as base or efficiency
            if base_stat in self.base_volume_stats:
                base_targets.append(target)
            elif base_stat in self.efficiency_stats:
                efficiency_targets.append(target)
            else:
                # Unknown target, add to base by default
                base_targets.append(target)
        
        # Return base targets first, then efficiency targets
        return base_targets + efficiency_targets

    def _build_feature_engine(
        self,
        game_data: pd.DataFrame,
        target: str,
        row: pd.Series,
        predictions: Optional[Dict[str, Any]] = None,
    ) -> FeatureEngine:
        return FeatureEngine(
            game_data=game_data,
            target_name=target,
            row=row,
            predicted_features=predictions,
            **self._fe_common_params
        )

    def predict_game(
        self,
        game_data: pd.DataFrame,
        upcoming_row: pd.Series,
        target_subset: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Predict selected targets for a single upcoming game using merged features.

        This method follows the notebook workflow:
        1. Build prediction features and save feature groupings
        2. Merge all feature grouping CSVs for each target
        3. Drop key columns, align to training columns, fillna(0.0)
        4. Scale using saved scaler
        5. Predict with saved model

        Parameters
        ----------
        game_data : DataFrame
            Historical games data.
        upcoming_row : Series
            Row describing the upcoming game context.
        target_subset : Sequence[str], optional
            If provided, restrict predictions to these targets.
        """
        predictions: Dict[str, float] = {}
        
        # Determine which targets to predict
        if target_subset:
            available_targets = [t for t in target_subset if t in self.models]
        else:
            available_targets = list(self.models.keys())
        
        # Get dependency-based prediction order
        ordered_targets = self._get_prediction_order(available_targets)
        
        # Key columns to drop (like the notebook)
        key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr']
        
        for target in ordered_targets:
            model = self.models.get(target)
            scaler = self.scalers.get(target)
            feature_names = self.model_feature_names.get(target)
            
            if model is None:
                logging.warning(f"No model found for {target}, skipping")
                continue
            
            # Build feature engine for this target
            fe = self._build_feature_engine(
                game_data=game_data,
                target=target,
                row=upcoming_row,
                predictions=predictions if predictions else None,
            )
            
            # Save feature groupings to CSV
            feat_dir = os.path.join(self.features_dir, target)
            os.makedirs(feat_dir, exist_ok=True)
            
            for key_name, grouped in fe.grouped_features_as_dfs.items():
                fe_key = f"{target}_{key_name}"
                grouped = grouped.copy()
                grouped['game_id'] = upcoming_row.get('game_id', upcoming_row.get('key', np.nan))
                grouped['game_date'] = upcoming_row.get('game_date', np.nan)
                grouped['home_abbr'] = upcoming_row.get('home_abbr', np.nan)
                grouped['away_abbr'] = upcoming_row.get('away_abbr', np.nan)
                key_cols_for_grouping = ['game_id', 'game_date', 'home_abbr', 'away_abbr']
                grouped = grouped[key_cols_for_grouping + [c for c in grouped.columns if c not in key_cols_for_grouping]]
                
                # Save to CSV for merging
                feat_path = os.path.join(feat_dir, f"{fe_key}.csv")
                grouped.to_csv(feat_path, index=False)
            
            # Merge all feature groupings from CSV (like the notebook)
            merged_df = self.merge_feature_groupings(target)
            
            if merged_df.empty:
                logging.warning(f"Failed to merge feature groupings for {target}, skipping")
                continue
            
            # Drop key columns and align to training columns
            X_pred = merged_df.drop(columns=key_cols, errors='ignore')
            
            # Align to training feature names if available
            if feature_names:
                # Ensure we have exactly the features the model was trained on
                missing_cols = [col for col in feature_names if col not in X_pred.columns]
                extra_cols = [col for col in X_pred.columns if col not in feature_names]
                
                if missing_cols:
                    logging.debug(f"Adding {len(missing_cols)} missing columns for {target}")
                    for col in missing_cols:
                        X_pred[col] = 0.0
                
                if extra_cols:
                    logging.debug(f"Dropping {len(extra_cols)} extra columns for {target}")
                
                # Reorder to match training
                X_pred = X_pred[feature_names]
            
            # Fill NaN with 0.0 BEFORE scaling (like the notebook)
            X_pred = X_pred.fillna(0.0)
            
            # Scale using saved scaler
            if scaler is not None:
                X_pred_scaled = scaler.transform(X_pred)
            else:
                logging.warning(f"No scaler found for {target}, using unscaled features")
                X_pred_scaled = X_pred.values
            
            # Predict
            pred_val = float(model.predict(X_pred_scaled)[0])
            predictions[target] = pred_val
            logging.debug(f"Predicted {target} = {pred_val:.2f}")

        logging.info(f"Predictions completed for {upcoming_row.get('game_id', upcoming_row.get('key', ''))}")

        return predictions

    def load_models_from_dir(self, targets: Sequence[str]) -> None:
        """Load models from `model_dir` named as target.pkl (joblib)."""
        for tgt in targets:
            key = tgt
            path = os.path.join(self.model_dir, f"{key}.pkl")
            if not os.path.exists(path):
                continue
            try:
                obj = joblib.load(path)
                if isinstance(obj, dict) and 'model' in obj:
                    self.models[key] = obj['model']
                    if 'scaler' in obj and obj['scaler'] is not None:
                        self.scalers[key] = obj['scaler']
                    if 'feature_names' in obj and isinstance(obj['feature_names'], list):
                        self.model_feature_names[key] = obj['feature_names']
                else:  # backward compatibility: raw model without metadata
                    self.models[key] = obj
                logging.info(f"Loaded model: {path}")
            except Exception as e:
                logging.warning(f"Failed loading {path}: {e}")

    def load_all_models(self) -> None:
        """Load all available models from model_dir by discovering .pkl files."""
        if not os.path.exists(self.model_dir):
            logging.warning(f"Model directory {self.model_dir} does not exist")
            return
        
        # Discover all .pkl files in model_dir
        discovered_targets = []
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pkl'):
                target_name = filename[:-4]  # Remove .pkl extension
                discovered_targets.append(target_name)
        
        if discovered_targets:
            logging.info(f"Discovered {len(discovered_targets)} model(s): {discovered_targets}")
            self.load_models_from_dir(discovered_targets)
        else:
            logging.warning(f"No .pkl models found in {self.model_dir}")

    def prepare_upcoming_games(self) -> pd.DataFrame:
        """Prepare upcoming games DataFrame from data_obj.previews."""
        previews = self.data_obj.previews.copy()
        if previews.empty:
            logging.warning("No previews available in data_obj.previews")
            return pd.DataFrame()
        
        # Filter to home team rows to avoid duplicates
        upcoming_games = previews[previews['is_home'] == 1].copy()
        upcoming_games = upcoming_games.rename(columns={'abbr': 'home_abbr', 'opp_abbr': 'away_abbr'})
        
        # Get current week/year info
        current_week = previews['week'].mode()[0] if not previews['week'].empty else previews['week'].iloc[0]
        current_year = previews['year'].mode()[0] if not previews['year'].empty else previews['year'].iloc[0]
        
        # Find last completed week
        last_week = self.data_obj.schedules[
            (self.data_obj.schedules['year'] == current_year) &
            (self.data_obj.schedules['week'] < current_week)
        ]['week'].max() if not self.data_obj.schedules.empty else 0
        
        upcoming_games['last_week'] = last_week

        # Add divisions and conferences for home and away teams
        upcoming_games['home_division'] = upcoming_games['home_abbr'].map(self.data_obj.DIVISION_MAPPINGS).fillna('Unknown')
        upcoming_games['home_conference'] = upcoming_games['home_abbr'].map(self.data_obj.CONFERENCE_MAPPINGS).fillna('Unknown')
        upcoming_games['away_division'] = upcoming_games['away_abbr'].map(self.data_obj.DIVISION_MAPPINGS).fillna('Unknown')
        upcoming_games['away_conference'] = upcoming_games['away_abbr'].map(self.data_obj.CONFERENCE_MAPPINGS).fillna('Unknown')
        
        return upcoming_games

    def predict_single_game(self, abbr: str):
        """ Get data for a single upcoming game by team abbreviation and predict it. """
        upcoming_games = self.prepare_upcoming_games()
        game_row = upcoming_games[
            (upcoming_games['home_abbr'] == abbr) | 
            (upcoming_games['away_abbr'] == abbr)
        ]

        if game_row.empty:
            logging.warning(f"No upcoming game found for team abbreviation: {abbr}")
            return {}
        
        game_data = self.data_obj.get_game_data_with_features()
        predictions = self.predict_game(game_data, game_row.iloc[0])
        return predictions

    def merge_feature_groupings(self, target: str) -> pd.DataFrame:
        """Merge all feature grouping files for a target into one DataFrame.
        
        Reads all CSV files from self.features_dir/{target}/ and merges them
        on the key columns (game_id, game_date, home_abbr, away_abbr).
        Note: Prediction features don't have 'target' column.
        
        Args:
            target: Target name (e.g., 'home_points', 'away_rush_yards')
            
        Returns:
            Merged DataFrame with all features for the target
        """
        target_dir = os.path.join(self.features_dir, target)
        if not os.path.exists(target_dir):
            logging.warning(f"Feature directory does not exist: {target_dir}")
            return pd.DataFrame()
        
        # Find all CSV files in the target directory
        csv_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
        if not csv_files:
            logging.warning(f"No CSV files found in {target_dir}")
            return pd.DataFrame()
        
        logging.info(f"Merging {len(csv_files)} feature grouping files for {target}")
        
        # Key columns that should appear in all files (predictions don't have 'target')
        key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr']
        
        merged_df = None
        for csv_file in csv_files:
            file_path = os.path.join(target_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                if merged_df is None:
                    # First file - use it as base
                    merged_df = df
                else:
                    # Merge with existing data
                    # Identify feature columns (non-key columns)
                    feature_cols = [col for col in df.columns if col not in key_cols]
                    
                    # Merge on key columns, adding only the feature columns
                    merged_df = merged_df.merge(
                        df[key_cols + feature_cols],
                        on=key_cols,
                        how='outer'
                    )
            except Exception as e:
                logging.warning(f"Failed to read {csv_file}: {e}")
                continue
        
        if merged_df is not None:
            logging.info(f"Merged features for {target}: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        else:
            logging.warning(f"Failed to merge any features for {target}")
            merged_df = pd.DataFrame()
        
        return merged_df

    def predict_all_next_games(self) -> pd.DataFrame:
        """Predict all upcoming games from data_obj.previews using all loaded models.
        
        Returns:
            DataFrame with columns: game_id, game_date, home_abbr, away_abbr, 
                                   and one column per target with predictions
        """
        if not self.models:
            logging.warning("No models loaded. Call load_all_models() or load_models_from_dir() first.")
            return pd.DataFrame()
        
        upcoming_games = self.prepare_upcoming_games()
        if upcoming_games.empty:
            logging.warning("No upcoming games to predict")
            return pd.DataFrame()
        
        # Get historical game data
        game_data = self.data_obj.get_game_data_with_features()
        
        # Predict for each upcoming game
        all_predictions = []
        for idx, row in upcoming_games.iterrows():
            logging.info(f"Predicting game: {row.get('home_abbr')} vs {row.get('away_abbr')}")
            
            try:
                predictions = self.predict_game(game_data, row)
                
                # Combine game info with predictions
                result = {
                    'game_id': row.get('game_id', row.get('key', '')),
                    'game_date': row.get('game_date'),
                    'home_abbr': row.get('home_abbr'),
                    'away_abbr': row.get('away_abbr'),
                    'week': row.get('week'),
                    'year': row.get('year'),
                }
                result.update(predictions)
                all_predictions.append(result)
            except Exception as e:
                logging.warning(f"Failed to predict game {row.get('game_id', '')}: {e}")
                continue
        
        if not all_predictions:
            logging.warning("No predictions were generated")
            return pd.DataFrame()
        
        predictions_df = pd.DataFrame(all_predictions)
        logging.info(f"Generated predictions for {len(predictions_df)} games")
        return predictions_df

# ---------------------------------------------------------------------------
# GameModelTrainer (Training)
# ---------------------------------------------------------------------------
@dataclass
class GameModelTrainer:
    data_obj: DataObject
    min_games: int = 3
    model_factory: Optional[Callable[[], Any]] = None
    root_dir: str = "./"
    model_dir: str = field(default="")  # Will be set in __post_init__
    features_dir: str = field(default="")  # Will be set in __post_init__
    testing: bool = False  # If True, sample a fraction of data for faster testing
    load_from_file: bool = False  # If True, load existing saved features from CSVs
    targets: Dict[str, List[str]] = field(default_factory=lambda: {
        'regression': [
            'home_points', 'away_points',
            'home_total_yards', 'away_total_yards',
            'home_pass_yards', 'away_pass_yards',
            'home_rush_yards', 'away_rush_yards',
            'home_pass_attempts', 'away_pass_attempts',
            'home_rush_attempts', 'away_rush_attempts'
        ],
        'classification': [
            'home_win'
        ]
    })

    def __post_init__(self):
        # Set directory paths based on root_dir
        if not self.model_dir:
            self.model_dir = os.path.join(self.root_dir, "models")
        if not self.features_dir:
            self.features_dir = os.path.join(self.root_dir, "features")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        if self.model_factory is None:
            # Use a top-level function to keep instance picklable in parallel
            self.model_factory = default_model_factory

        # CRITICAL: Force DataObject to load game data and populate PBP column lists
        # This must happen BEFORE caching column lists in _fe_common_params
        # Otherwise all PBP-related column lists will be empty []
        self.game_data = self.data_obj.get_game_data_with_features()
        logging.info("Preloaded game data to populate PBP column lists")
        logging.info(f"PBP columns available - play_type: {len(self.data_obj.play_type_columns)}, "
                    f"redzone: {len(self.data_obj.redzone_columns)}, "
                    f"team_epa: {len(self.data_obj.team_epa_columns)}, "
                    f"big_play_pos: {len(self.data_obj.big_play_position_columns)}, "
                    f"player_epa_pos: {len(self.data_obj.player_epa_position_columns)}")

        self.features_cache = {}
        # Cache team position ratings to avoid recalculation on every feature engine build
        self._team_position_ratings = self.data_obj.get_team_position_ratings()
        self._available_positions = sorted(self._team_position_ratings['pos'].unique()) if not self._team_position_ratings.empty else []
        
        # Cache common FeatureEngine parameters to avoid repeated lookups
        self._fe_common_params = {
            'data_obj': self.data_obj,
            'schedules': self.data_obj.schedules,
            'standings': self.data_obj.standings,
            'team_ranks': self.data_obj.team_ranks,
            'team_ranks_str_cols': self.data_obj.team_ranks_str_cols,
            'team_position_ratings': self._team_position_ratings,
            'redzone_columns': getattr(self.data_obj, 'redzone_columns', []),
            'team_epa_columns': getattr(self.data_obj, 'team_epa_columns', []),
            'play_type_columns': getattr(self.data_obj, 'play_type_columns', []),
            'yards_togo_columns': getattr(self.data_obj, 'yards_togo_columns', []),
            'yards_gained_columns': getattr(self.data_obj, 'yards_gained_columns', []),
            'big_play_position_columns': getattr(self.data_obj, 'big_play_position_columns', []),
            'player_epa_position_columns': getattr(self.data_obj, 'player_epa_position_columns', []),
            'available_positions': self._available_positions,
        }

    def _build_feature_engine(
        self,
        game_data: pd.DataFrame,
        target: str,
        row: pd.Series,
        predictions: Optional[Dict[str, Any]] = None,
    ) -> FeatureEngine:
        return FeatureEngine(
            game_data=game_data,
            target_name=target,
            row=row,
            predicted_features=predictions,
            **self._fe_common_params
        )
    
    def _test_build_feature_engine(self) -> None:
        """Test helper to validate FeatureEngine instantiation."""
        game_data = self.game_data.copy()
        if game_data.empty:
            logging.warning("No game data available for FeatureEngine test")
            return
        
        sample_row = game_data.iloc[-1]
        target = 'home_points'
        
        try:
            fe = self._build_feature_engine(
                game_data=game_data,
                target=target,
                row=sample_row,
                predictions=None
            )
            logging.info(f"FeatureEngine test successful for target {target}")
            
            for fe_key, feat_df in fe.grouped_features_as_dfs.items():
                logging.info(f"Feature grouping '{fe_key}': {len(feat_df)} rows, {len(feat_df.columns)} columns")

        except Exception as e:
            logging.error(f"FeatureEngine test failed: {e}")
            traceback.print_exc()

    def _build_single_training_row(self, game_data: pd.DataFrame, idx: int, target: str) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """Build a single training row (thread-safe helper).
        
        Args:
            game_data: Full game data (sorted by date)
            idx: Index of the game to build features for
            target: Target variable name
            
        Returns:
            Tuple of (feature_dict, grouped_features_dict)
        """
        current = game_data.iloc[idx]
        
        fe = self._build_feature_engine(
            game_data=game_data.iloc[:idx],  # Use only prior games
            target=target,
            row=current,
            predictions=None  # No predictions for training
        )
        
        # Build grouped features with metadata
        grouped_features = {}
        for key_name, grouped in fe.grouped_features_as_dfs.items():
            fe_key = f"{target}_{key_name}"
            grouped = grouped.copy()  # Thread-safe copy
            grouped['game_id'] = current.get('game_id', current.get('key', np.nan))
            grouped['game_date'] = current.get('game_date', np.nan)
            grouped['home_abbr'] = current.get('home_abbr', np.nan)
            grouped['away_abbr'] = current.get('away_abbr', np.nan)
            grouped['target'] = current.get(target, np.nan)
            key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target']
            grouped = grouped[key_cols + [c for c in grouped.columns if c not in key_cols]]
            grouped_features[fe_key] = grouped
        
        # Build feature row
        feat = fe.features.copy()
        feat[f"target_{target}"] = current.get(target, np.nan)
        
        return feat, grouped_features

    def build_training_rows(self, game_data: pd.DataFrame, target: str, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Build training rows from historical game data using parallel processing.
        
        Args:
            game_data: DataFrame with historical game data
            target: Target variable name
            max_workers: Maximum number of threads (default: None = CPU count)
            
        Returns:
            List of feature dictionaries for training
        """
        rows: List[Dict[str, Any]] = []
        game_data = game_data.sort_values('game_date')
        
        # Filter to games with actual results
        game_data = game_data[game_data['home_points'].notna() & game_data['away_points'].notna()]
        
        # Thread-safe lock for features_cache updates
        cache_lock = Lock()
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(os.cpu_count() or 8, 16)  # Cap at 16 threads
        
        indices = list(range(self.min_games, len(game_data)))
        total_games = len(indices)
        
        logging.info(f"Building {total_games} training rows for {target} using {max_workers} threads")
        
        # Process games in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._build_single_training_row, game_data, idx, target): idx
                for idx in indices
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_idx), total=total_games, desc=f"Training rows ({target})"):
                idx = future_to_idx[future]
                try:
                    feat, grouped_features = future.result()
                    
                    # Thread-safe append to rows list
                    rows.append(feat)
                    
                    # Thread-safe update to features_cache
                    with cache_lock:
                        for fe_key, grouped in grouped_features.items():
                            if fe_key not in self.features_cache:
                                self.features_cache[fe_key] = grouped
                            else:
                                self.features_cache[fe_key] = pd.concat(
                                    [self.features_cache[fe_key], grouped], 
                                    ignore_index=True
                                )
                except Exception as e:
                    logging.error(f"Failed to build training row at index {idx}: {e}")
                    logging.error(traceback.format_exc())
        
        # Sort rows by game_date to maintain chronological order
        if rows:
            # Extract game_date from first row to determine sorting strategy
            rows_df = pd.DataFrame(rows)
            if 'game_date' in rows_df.columns:
                rows_df = rows_df.sort_values('game_date')
                rows = rows_df.to_dict('records')
        
        logging.info(f"Completed building {len(rows)} training rows for {target}")
        
        return rows

    def merge_feature_groupings(self, target: str) -> pd.DataFrame:
        """Merge all feature grouping files for a target into one DataFrame.
        
        Reads all CSV files from self.features_dir/{target}/ and merges them
        on the key columns (game_id, game_date, home_abbr, away_abbr, target).
        
        Args:
            target: Target name (e.g., 'home_points', 'away_rush_yards')
            
        Returns:
            Merged DataFrame with all features for the target
        """
        target_dir = os.path.join(self.features_dir, target)
        if not os.path.exists(target_dir):
            logging.warning(f"Feature directory does not exist: {target_dir}")
            return pd.DataFrame()
        
        # Find all CSV files in the target directory
        csv_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
        if not csv_files:
            logging.warning(f"No CSV files found in {target_dir}")
            return pd.DataFrame()
        
        logging.info(f"Merging {len(csv_files)} feature grouping files for {target}")
        
        # Key columns that should appear in all files
        key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target']
        
        merged_df = None
        for csv_file in csv_files:
            file_path = os.path.join(target_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                if merged_df is None:
                    # First file - use it as base
                    merged_df = df
                else:
                    # Merge with existing data
                    # Identify feature columns (non-key columns)
                    feature_cols = [col for col in df.columns if col not in key_cols]
                    
                    # Merge on key columns, adding only the feature columns
                    merged_df = merged_df.merge(
                        df[key_cols + feature_cols],
                        on=key_cols,
                        how='outer'
                    )
            except Exception as e:
                logging.warning(f"Failed to read {csv_file}: {e}")
                continue
        
        if merged_df is not None:
            logging.info(f"Merged features for {target}: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        else:
            logging.warning(f"Failed to merge any features for {target}")
            merged_df = pd.DataFrame()
        
        return merged_df

    def train_target(self, target: str) -> Tuple[Any, Dict[str, Any]]:
        """Train a model for a specific target using merged feature groupings.
        
        This method follows the notebook workflow:
        1. Build training rows and save feature groupings
        2. Merge all feature grouping CSVs for the target
        3. Drop key columns and target, fillna(0.0)
        4. Train/test split
        5. Fit scaler on training data
        6. Train model on scaled features
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        game_data = self.game_data.copy()
        
        if game_data.empty:
            raise ValueError("game_data is empty; cannot train")
        
        if self.testing:
            game_data = game_data.sample(frac=0.1, random_state=42)
            logging.info(f"Testing mode: sampled {len(game_data)} rows for training")

        if not self.load_from_file:
            # Build training rows and save feature groupings to CSV
            all_rows: List[Dict[str, Any]] = self.build_training_rows(game_data, target)
            
            if not all_rows:
                raise ValueError(f"No training rows generated for {target}")
            
            # Save feature groups to CSV for merging
            feat_dir = os.path.join(self.features_dir, target)
            os.makedirs(feat_dir, exist_ok=True)
            
            for fe_key, feat_df in self.features_cache.items():
                if fe_key.startswith(f"{target}_"):
                    feat_path = os.path.join(feat_dir, f"{fe_key}.csv")
                    feat_df.to_csv(feat_path, index=False)
                    logging.debug(f"Saved feature group {fe_key} -> {feat_path}")
            
        # Merge all feature groupings from CSV (like the notebook)
        merged_df = self.merge_feature_groupings(target)
        
        if merged_df.empty:
            raise ValueError(f"Failed to merge feature groupings for {target}")
        
        # Define key columns to drop (like the notebook)
        key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr']
        
        # Drop key columns and target, then fillna(0.0) BEFORE scaling
        X = merged_df.drop(columns=key_cols + ['target']).fillna(0.0)
        y = merged_df['target'].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit scaler on training data only
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = self.model_factory() if callable(self.model_factory) else default_model_factory()
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics on test set
        y_pred = model.predict(X_test_scaled)
        metrics = {
            'mean_squared_error': float(mean_squared_error(y_test, y_pred)),
            'root_mean_squared_error': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mean_absolute_error': float(mean_absolute_error(y_test, y_pred)),
            'r2_score': float(r2_score(y_test, y_pred)),
            'num_samples': len(merged_df),
            'num_features': X.shape[1],
            'test_size': len(X_test),
        }
        
        # Extract feature importances if available
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            # For tree-based models (RandomForest, etc.)
            importances = model.feature_importances_
            feature_names = list(X.columns)
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            feature_importance = {
                'top_10_features': [
                    {'feature': name, 'importance': imp} 
                    for name, imp in sorted_features[:10]
                ],
                'bottom_10_features': [
                    {'feature': name, 'importance': imp} 
                    for name, imp in sorted_features[-10:]
                ]
            }
        elif hasattr(model, 'coef_'):
            # For linear models
            coefs = model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]
            feature_names = list(X.columns)
            coef_dict = {name: float(abs(coef)) for name, coef in zip(feature_names, coefs)}
            
            # Sort by absolute coefficient value
            sorted_features = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
            
            feature_importance = {
                'top_10_features': [
                    {'feature': name, 'weight': weight} 
                    for name, weight in sorted_features[:10]
                ],
                'bottom_10_features': [
                    {'feature': name, 'weight': weight} 
                    for name, weight in sorted_features[-10:]
                ]
            }
        else:
            # Fallback: Use correlation-based feature importance
            # This works for models without built-in feature importance (e.g., HistGradientBoostingRegressor)
            try:
                # Create a DataFrame with features and target
                X_with_target = X.copy()
                X_with_target['_target_'] = y
                
                # Calculate correlation with target
                correlations = X_with_target.corr()['_target_'].drop('_target_')
                
                # Use absolute correlation as importance measure
                correlation_dict = {name: float(abs(corr)) for name, corr in correlations.items()}
                
                # Sort by absolute correlation
                sorted_features = sorted(correlation_dict.items(), key=lambda x: x[1], reverse=True)
                
                feature_importance = {
                    'top_10_features': [
                        {'feature': name, 'correlation': corr} 
                        for name, corr in sorted_features[:10]
                    ],
                    'bottom_10_features': [
                        {'feature': name, 'correlation': corr} 
                        for name, corr in sorted_features[-10:]
                    ]
                }
                logging.info(f"Using correlation-based feature importance for {target}")
            except Exception as e:
                logging.warning(f"Could not calculate feature importance for {target}: {e}")

        # Combine metrics with feature importance
        metrics.update(feature_importance)
        
        # Save model, scaler, and feature names
        joblib.dump({
            'model': model, 
            'feature_names': list(X.columns),
            'scaler': scaler
        }, os.path.join(self.model_dir, f"{target}.pkl"))
        
        logging.info(f"Trained & saved model {target}; rows={len(merged_df)} features={X.shape[1]} R²={metrics['r2_score']:.4f} RMSE={metrics['root_mean_squared_error']:.4f}")

        return model, metrics

    def _train_one_safe(self, target: str):
        try:
            model, metrics = self.train_target(target)
            return (target, model, metrics, None)
        except Exception as e:
            logging.warning(f"Skipping {target}: {e}\n{traceback.format_exc()}")
            return (target, None, None, e)

    def train_many(self, targets: Sequence[str], save_metrics: bool = True) -> Dict[str, Any]:
        """Train multiple targets sequentially (synchronous to reuse DataObject).
        
        Args:
            targets: List of target names to train
            save_metrics: If True, save metrics to JSON file
            
        Returns:
            Dictionary mapping target names to trained models
        """
        if not targets:
            return {}

        results = []
        all_metrics = {}
        
        # Train each target sequentially to reuse the same DataObject
        for target in targets:
            logging.info(f"Training target: {target}")
            target_name, model, metrics, error = self._train_one_safe(target)
            results.append((target_name, model, metrics, error))
            
            if metrics is not None:
                all_metrics[target_name] = metrics

        out: Dict[str, Any] = {k: m for (k, m, metrics, e) in results if m is not None}
        
        # Save metrics if requested
        if save_metrics and all_metrics:
            metrics_path = os.path.join(self.model_dir, 'model_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            logging.info(f"Saved model metrics to {metrics_path}")
        
        return out

    def train_all_targets(self, save_metrics: bool = True) -> Dict[str, Any]:
        """Train all available targets (BASE_TARGETS + DERIVED_TARGETS).
        
        Args:
            save_metrics: If True, save metrics to model_metrics.json (default: True)
        
        Returns:
            Dictionary mapping target names to trained models
        """
        all_targets = self.targets['regression'] + self.targets['classification']
        
        logging.info(f"Training all targets: {all_targets}")
        return self.train_many(all_targets, save_metrics=save_metrics)

class Runner:
    def __init__(self):
        self.data_obj = DataObject(
            league='nfl',
            storage_mode='local',
            local_root='G:/My Drive/python/winsight_api/sports-data-storage-copy/'
        )
        self.local_predictions_dir = './game_predictions/'
        os.makedirs(self.local_predictions_dir, exist_ok=True)

    def train_all_targets(self):
        """Example 1: Train all targets"""
        trainer = GameModelTrainer(self.data_obj, testing=False, load_from_file=True)
        try:
            trained_models = trainer.train_all_targets()
            logging.info(f"Trained {len(trained_models)} models: {list(trained_models.keys())}")
        except Exception as e:
            logging.warning(f"Training failed: {e}")

    def predict_all_next_games(self):
        """Example 2: Load all models and predict all next games"""
        predictor = GamePredictor(data_obj=self.data_obj, models={})
        predictor.load_all_models()
        
        if predictor.models:
            predictions_df = predictor.predict_all_next_games()
            if not predictions_df.empty:
                week, year = (predictions_df['week'].iloc[0], predictions_df['year'].iloc[0])
                _dir = os.path.join(self.local_predictions_dir, f'{year}_week_{week}')
                os.makedirs(_dir, exist_ok=True)
                filename = f'next_game_predictions_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'

                # Write locally
                predictions_df.round(2).to_csv(os.path.join(_dir, filename), index=False)
                logging.info(f"Saved predictions to {os.path.join(_dir, filename)}")

                # Upload to S3
                upload_file_to_s3(
                    file_path=os.path.join(_dir, filename), 
                    bucket_name=os.getenv("SPORTS_DATA_BUCKET_NAME", ""),
                    s3_key=f'nfl/game_predictions/{year}_week_{week}/{filename}'
                )
        else:
            logging.warning("No models loaded, skipping predictions")

    def predict_single_game(self):
        """Example 3: Predict a single game by team abbreviation"""
        predictor = GamePredictor(data_obj=self.data_obj, models={})
        predictor.load_all_models()
        if predictor.models:
            team_abbr = 'GNB'  # Example: Green Bay Packers
            single_game_predictions = predictor.predict_single_game(team_abbr)
            if single_game_predictions:
                logging.info(f"Predictions for team {team_abbr}:\n{single_game_predictions}")
            else:
                logging.warning(f"No predictions generated for team {team_abbr}")
        else:
            logging.warning("No models loaded, skipping single game prediction")

    def write_merged_features(self):
        """Example 4: Write all feature groupings for a target to CSV"""
        trainer = GameModelTrainer(self.data_obj)
        df = trainer.merge_feature_groupings('home_points')
        if not df.empty:
            logging.info(f"Merged feature groupings for 'home_points': {len(df)} rows, {len(df.columns)} columns")
            df.to_csv('merged_home_points_features.csv', index=False)
            logging.info("Saved merged features to merged_home_points_features.csv")
        else:
            logging.warning("No features merged for 'home_points'")
        
        predictor = GamePredictor(data_obj=self.data_obj, models={})
        df_pred = predictor.merge_feature_groupings('home_points')
        if not df_pred.empty:
            logging.info(f"Merged feature groupings for 'home_points' (prediction): {len(df_pred)} rows, {len(df_pred.columns)} columns")
            df_pred.to_csv('merged_home_points_features_prediction.csv', index=False)
            logging.info("Saved merged features to merged_home_points_features_prediction.csv")
        else:
            logging.warning("No features merged for 'home_points' (prediction)")

    def create_all_past_predictions_from_merged(self):
        trainer = GameModelTrainer(self.data_obj)
        predictor = GamePredictor(data_obj=self.data_obj, models={})
        predictor.load_all_models()
        df = pd.DataFrame()
        for target in trainer.targets['regression'] + trainer.targets['classification']:
            merged = trainer.merge_feature_groupings(target)
            if not merged.empty:
                model = predictor.models.get(target)
                scaler = predictor.scalers.get(target)
                feature_names = predictor.model_feature_names.get(target)
                source = merged.copy()[['game_id', 'game_date', 'home_abbr', 'away_abbr']]
                merged = merged.drop(columns=['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target'], errors='ignore').fillna(0.0)
                if model and scaler and feature_names:
                    X = merged[feature_names].fillna(0.0)
                    X_scaled = scaler.transform(X)
                    preds = model.predict(X_scaled)
                    source[f'predicted_{target}'] = preds
                    if df.empty:
                        df = source
                    else:
                        df = df.merge(source, on=['game_id', 'game_date', 'home_abbr', 'away_abbr'], how='outer')
                    logging.info(f"Added predictions for target {target}")
                else:
                    logging.warning(f"Missing model/scaler/feature_names for target {target}, skipping predictions")

        if not df.empty:
            df.round(2).to_csv(os.path.join(self.local_predictions_dir, 'all_past_game_predictions.csv'), index=False)
            logging.info("Saved all past game predictions to all_past_game_predictions.csv")
            upload_file_to_s3(
                file_path=os.path.join(self.local_predictions_dir, 'all_past_game_predictions.csv'), 
                bucket_name=os.getenv("SPORTS_DATA_BUCKET_NAME", ""),
                s3_key='nfl/game_predictions/all_past_game_predictions.csv'
            )

    def update_all_past_predictions(self):
        """Example 5: Update all past game predictions from merged features"""
        boxscores = self.data_obj.boxscores
        if boxscores.empty:
            logging.warning("No boxscore data available to update past predictions")
            return
        # Find missing game_ids in existing all_past_game_predictions.csv
        existing_predictions_path = os.path.join(self.local_predictions_dir, 'all_past_game_predictions.csv')
        if os.path.exists(existing_predictions_path):
            existing_df = pd.read_csv(existing_predictions_path)
            existing_keys = existing_df['game_id'].unique()
            boxscore_keys = boxscores['key'].unique()
            missing_keys = set(boxscore_keys) - set(existing_keys)
            if missing_keys:
                logging.info(f"Found {len(missing_keys)} missing game predictions to update")
                # Create predictions for missing games only
                predictor = GamePredictor(data_obj=self.data_obj, models={})
                predictor.load_all_models()
                game_data = self.data_obj.get_game_data_with_features()
                trainer = GameModelTrainer(self.data_obj)
                new_df = pd.DataFrame()
                for target in trainer.targets['regression'] + trainer.targets['classification']:
                    rows = []
                    for key in missing_keys:
                        feat, _ = trainer._build_single_training_row(game_data, game_data[game_data['key'] == key].index[0], target)
                        feat.update({'game_id': key})
                        rows.append(feat)
                    df = pd.DataFrame(rows).drop(columns=[f'target_{target}'], errors='ignore').fillna(0.0)
                    model = predictor.models.get(target)
                    scaler = predictor.scalers.get(target)
                    feature_names = predictor.model_feature_names.get(target)
                    if model and scaler and feature_names:
                        X = df[feature_names].fillna(0.0)
                        X_scaled = scaler.transform(X)
                        preds = model.predict(X_scaled)
                        df[f'predicted_{target}'] = preds
                        if new_df.empty:
                            new_df = df[['game_id', f'predicted_{target}']]
                        else:
                            new_df = new_df.merge(
                                df[['game_id', f'predicted_{target}']], 
                                on='game_id', 
                                how='outer'
                            )
                        logging.info(f"Updated predictions for target {target}")
                new_df = new_df.merge(
                    boxscores[['key', 'game_date', 'home_abbr', 'away_abbr']].rename(columns={'key': 'game_id'}),
                    on='game_id',
                    how='left'
                )
                # Combine with existing predictions
                if not new_df.empty:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.round(2).to_csv(existing_predictions_path, index=False)
                    logging.info("Updated all past game predictions with new data")
                    upload_file_to_s3(
                        file_path=existing_predictions_path, 
                        bucket_name=os.getenv("SPORTS_DATA_BUCKET_NAME", ""),
                        s3_key='nfl/game_predictions/all_past_game_predictions.csv'
                    )
            else:
                logging.info("No missing game predictions found; all up to date")

if __name__ == "__main__":
    Runner().update_all_past_predictions()