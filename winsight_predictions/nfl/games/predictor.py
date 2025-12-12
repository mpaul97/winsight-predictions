from __future__ import annotations

from dataclasses import dataclass, field
import traceback
from typing import Dict, Any, List, Optional, Sequence, Callable, Tuple
import os
import logging
import joblib
import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO

load_dotenv()

try:
    from .features import FeatureEngine
    from .trainer import GameModelTrainer
    from ..data_object import DataObject
    from ..helpers import upload_file_to_s3
except ImportError:
    from features import FeatureEngine
    from trainer import GameModelTrainer
    # Get the absolute path of the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(current_dir))
    from data_object import DataObject
    from helpers import upload_file_to_s3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# GamePredictor (Inference)
# ---------------------------------------------------------------------------
@dataclass
class GamePredictor:
    data_obj: DataObject
    models: Dict[str, Any] = field(default_factory=dict) # key: target -> fitted model
    min_games: int = 3
    root_dir: str = "./"
    model_dir: str = field(default="")  # Will be set in __post_init__
    features_dir: str = field(default="")  # Will be set in __post_init__
    use_saved_scalers: bool = False
    scalers: Dict[str, Any] = field(default_factory=dict)  # optional per-target scaler
    model_feature_names: Dict[str, List[str]] = field(default_factory=dict)  # stored feature ordering per model
    use_residual_training: Dict[str, bool] = field(default_factory=dict)  # whether model uses residual training
    global_means: Dict[str, float] = field(default_factory=dict)  # global mean for residual models
    team_baselines: Dict[str, Dict] = field(default_factory=dict)  # team baselines for residual models
    predictions_bucket_name: str = 'LEAGUE_PREDICTIONS_BUCKET_NAME'

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

    classification_targets: List[str] = field(default_factory=lambda: [
        'home_win'
    ])

    def __post_init__(self):
        # Set directory paths based on root_dir
        if not self.model_dir:
            self.model_dir = os.path.join(self.root_dir, "models")
        if not self.features_dir:
            self.features_dir = os.path.join(self.root_dir, "predicted_features")
        
        if self.data_obj.storage_mode == 'local':
            os.makedirs(self.model_dir, exist_ok=True)
            
        os.makedirs(self.features_dir, exist_ok=True)

        self.load_all_models()
        
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

    def load_all_models(self) -> None:
        """Load all available model bundles from model_dir or S3."""
        count = 0
        
        if self.data_obj.storage_mode == "s3":
            # Load models from S3
            bucket_name = os.getenv(self.predictions_bucket_name)
            if not bucket_name:
                logging.error(f"{self.predictions_bucket_name} environment variable not set for S3 mode.")
                return
            
            s3_client = boto3.client('s3')
            s3_prefix = 'nfl/games/'
            
            try:
                # List all objects under the nfl/games/ prefix
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
                
                for page in pages:
                    if 'Contents' not in page:
                        continue
                    
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('_model.pkl'):
                            # Extract target from S3 key
                            # Expected format: nfl/games/TARGET_model.pkl
                            filename = key.replace(s3_prefix, '')
                            target = filename.replace('_model.pkl', '')
                            
                            model_bundle = self._load_model_from_s3(bucket_name, key, s3_client)
                            if model_bundle:
                                if isinstance(model_bundle, dict) and 'model' in model_bundle:
                                    self.models[target] = model_bundle.get('model')
                                    self.scalers[target] = model_bundle.get('scaler')
                                    self.model_feature_names[target] = model_bundle.get('feature_names', [])
                                    # Extract residual training metadata
                                    self.use_residual_training[target] = model_bundle.get('use_residual_training', False)
                                    if self.use_residual_training[target]:
                                        self.global_means[target] = model_bundle.get('global_mean', 0.0)
                                        self.team_baselines[target] = model_bundle.get('team_baselines', {})
                                    count += 1
                                else:
                                    # Backward compatibility: raw model without metadata
                                    self.models[target] = model_bundle
                                    self.use_residual_training[target] = False
                                    count += 1
            except Exception as e:
                logging.error(f"Failed listing S3 models: {e}")
        
        else:
            # Load models from local filesystem
            if not os.path.exists(self.model_dir):
                logging.warning(f"Model directory {self.model_dir} does not exist")
                return
            
            for filename in os.listdir(self.model_dir):
                if filename.endswith('_model.pkl'):
                    target = filename.replace('_model.pkl', '')
                    model_path = os.path.join(self.model_dir, filename)
                    
                    try:
                        model_bundle = joblib.load(model_path)
                        if isinstance(model_bundle, dict) and 'model' in model_bundle:
                            self.models[target] = model_bundle.get('model')
                            self.scalers[target] = model_bundle.get('scaler')
                            self.model_feature_names[target] = model_bundle.get('feature_names', [])
                            # Extract residual training metadata
                            self.use_residual_training[target] = model_bundle.get('use_residual_training', False)
                            if self.use_residual_training[target]:
                                self.global_means[target] = model_bundle.get('global_mean', 0.0)
                                self.team_baselines[target] = model_bundle.get('team_baselines', {})
                            count += 1
                        else:
                            # Backward compatibility: raw model without metadata
                            self.models[target] = model_bundle
                            self.use_residual_training[target] = False
                            count += 1
                    except Exception as e:
                        logging.warning(f"Failed loading {model_path}: {e}")
        
        logging.info(f"Loaded {count} model bundles from {self.data_obj.storage_mode} storage")

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
                group_name = csv_file.replace('.csv', '').replace(f"{target}_", '')

                df = df.rename(columns={col: f"{col}_{group_name}" for col in df.columns if col not in key_cols})

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
                    logging.info(f"Adding {len(missing_cols)} missing columns for {target}")
                    # Create missing columns DataFrame and concat instead of iterative insertion
                    missing_df = pd.DataFrame(0.0, index=X_pred.index, columns=missing_cols)
                    X_pred = pd.concat([X_pred, missing_df], axis=1)
                
                if extra_cols:
                    logging.info(f"Dropping {len(extra_cols)} extra columns for {target}")
                
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
            if target in self.classification_targets:
                # For classification, use predict_proba to get probability of positive class
                pred_val = float(model.predict_proba(X_pred_scaled)[0, 1])
            else:
                pred_val = float(model.predict(X_pred_scaled)[0])
            
            # If model uses residual training, add back the baseline (only for regression targets)
            if self.use_residual_training.get(target, False) and target not in self.classification_targets:
                baselines = self.team_baselines.get(target, {})
                global_mean = self.global_means.get(target, 0.0)
                
                # Determine which team we're predicting for (home or away)
                if target.startswith('home_'):
                    team_abbr = upcoming_row.get('home_abbr')
                elif target.startswith('away_'):
                    team_abbr = upcoming_row.get('away_abbr')
                else:
                    team_abbr = upcoming_row.get('home_abbr')  # default to home
                
                # Check if baselines are time-varying or static
                if baselines:
                    sample_key = next(iter(baselines.keys()))
                    if isinstance(sample_key, tuple):
                        # Time-varying baselines (team_abbr, game_date): find most recent baseline for team
                        # Since upcoming game_date is new, get the last available baseline
                        team_baselines_filtered = {
                            (team, date): val 
                            for (team, date), val in baselines.items() 
                            if team == team_abbr
                        }
                        
                        if team_baselines_filtered:
                            # Get the most recent baseline (max date)
                            most_recent_key = max(team_baselines_filtered.keys(), key=lambda x: x[1])
                            baseline = team_baselines_filtered[most_recent_key]
                        else:
                            # Team not in training data, use global mean
                            baseline = global_mean
                    else:
                        # Static baselines (team_abbr only)
                        baseline = baselines.get(team_abbr, global_mean)
                else:
                    baseline = global_mean
                
                # Add baseline to residual prediction
                pred_val = pred_val + baseline
                logging.debug(f"Denormalized residual prediction for {team_abbr} {target}: "
                             f"residual={pred_val - baseline:.2f}, baseline={baseline:.2f}, "
                             f"final={pred_val:.2f}")
            
            predictions[target] = float(round(pred_val, 2))
            logging.debug(f"Predicted {target} = {pred_val:.2f}")

        logging.info(f"Predictions completed for {upcoming_row.get('game_id', upcoming_row.get('key', ''))}")

        return predictions

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

    def predict_all_next_games(self, save_to_file: bool = False, upload_to_s3: bool = False) -> pd.DataFrame:
        """Predict all upcoming games from data_obj.previews using all loaded models.
        
        Args:
            save_to_file: If True, save predictions to CSV file
            upload_to_s3: If True, upload predictions to S3
        
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
        
        # Save to file if requested
        if save_to_file and not predictions_df.empty:
            local_predictions_dir = os.path.join(self.root_dir, 'game_predictions')
            os.makedirs(local_predictions_dir, exist_ok=True)
            
            week, year = (predictions_df['week'].iloc[0], predictions_df['year'].iloc[0])
            _dir = os.path.join(local_predictions_dir, f'{year}_week_{week}')
            os.makedirs(_dir, exist_ok=True)
            filename = f'next_game_predictions_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
            file_path = os.path.join(_dir, filename)
            
            predictions_df.round(2).to_csv(file_path, index=False)
            logging.info(f"Saved predictions to {file_path}")
            
            # Upload to S3 if requested
            if upload_to_s3:
                upload_file_to_s3(
                    file_path=file_path,
                    bucket_name=os.getenv("SPORTS_DATA_BUCKET_NAME", ""),
                    s3_key=f'nfl/game_predictions/{year}_week_{week}/{filename}'
                )
        
        return predictions_df

    def create_all_past_predictions_from_merged(self, save_to_file: bool = False, upload_to_s3: bool = False) -> pd.DataFrame:
        """Create predictions for all historical games using merged feature groupings.
        
        Args:
            save_to_file: If True, save predictions to CSV file
            upload_to_s3: If True, upload predictions to S3
            
        Returns:
            DataFrame with predictions for all historical games
        """ 
        trainer = GameModelTrainer(self.data_obj)
        df = pd.DataFrame()
        
        all_targets = trainer.targets['regression'] + trainer.targets['classification']
        
        for target in all_targets:
            merged = trainer._load_and_merge_features(target)
            if merged is None or merged.empty:
                logging.warning(f"No merged features found for {target}")
                continue
            
            model = self.models.get(target)
            scaler = self.scalers.get(target)
            feature_names = self.model_feature_names.get(target)
            
            if not (model and scaler and feature_names):
                logging.warning(f"Missing model/scaler/feature_names for target {target}, skipping predictions")
                continue
            
            source = merged.copy()[['game_id', 'game_date', 'home_abbr', 'away_abbr']]
            key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr']
            X = merged.drop(columns=key_cols + ['target'], errors='ignore')
            
            # Align to model's feature names
            missing_cols = [col for col in feature_names if col not in X.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=X.index, columns=missing_cols)
                X = pd.concat([X, missing_df], axis=1)
            X = X[feature_names].fillna(0.0)
            
            X_scaled = scaler.transform(X)
            
            # Use predict_proba for classification models
            if target in self.classification_targets:
                preds = model.predict_proba(X_scaled)[:, 1]
            else:
                preds = model.predict(X_scaled)
            
            # If model uses residual training, add back the baseline (only for regression targets)
            if self.use_residual_training.get(target, False) and target not in self.classification_targets:
                baselines = self.team_baselines.get(target, {})
                global_mean = self.global_means.get(target, 0.0)
                
                # Determine which team column to use based on target name
                is_home_target = target.startswith('home_')
                team_col = 'home_abbr' if is_home_target else 'away_abbr'
                
                if baselines:
                    sample_key = next(iter(baselines.keys()))
                    if isinstance(sample_key, tuple):
                        # Time-varying baselines (team_abbr, game_date)
                        baseline_values = source.apply(
                            lambda row: baselines.get((row[team_col], row['game_date']), global_mean),
                            axis=1
                        ).values
                    else:
                        # Static baselines (team_abbr only)
                        baseline_values = source[team_col].map(baselines).fillna(global_mean).values
                    
                    preds = preds + baseline_values
                    logging.info(f"Added baselines to residual predictions for {target}")
            
            source[f'predicted_{target}'] = preds
            
            if df.empty:
                df = source
            else:
                df = df.merge(source, on=['game_id', 'game_date', 'home_abbr', 'away_abbr'], how='outer')
            
            logging.info(f"Added predictions for target {target}")
        
        if save_to_file and not df.empty:
            local_predictions_dir = os.path.join(self.root_dir, 'game_predictions')
            os.makedirs(local_predictions_dir, exist_ok=True)
            file_path = os.path.join(local_predictions_dir, 'all_past_game_predictions.csv')
            
            df.round(2).to_csv(file_path, index=False)
            logging.info(f"Saved all past game predictions to {file_path}")
            
            if upload_to_s3:
                upload_file_to_s3(
                    file_path=file_path,
                    bucket_name=os.getenv("SPORTS_DATA_BUCKET_NAME", ""),
                    s3_key='nfl/game_predictions/all_past_game_predictions.csv'
                )
        
        return df

    def update_all_past_predictions(self, save_to_file: bool = False, upload_to_s3: bool = False) -> pd.DataFrame:
        """Update all past game predictions by adding missing game predictions.
        
        This method checks for games that exist in boxscores but are missing from the 
        existing predictions file, and only creates predictions for those missing games.
        
        Args:
            save_to_file: If True, save updated predictions to CSV file
            upload_to_s3: If True, upload updated predictions to S3
            
        Returns:
            DataFrame with all predictions including newly added ones
        """
        boxscores = self.data_obj.boxscores
        if boxscores.empty:
            logging.warning("No boxscore data available to update past predictions")
            return pd.DataFrame()
        
        local_predictions_dir = './game_predictions/'
        os.makedirs(local_predictions_dir, exist_ok=True)
        existing_predictions_path = os.path.join(local_predictions_dir, 'all_past_game_predictions.csv')
        
        # Check for missing game_ids
        if not os.path.exists(existing_predictions_path):
            logging.info("No existing predictions file found. Creating all predictions from scratch.")
            return self.create_all_past_predictions_from_merged(save_to_file=save_to_file, upload_to_s3=upload_to_s3)
        
        existing_df = pd.read_csv(existing_predictions_path)
        existing_keys = set(existing_df['game_id'].unique())
        boxscore_keys = set(boxscores['key'].unique())
        missing_keys = boxscore_keys - existing_keys
        
        if not missing_keys:
            logging.info("No missing game predictions found; all up to date")
            return existing_df
        
        logging.info(f"Found {len(missing_keys)} missing game predictions to update \n {missing_keys}")
        
        # Create predictions for missing games only using merged features
        trainer = GameModelTrainer(self.data_obj)
        new_df = pd.DataFrame()
        all_targets = trainer.targets['regression'] + trainer.targets['classification']
        
        for target in all_targets:
            # Load merged features for this target
            merged = trainer._load_and_merge_features(target)
            if merged is None or merged.empty:
                logging.warning(f"No merged features found for {target}")
                continue
            
            # Filter to only missing game_ids
            merged_missing = merged[merged['game_id'].isin(missing_keys)].copy()
            if merged_missing.empty:
                logging.warning(f"No features found for missing games for target {target}")
                continue
            
            model = self.models.get(target)
            scaler = self.scalers.get(target)
            feature_names = self.model_feature_names.get(target)
            
            if not (model and scaler and feature_names):
                logging.warning(f"Missing model/scaler/feature_names for target {target}, skipping")
                continue
            
            # Prepare features
            source = merged_missing[['game_id', 'game_date', 'home_abbr', 'away_abbr']].copy()
            key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr']
            X = merged_missing.drop(columns=key_cols + ['target'], errors='ignore')
            
            # Align to model's feature names
            missing_cols = [col for col in feature_names if col not in X.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=X.index, columns=missing_cols)
                X = pd.concat([X, missing_df], axis=1)
            X = X[feature_names].fillna(0.0)
            
            # Make predictions
            X_scaled = scaler.transform(X)
            
            # Use predict_proba for classification models
            if target in self.classification_targets:
                preds = model.predict_proba(X_scaled)[:, 1]
            else:
                preds = model.predict(X_scaled)
            
            # If model uses residual training, add back the baseline (only for regression targets)
            if self.use_residual_training.get(target, False) and target not in self.classification_targets:
                baselines = self.team_baselines.get(target, {})
                global_mean = self.global_means.get(target, 0.0)
                
                # Determine which team column to use based on target name
                is_home_target = target.startswith('home_')
                team_col = 'home_abbr' if is_home_target else 'away_abbr'
                
                if baselines:
                    sample_key = next(iter(baselines.keys()))
                    if isinstance(sample_key, tuple):
                        # Time-varying baselines (team_abbr, game_date)
                        baseline_values = source.apply(
                            lambda row: baselines.get((row[team_col], row['game_date']), global_mean),
                            axis=1
                        ).values
                    else:
                        # Static baselines (team_abbr only)
                        baseline_values = source[team_col].map(baselines).fillna(global_mean).values
                    
                    preds = preds + baseline_values
                    logging.info(f"Added baselines to residual predictions for {target}")
            
            # Build result DataFrame
            source[f'predicted_{target}'] = preds
            
            if new_df.empty:
                new_df = source
            else:
                new_df = new_df.merge(source, on=['game_id', 'game_date', 'home_abbr', 'away_abbr'], how='outer')
            
            logging.info(f"Updated predictions for target {target}")
        
        # Combine with existing predictions
        if not new_df.empty:
            # Combine with existing predictions
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            if save_to_file:
                combined_df.round(2).to_csv(existing_predictions_path, index=False)
                logging.info(f"Updated all past game predictions to {existing_predictions_path}")
                
                if upload_to_s3:
                    upload_file_to_s3(
                        file_path=existing_predictions_path,
                        bucket_name=os.getenv("SPORTS_DATA_BUCKET_NAME", ""),
                        s3_key='nfl/game_predictions/all_past_game_predictions.csv'
                    )
            
            return combined_df
        else:
            logging.warning("No new predictions were generated")
            return existing_df

if __name__ == "__main__":
    # Initialize data object
    # data_obj = DataObject(
    #     league='nfl',
    #     storage_mode='local',
    #     local_root=os.path.join(sys.path[0], "..", "..", "..", "..", "sports-data-storage-copy/")
    # )

    data_obj = DataObject(
        storage_mode='s3',
        s3_bucket=os.getenv('SPORTS_DATA_BUCKET_NAME')
    )
    
    # Create predictor and load all models
    predictor = GamePredictor(
        data_obj=data_obj,
        root_dir='./',
        predictions_bucket_name='LEAGUE_PREDICTIONS_BUCKET_NAME'
    )
    
    # Example: Predict all next games and save to file
    predictions_df = predictor.predict_all_next_games(save_to_file=True, upload_to_s3=True)
    
    # # # Example: Predict a single game
    # single_game_predictions = predictor.predict_single_game('GNB')
    # if single_game_predictions:
    #     logging.info(f"Predictions: {single_game_predictions}")
    
    # Example: Create all past predictions from merged features
    # predictor.create_all_past_predictions_from_merged(save_to_file=True, upload_to_s3=True)

    # Example: Update all past predictions
    # predictor.update_all_past_predictions(save_to_file=True, upload_to_s3=True)