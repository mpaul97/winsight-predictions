from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Any, List, Optional, Sequence, Callable, Tuple
import os
import logging
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import traceback
import asyncio
from asyncio import Semaphore
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import boto3
from dotenv import load_dotenv
    
load_dotenv()

try:
    from .features import FeatureEngine
    from ..data_object import DataObject
except ImportError:
    from features import FeatureEngine
    # Get the absolute path of the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(current_dir))
    from data_object import DataObject

s3 = boto3.client('s3')

logging.basicConfig(level=logging.INFO)

class GameModelTrainer:

    def __init__(self, data_obj: DataObject, use_residual_training: bool = False):
        self.data_obj = data_obj
        self.min_games = 3
        self.use_residual_training = use_residual_training
        
        # Residual training attributes
        self.global_mean = None
        self.team_baselines = None

        self.features_dir = "./features/"
        self.models_dir = "./models/"
        if __name__ == "__main__":
            os.makedirs(self.features_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)

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

        self.targets: Dict[str, List[str]] = {
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
        }

        self.classification_targets = self.targets['classification']

        return
    
    def _build_feature_engine(
        self,
        game_data: pd.DataFrame,
        target: str,
        row: pd.Series
    ) -> FeatureEngine:
        """Build FeatureEngine with optional cached shared features.
        
        Args:
            game_data: Historical game data
            target: Target variable name
            row: Current game row
            
        Returns:
            FeatureEngine instance with features computed
        """
        return FeatureEngine(
            game_data=game_data,
            target_name=target,
            row=row,
            predicted_features=None,
            **self._fe_common_params
        )

    def _process_game_features(self, game_data: pd.DataFrame, targets: List[str]) -> Dict[str, pd.DataFrame]:
        """Process features for all games for all targets.
        
        Args:
            game_data: Sorted game data
            targets: List of target variable names
            
        Returns:
            Dictionary of feature groupings as DataFrames
        """
        try:
            game_frames = {}
            game_data = game_data.sort_values('game_date').reset_index(drop=True)

            for target in targets:
                # Check if target exists in the data
                if target not in game_data.columns:
                    logging.warning(f"Target '{target}' not found in columns. Skipping.")
                    continue
                
                # Use tqdm to track progress
                valid_indices = [idx for idx in range(len(game_data)) if idx >= self.min_games]
                
                for idx in tqdm(valid_indices, desc=f"Processing {target}", unit="game"):
                    row = game_data.iloc[idx]
                    prior_games = game_data.iloc[:idx]
                    try:
                        fe = self._build_feature_engine(
                            game_data=prior_games,
                            target=target,
                            row=row
                        )
                        frames = fe.grouped_features_as_dfs
                        for group_name, df in frames.items():
                            df['game_id'] = row.get('game_id', row.get('key', np.nan))
                            df['game_date'] = row['game_date']
                            df['home_abbr'] = row.get('home_abbr', '')
                            df['away_abbr'] = row.get('away_abbr', '')
                            df['target'] = row[target]
                            df = df[['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target'] + 
                                   [col for col in df.columns if col not in ['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target']]]
                            fe_key = f"{target}-{group_name}"
                            if fe_key not in game_frames:
                                game_frames[fe_key] = df
                            else:
                                game_frames[fe_key] = pd.concat([game_frames[fe_key], df], ignore_index=True)
                    except Exception as e:
                        logging.error(f"Error processing game at idx {idx}, target {target}: {e}")
                        raise  # Re-raise to see the full traceback
            return game_frames
        except Exception as e:
            logging.error(f"Fatal error in _process_game_features: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _create_feature_groupings_async(self, executor: ProcessPoolExecutor, targets: List[str]):
        """Create feature groupings for all targets using parallel processing.
        
        Args:
            executor: ProcessPoolExecutor for parallel processing
            targets: List of target variable names
        """
        game_data = self.game_data.copy()
        
        if game_data.empty:
            logging.warning(f"No game data found")
            return

        # Filter to games with actual results
        game_data = game_data[game_data['home_points'].notna() & game_data['away_points'].notna()]
        
        logging.info(f"Processing {len(game_data)} games for {len(targets)} targets")
        
        try:
            loop = asyncio.get_running_loop()
            
            logging.info(f"Processing features for all targets")
            
            # Run the synchronous, CPU-bound function in a separate process
            all_frames = await loop.run_in_executor(
                executor,
                self._process_game_features,
                game_data,
                targets
            )
            
            # Save feature groups to CSV
            for fe_key, df in all_frames.items():
                logging.info(f"Feature Group: {fe_key}, Shape: {df.shape}")
                target, group_name = fe_key.split("-", 1)  # Split target and group_name
                _dir = os.path.join(self.features_dir, target)
                os.makedirs(_dir, exist_ok=True)
                df.to_csv(f"{_dir}/{group_name}_features.csv", index=False)
            
            all_frames.clear()
            
        except Exception as e:
            logging.error(f"Failed to process targets: {e}")
            import traceback
            traceback.print_exc()

    async def _run_all_feature_groupings(self, targets: List[str]):
        """Run feature grouping creation for all targets.
        
        Args:
            targets: List of target names to process
        """
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Process all targets in a single async task
            await self._create_feature_groupings_async(executor, targets)

    def create_feature_groupings(self, target: Optional[str] = None, targets: Optional[List[str]] = None):
        """Create feature groupings for specified targets.
        
        Args:
            target: Single target to process (optional)
            targets: List of targets to process (optional)
        """
        if target:
            target_list = [target]
        elif targets:
            target_list = targets
        else:
            target_list = self.targets['regression'] + self.targets['classification']

        try:
            asyncio.run(self._run_all_feature_groupings(target_list))
        except Exception as e:
            logging.error(f"An error occurred during feature creation: {e}")
            traceback.print_exc()

        return

    def debug_feature_grouping(self, game_id: Optional[str] = None, target: str = 'home_points'):
        """
        Builds and inspects feature groupings for a single game for debugging.

        Args:
            game_id: The game's ID (e.g., '202411090gnb'). If None, uses last game.
            target: The target variable to generate features for (e.g., 'home_points').
        
        Returns:
            The populated FeatureEngine instance for inspection.
        """
        game_data = self.data_obj.get_game_data_with_features()
        game_data = game_data.sort_values('game_date').reset_index(drop=True)

        if len(game_data) < self.min_games + 1:
            logging.error(f"Not enough games to debug. Found {len(game_data)}, need at least {self.min_games + 1}.")
            return None

        # Use the specified game or the last game as the target row
        if game_id:
            game_data_filtered = game_data[game_data.get('game_id', game_data.get('key', '')) == game_id]
            if game_data_filtered.empty:
                logging.error(f"Game {game_id} not found in game data.")
                return None
            row_idx = game_data_filtered.index[0]
            row = game_data.loc[row_idx]
            prior_games = game_data.iloc[:row_idx]
        else:
            row = game_data.iloc[-1]
            prior_games = game_data.iloc[:-1]
        
        game_id_display = row.get('game_id', row.get('key', 'unknown'))
        logging.info(f"Debugging FeatureEngine for game {game_id_display} targeting '{target}' for game on {row['game_date']}")
        logging.info(f"Using {len(prior_games)} prior games.")

        fe = self._build_feature_engine(
            game_data=prior_games,
            target=target,
            row=row
        )

        # frames = fe.grouped_features_as_dfs
        # for group_name, df in frames.items():
        #     print("-" * 50)
        #     print(f"Feature Group: '{group_name}'")
        #     print(f"Shape: {df.shape}")
        #     print("Head:")
        #     print(df.head())
        #     print("-" * 50)
            
        return fe

    def _load_and_merge_features(self, target: str) -> Optional[pd.DataFrame]:
        """Load and merge all feature groups for a specific target.
        
        Args:
            target: Target variable
            
        Returns:
            Merged DataFrame or None if loading fails
        """
        
        feature_group_dir = os.path.join(self.features_dir, target)
        if not os.path.exists(feature_group_dir):
            logging.debug(f"Feature directory does not exist: {feature_group_dir}. Skipping.")
            return None
        
        feature_files = [f for f in os.listdir(feature_group_dir) if f.endswith('_features.csv')]
        if not feature_files:
            logging.warning(f"No feature files found for Target: {target}. Skipping.")
            return None
        
        logging.info(f"Loading {len(feature_files)} feature groups for Target: {target}")
        
        df = None
        
        for i, feature_file in enumerate(feature_files):
            group_name = feature_file.replace('_features.csv', '')
            feature_path = os.path.join(feature_group_dir, feature_file)
            logging.info(f"  Loading feature group {i+1}/{len(feature_files)}: {group_name}")
            
            try:
                # Load with optimized dtypes
                features_df = pd.read_csv(feature_path, low_memory=False)
                
                # CRITICAL: Normalize merge keys to prevent Cartesian products
                # Convert game_date to string format for consistent merging
                if 'game_date' in features_df.columns:
                    features_df['game_date'] = pd.to_datetime(features_df['game_date'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Ensure game_id is string
                if 'game_id' in features_df.columns:
                    features_df['game_id'] = features_df['game_id'].astype(str)
                
                # Check for duplicates in merge keys
                dup_count = features_df.duplicated(subset=['game_id', 'game_date', 'target']).sum()
                if dup_count > 0:
                    logging.warning(f"  WARNING: Found {dup_count} duplicate rows in {group_name}, removing duplicates")
                    features_df = features_df.drop_duplicates(subset=['game_id', 'game_date', 'target'], keep='first')
                
                logging.info(f"    Feature group {group_name} has {len(features_df)} rows")
                
                # Optimize data types to reduce memory
                for col in features_df.columns:
                    if col in ['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target']:
                        continue
                    # Convert to float32 instead of float64 to save memory
                    if features_df[col].dtype in ['float64', 'int64']:
                        try:
                            features_df[col] = features_df[col].astype('float32')
                        except:
                            pass
                
                # Rename columns
                features_df.columns = [f"{col}_{group_name}" if col not in ['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target'] else col for col in features_df.columns]
                
                # Merge incrementally
                if df is None:
                    df = features_df
                    logging.info(f"    Initial dataframe: {df.shape}")
                else:
                    # Normalize merge keys in existing df too
                    if 'game_date' in df.columns:
                        df['game_date'] = pd.to_datetime(df['game_date'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
                    if 'game_id' in df.columns:
                        df['game_id'] = df['game_id'].astype(str)
                    
                    # Validate merge will work correctly
                    before_rows = len(df)
                    df = pd.merge(df, features_df, on=['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target'], how='inner', validate='1:1')
                    after_rows = len(df)
                    
                    if after_rows > before_rows * 1.1:  # More than 10% increase is suspicious
                        logging.error(f"    ERROR: Merge created unexpected rows! Before: {before_rows}, After: {after_rows}")
                        logging.error(f"    This indicates duplicate keys. Aborting this feature group.")
                        # Rollback by skipping this merge
                        continue
                    
                    logging.info(f"    Shape after merge: {df.shape}")
                
                # Clean up
                del features_df
                
            except pd.errors.MergeError as e:
                logging.error(f"Merge validation failed for {group_name}: {e}")
                logging.error(f"This usually means duplicate keys exist. Skipping this feature group.")
                continue
            except MemoryError as e:
                logging.error(f"Memory error loading {group_name}: {e}")
                logging.info("Attempting to continue with already loaded features...")
                break
            except Exception as e:
                logging.error(f"Error loading {group_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if df is None or df.empty:
            logging.warning(f"No features loaded for Target: {target}.")
            return None
        
        return df

    def _select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 500) -> pd.DataFrame:
        """Perform feature selection to reduce dimensionality.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            max_features: Maximum number of features to retain
            
        Returns:
            Filtered DataFrame with selected features
        """
        from sklearn.feature_selection import VarianceThreshold
        
        logging.info("Performing feature selection...")
        
        # 1. Remove columns with too many missing values (>50%)
        missing_pct = X.isnull().sum() / len(X)
        cols_to_keep = missing_pct[missing_pct < 0.5].index.tolist()
        X = X[cols_to_keep]
        logging.info(f"  After removing high-missing columns: {X.shape[1]} features")
        
        # 2. Fill remaining NaN with 0
        X = X.fillna(0.0)
        
        # 3. Remove zero-variance features
        selector = VarianceThreshold(threshold=0.0)
        X_var = selector.fit_transform(X)
        X = pd.DataFrame(X_var, columns=X.columns[selector.get_support()], index=X.index)
        logging.info(f"  After removing zero-variance: {X.shape[1]} features")
        
        # 4. If still too many features, select top features by correlation with target
        if X.shape[1] > max_features:
            logging.info(f"  Reducing from {X.shape[1]} to {max_features} features using correlation selection")
            correlations = X.corrwith(y).abs()
            top_features = correlations.nlargest(max_features).index.tolist()
            X = X[top_features]
            logging.info(f"  After correlation selection: {X.shape[1]} features")
        
        return X

    def _compute_team_baselines(self, df: pd.DataFrame, target_col: str, prior_weight: int = 5, window: int = 10, use_rolling: bool = True, normal_window_mean: bool = False) -> None:
        """Compute team-level shrinkage baselines using rolling window or total mean.
        
        Args:
            df: DataFrame containing game_id, game_date, home_abbr, away_abbr, and target column
            target_col: Name of the target column
            prior_weight: Weight for global mean in shrinkage estimation
            window: Number of recent games to use for baseline (default: 10, only used if use_rolling=True)
            use_rolling: If True, uses rolling window mean. If False, uses total career mean per team
            normal_window_mean: If True, uses normal window mean calculation
        """
        self.global_mean = df[target_col].mean()
        
        # Determine which team to track based on target name
        # For 'home_points', 'home_total_yards', etc. -> track home team
        # For 'away_points', 'away_total_yards', etc. -> track away team
        is_home_target = target_col.startswith('home_')
        team_col = 'home_abbr' if is_home_target else 'away_abbr'
        
        # Create a unified team view where each row represents a team's performance
        team_games = []
        
        for idx, row in df.iterrows():
            team_games.append({
                'team_abbr': row[team_col],
                'game_date': row['game_date'],
                'game_id': row['game_id'],
                'value': row[target_col]
            })
        
        team_df = pd.DataFrame(team_games).sort_values(['team_abbr', 'game_date']).copy()
        
        if use_rolling:
            # Compute rolling mean for each team
            team_df['rolling_mean'] = (
                team_df.fillna(0.0).groupby('team_abbr')['value']
                .transform(lambda x: x.rolling(window=window, min_periods=2).mean())
            )
            team_df['rolling_count'] = (
                team_df.fillna(0.0).groupby('team_abbr')['value']
                .transform(lambda x: x.rolling(window=window, min_periods=2).count())
            )
            
            # Apply shrinkage: blend rolling mean with global mean
            team_df['baseline'] = (
                team_df['rolling_mean'] * team_df['rolling_count'] + 
                self.global_mean * prior_weight
            ) / (team_df['rolling_count'] + prior_weight)
            
            # Fill NaN with global mean
            team_df['baseline'] = team_df['baseline'].fillna(self.global_mean)
            
            # KEY FIX: Store baseline keyed by (team_abbr, game_date) for team-level time-series
            self.team_baselines = team_df.set_index(['team_abbr', 'game_date'])['baseline'].to_dict()
            
            logging.info(f"Computed rolling team baselines (window={window}) - "
                        f"Global mean: {self.global_mean:.4f}, "
                        f"Unique team-date baselines: {len(self.team_baselines)}")
        elif normal_window_mean:
            baselines = []
            
            for team_abbr, group in team_df.groupby('team_abbr'):
                group = group.sort_values('game_date').reset_index(drop=True)
                
                for idx in range(len(group)):
                    # Get last N games (not including current game)
                    if idx == 0:
                        # First game - use global mean
                        baseline = self.global_mean
                    else:
                        # Get previous games (up to last N)
                        start_idx = max(0, idx - window)
                        last_n = group.iloc[start_idx:idx]['value'].dropna()
                        
                        if len(last_n) == 0:
                            baseline = self.global_mean
                        else:
                            # Simple mean with shrinkage
                            team_mean = last_n.mean()
                            count = len(last_n)
                            baseline = (team_mean * count + self.global_mean * prior_weight) / (count + prior_weight)
                    
                    baselines.append({
                        'team_abbr': team_abbr,
                        'game_date': group.iloc[idx]['game_date'],
                        'baseline': baseline
                    })
            
            # Convert to dictionary with team-level keys
            baselines_df = pd.DataFrame(baselines)
            self.team_baselines = baselines_df.set_index(['team_abbr', 'game_date'])['baseline'].to_dict()
            
            logging.info(f"Computed normal window team baselines (window={window}) - "
                        f"Global mean: {self.global_mean:.4f}, "
                        f"Unique team-date baselines: {len(self.team_baselines)}")
        else:
            # Compute total mean for each team
            team_stats = team_df.groupby("team_abbr")['value'].agg(["mean", "count"])
            team_stats["baseline"] = (
                team_stats["mean"] * team_stats["count"] + self.global_mean * prior_weight
            ) / (team_stats["count"] + prior_weight)
            
            # For static baselines, just use team_abbr as key
            self.team_baselines = team_stats["baseline"].to_dict()
            
            logging.info(f"Computed total team baselines - "
                        f"Global mean: {self.global_mean:.4f}, "
                        f"Unique teams: {len(team_stats)}")
        return

    def _compute_team_baselines_ewm(self, df: pd.DataFrame, target_col: str, prior_weight: int = 3, span: int = 15) -> None:
        """Compute team baselines using exponentially weighted moving average.
        
        Args:
            df: DataFrame containing game_id, game_date, home_abbr, away_abbr, and target column
            target_col: Name of the target column
            prior_weight: Weight for global mean in shrinkage estimation
            span: Controls decay rate (higher = more history, lower = more recent emphasis)
        """
        self.global_mean = df[target_col].mean()
        
        # Determine which team to track based on target name
        is_home_target = target_col.startswith('home_')
        team_col = 'home_abbr' if is_home_target else 'away_abbr'
        
        # Create unified team view
        team_games = []
        for idx, row in df.iterrows():
            team_games.append({
                'team_abbr': row[team_col],
                'game_date': row['game_date'],
                'game_id': row['game_id'],
                'value': row[target_col]
            })
        
        team_df = pd.DataFrame(team_games).sort_values(['team_abbr', 'game_date']).copy()
        
        # Compute EWM for each team (more weight on recent games)
        team_df['ewm_mean'] = (
            team_df.groupby('team_abbr')['value']
            .transform(lambda x: x.ewm(span=span, min_periods=2).mean())
        )
        
        # Apply shrinkage to global mean
        effective_count = span  # Approximate effective sample size for EWM
        team_df['baseline'] = (
            team_df['ewm_mean'] * effective_count + self.global_mean * prior_weight
        ) / (effective_count + prior_weight)
        
        team_df['baseline'] = team_df['baseline'].fillna(self.global_mean)
        
        # KEY FIX: Store baseline keyed by (team_abbr, game_date) for team-level time-series
        self.team_baselines = team_df.set_index(['team_abbr', 'game_date'])['baseline'].to_dict()
        
        logging.info(f"Computed EWM team baselines (span={span}) - "
                    f"Global mean: {self.global_mean:.4f}, "
                    f"Unique team-date baselines: {len(self.team_baselines)}")
        return

    def _train_and_evaluate_model(self, X: pd.DataFrame, y: pd.Series, target: str, df_with_game_id: pd.DataFrame = None):
        """Train and evaluate a model for the given features and target.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            target: Target variable
            df_with_game_id: Original DataFrame with 'game_id' column (required for residual training)
        """
        logging.info(f"Final feature shape for modeling: {X.shape}")
        
        # Determine model type
        is_classification = target in self.classification_targets
        
        # Handle residual target training for non-classification targets
        use_residuals = self.use_residual_training and not is_classification
        
        if use_residuals:
            if df_with_game_id is None or 'game_id' not in df_with_game_id.columns:
                logging.warning(f"Residual training enabled but no game_id column available. Falling back to regular training.")
                use_residuals = False
            else:
                logging.info(f"Using residual target training for {target}")
                # Compute team baselines with exponentially weighted moving average
                self._compute_team_baselines_ewm(df_with_game_id, 'target', prior_weight=1, span=10)
                
                # Determine which team column to use based on target name
                is_home_target = target.startswith('home_')
                team_col = 'home_abbr' if is_home_target else 'away_abbr'
                
                # Add team baseline (handle both time-varying and static baselines)
                if isinstance(self.team_baselines, dict):
                    # Check if it's time-varying (tuple keys) or static (string keys)
                    sample_key = next(iter(self.team_baselines.keys()))
                    if isinstance(sample_key, tuple):
                        # Time-varying baselines (team_abbr, game_date)
                        df_with_game_id['team_baseline'] = df_with_game_id.apply(
                            lambda row: self.team_baselines.get((row[team_col], row['game_date']), self.global_mean),
                            axis=1
                        )
                    else:
                        # Static baselines (team_abbr only)
                        df_with_game_id['team_baseline'] = df_with_game_id[team_col].map(self.team_baselines).fillna(self.global_mean)
                
                df_with_game_id['residual_target'] = df_with_game_id['target'] - df_with_game_id['team_baseline']
                
                # Use residual target for training
                y = df_with_game_id['residual_target'].copy().fillna(0.0)
                logging.info(f"Residual target stats - Mean: {y.mean():.4f}, Std: {y.std():.4f}, Min: {y.min():.4f}, Max: {y.max():.4f}")
        
        # Log top correlations
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        logging.info(f"Top 10 feature correlations with target:\n{correlations.head(10)}")
        
        # cast if is_classification
        if is_classification:
            y = y.astype(int)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logging.info("Training model...")
        if is_classification:
            model = HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=1200,
                max_depth=6,
                early_stopping=True,
                n_iter_no_change=20,
                l2_regularization=0.5,
                random_state=42
            )
        else:
            model = HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=1200,
                max_depth=6,
                early_stopping=True,
                n_iter_no_change=20,
                l2_regularization=0.5,
                random_state=42
            )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        sk_score = model.score(X_test_scaled, y_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"Model performance for Target: {target} -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, Score: {sk_score:.4f}")
        
        # Prepare metrics for JSON
        top_10_features = [
            {"feature": feat, "correlation": float(corr)}
            for feat, corr in correlations.head(10).items()
        ]
        bottom_10_features = [
            {"feature": feat, "correlation": float(corr)}
            for feat, corr in correlations.tail(10).items()
        ]
        
        metrics = {
            "score": float(sk_score),
            "mean_squared_error": float(mse),
            "root_mean_squared_error": float(rmse),
            "mean_absolute_error": float(mae),
            "r2_score": float(r2),
            "num_samples": int(len(X)),
            "num_features": int(X.shape[1]),
            "test_size": int(len(X_test)),
            "top_10_features": top_10_features,
            "bottom_10_features": bottom_10_features
        }
        
        # Save metrics to JSON
        self._save_model_metrics(target, metrics)
        
        # Save model, scaler, and feature names together
        model_dir = self.models_dir
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{target}_model.pkl")
        
        model_bundle = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'use_residual_training': use_residuals
        }
        
        # Add residual training baseline data if used
        if use_residuals:
            model_bundle['global_mean'] = self.global_mean
            model_bundle['team_baselines'] = self.team_baselines
            logging.info(f"Saved residual training baseline data (global_mean, team_baselines) to model bundle")
        
        joblib.dump(model_bundle, model_path)
        logging.info(f"Saved model bundle (model, scaler, feature_names) to {model_path}")
        
        # Clean up memory
        del X_train, X_test, X_train_scaled, X_test_scaled

    def _save_model_metrics(self, target: str, metrics: dict):
        """Save model metrics to model_metrics.json file.
        
        Args:
            target: Target variable
            metrics: Dictionary containing model metrics
        """
        metrics_file = os.path.join(self.models_dir, "model_metrics.json")
        
        # Load existing metrics if file exists
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Add new metrics with target key
        all_metrics[target] = metrics
        
        # Sort by R2 score (best models first)
        sorted_metrics = dict(sorted(
            all_metrics.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        ))
        
        # Save back to file
        with open(metrics_file, 'w') as f:
            json.dump(sorted_metrics, f, indent=2)
        
        logging.info(f"Saved metrics for {target} to {metrics_file}")

    def train_models(self, max_features: int = 3000):
        """Load feature groupings and train models with memory-efficient approach.
        
        Args:
            max_features: Maximum number of features to use (helps prevent memory issues)
        """
        all_targets = self.targets['regression'] + self.targets['classification']
        
        for target in all_targets:
            # Load and merge all feature groups
            df = self._load_and_merge_features(target)
            if df is None:
                continue
            
            # Prepare data for modeling
            logging.info(f"Total data shape before feature selection: {df.shape}")
            
            # Extract features and target
            key_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr']
            
            # CRITICAL: For classification, drop rows where target is NaN
            if target in self.classification_targets:
                df = df.dropna(subset=['target'])

            X = df.drop(columns=key_cols + ['target']).select_dtypes(include=[np.number])
            y = df['target'].fillna(0.0)
            
            # Perform feature selection
            X = self._select_features(X, y, max_features)
            
            # Train and evaluate model
            self._train_and_evaluate_model(X, y, target, df_with_game_id=df[['game_id', 'home_abbr', 'away_abbr', 'game_date', 'target']].copy())
            
            # Clean up memory
            del df, X, y
                
        return

    def upload_models_to_s3(self):
        """Upload all models in the models directory to an S3 bucket."""
        BUCKET_NAME = os.getenv("LEAGUE_PREDICTIONS_BUCKET_NAME")
        if not BUCKET_NAME:
            logging.error("LEAGUE_PREDICTIONS_BUCKET_NAME environment variable not set.")
            return

        for root, _, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith(".pkl") or file.endswith(".json"):
                    local_path = os.path.join(root, file)
                    s3_path = os.path.relpath(local_path, self.models_dir)
                    s3_path = str(os.path.join('nfl', 'games', s3_path)).replace("\\", "/")
                    try:
                        s3.upload_file(local_path, BUCKET_NAME, s3_path)
                        logging.info(f"Uploaded {local_path} to s3://{BUCKET_NAME}/{s3_path}")
                    except Exception as e:
                        logging.error(f"Failed to upload {local_path} to S3: {e}")

        return

if __name__ == "__main__":
    data_obj = DataObject(
        league='nfl',
        storage_mode='local',
        local_root=os.path.join(sys.path[0], "..", "..", "..", "..", "sports-data-storage-copy/")
    )
    trainer = GameModelTrainer(data_obj, use_residual_training=True)

    # trainer.create_feature_groupings()
    
    # Example for debugging a single game
    # fe_instance = trainer.debug_feature_grouping(game_id='202512070gnb', target='home_win')

    # trainer.train_models()

    trainer.upload_models_to_s3()