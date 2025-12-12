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

class PlayerModelTrainer:

    def __init__(self, data_obj: DataObject, use_residual_training: bool = False):
        self.data_obj = data_obj
        self.min_games = 2
        self.use_residual_training = use_residual_training
        
        # Residual training attributes
        self.global_mean = None
        self.player_baselines = None

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

        # Pre-load data required by child processes to prevent redundant loading.
        self.data_obj.player_group_ranks
        self.data_obj.game_predictions
        logging.info("Preloaded player group ranks and game predictions for child processes.")

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
            'QB': [
                'attempted_passes', 'completed_passes', 'passing_yards',
                'passing_touchdowns', 'interceptions_thrown',
                'rush_attempts', 'rush_yards', 'rush_touchdowns', 
                'fantasy_points',
                'over_under_completed_passes_22+', 'over_under_passing_yards_250+',
                'over_under_passing_touchdowns_2+', 'over_under_interceptions_thrown_1+',
                'over_under_rush_yards_60+', 'over_under_rush_touchdowns_1+',
            ],
            'RB': [
                'rush_attempts', 'rush_yards', 'rush_touchdowns',
                'times_pass_target', 'receptions', 'receiving_yards',
                'receiving_touchdowns', 'fantasy_points',
                'over_under_rush_yards_60+', 'over_under_rush_touchdowns_1+',
                'over_under_receptions_5+', 'over_under_receiving_yards_60+', 'over_under_receiving_touchdowns_1+',
                'over_under_rush_yards_&_receiving_yards_100+', 'over_under_rush_touchdowns_&_receiving_touchdowns_1+'
            ],
            'WR': [
                'times_pass_target', 'receptions', 'receiving_yards', 'receiving_touchdowns',
                'rush_attempts', 'rush_yards', 'fantasy_points',
                'over_under_rush_yards_60+', 'over_under_rush_touchdowns_1+',
                'over_under_receptions_5+', 'over_under_receiving_yards_60+', 'over_under_receiving_touchdowns_1+',
                'over_under_rush_yards_&_receiving_yards_100+', 'over_under_rush_touchdowns_&_receiving_touchdowns_1+'
            ],
            'TE': [
                'times_pass_target', 'receptions', 'receiving_yards', 'receiving_touchdowns',
                'fantasy_points',
                'over_under_receptions_5+', 'over_under_receiving_yards_60+', 'over_under_receiving_touchdowns_1+'
            ]
        }

        self.classification_targets = [
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
        ]

        self.player_data = self.data_obj.player_data

        return
    
    def _build_feature_engine(
        self,
        prior_games: pd.DataFrame,
        target: str,
        row: pd.Series,
        position: str
    ) -> FeatureEngine:
        """Build FeatureEngine with optional cached shared features.
        
        Args:
            prior_games: Historical games for the player
            target: Target variable name
            row: Current game row
            position: Player position
            predictions: Optional predictions for dependent features
            cached_shared_features: Pre-computed shared features to inject into FeatureEngine
            
        Returns:
            FeatureEngine instance with features computed or injected
        """
        return FeatureEngine(
            prior_games=prior_games,
            target_name=target,
            row=row,
            position=position,
            player_data=self.data_obj.player_data,
            player_data_big_plays=self.data_obj.player_data[
                ['key','game_date','home_abbr','away_abbr','abbr','pos', *self.data_obj.big_play_stat_columns]
            ] if not self.data_obj.player_data.empty else pd.DataFrame(),
            standings=self.data_obj.standings,
            team_ranks=self.data_obj.team_ranks,
            player_group_ranks=self.data_obj.player_group_ranks,
            advanced_stat_cols=self.data_obj.advanced_stat_cols,
            big_play_stat_columns=self.data_obj.big_play_stat_columns,
            game_predictions=self.data_obj.game_predictions
        )

    def _process_player_features(self, player_id: str, player_games: pd.DataFrame, position: str, targets: List[str]) -> Dict[str, pd.DataFrame]:
        try:
            player_frames = {}
            player_games = player_games.sort_values('game_date').reset_index(drop=True)

            for target in targets:
                # Check if target exists in the data
                if target not in player_games.columns:
                    logging.warning(f"Target '{target}' not found in columns for player {player_id}. Skipping.")
                    continue
                    
                for idx, row in player_games.iterrows():
                    if idx > self.min_games:
                        prior_games = player_games.iloc[:idx]
                        try:
                            fe = self._build_feature_engine(
                                prior_games=prior_games,
                                target=target,
                                row=row,
                                position=position
                            )
                            frames = fe.grouped_features_as_dfs
                            for group_name, df in frames.items():
                                df['pid'] = player_id
                                df['game_date'] = row['game_date']
                                df['target'] = row[target]
                                df = df[['pid', 'game_date', 'target'] + [col for col in df.columns if col not in ['pid', 'game_date', 'target']]]
                                fe_key = f"{position}-{target}-{group_name}"
                                if fe_key not in player_frames:
                                    player_frames[fe_key] = df
                                else:
                                    player_frames[fe_key] = pd.concat([player_frames[fe_key], df], ignore_index=True)
                        except Exception as e:
                            logging.error(f"Error processing player {player_id}, target {target}, game idx {idx}: {e}")
                            raise  # Re-raise to see the full traceback
            return player_frames
        except Exception as e:
            logging.error(f"Fatal error in _process_player_features for player {player_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _create_feature_groupings_for_position(self, executor: ProcessPoolExecutor, position: str, targets: list[str]):
        player_data = self.player_data.copy()
        player_data = player_data[player_data['pos'] == position]
        
        if player_data.empty:
            logging.warning(f"No player data found for position: {position}")
            return

        all_players = player_data['pid'].unique()
        logging.info(f"Found {len(all_players)} players for position {position}")
        
        all_frames = {}
        
        # Limit concurrency to avoid overwhelming the system - reduce for debugging
        max_workers = 8 # Limit to 8 concurrent processes
        sem = asyncio.Semaphore(max_workers)

        async def process_player_with_semaphore(player_id):
            async with sem:
                try:
                    loop = asyncio.get_running_loop()
                    player_games = player_data[player_data['pid'] == player_id]
                    
                    logging.debug(f"Processing player {player_id} with {len(player_games)} games")
                    
                    # Run the synchronous, CPU-bound function in a separate process
                    player_result = await loop.run_in_executor(
                        executor,
                        self._process_player_features,
                        player_id,
                        player_games,
                        position,
                        targets
                    )
                    return player_result
                except Exception as e:
                    logging.error(f"Failed to process player {player_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    return {}  # Return empty dict instead of crashing

        tasks = [process_player_with_semaphore(pid) for pid in all_players]
        
        successful = 0
        failed = 0
        
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {position} players"):
            try:
                player_frames = await future
                if player_frames:
                    successful += 1
                    for fe_key, df in player_frames.items():
                        if fe_key not in all_frames:
                            all_frames[fe_key] = df
                        else:
                            all_frames[fe_key] = pd.concat([all_frames[fe_key], df], ignore_index=True)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logging.error(f"Error collecting player results: {e}")

        logging.info(f"Position {position}: {successful} players processed successfully, {failed} failed")

        for fe_key, df in all_frames.items():
            logging.info(f"Feature Group: {fe_key}, Shape: {df.shape}")
            pos, target, group_name = fe_key.split("-")
            _dir = os.path.join(self.features_dir, pos, target)
            os.makedirs(_dir, exist_ok=True)
            df.to_csv(f"{_dir}/{group_name}_features.csv", index=False)
        
        all_frames.clear()

    async def _run_all_feature_groupings(self, positions: List[str], targets_override: Optional[list[str]] = None):
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            tasks = []
            for pos in positions:
                pos_targets = targets_override if targets_override else self.targets.get(pos, [])
                if not pos_targets:
                    logging.warning(f"No targets defined for position: {pos}")
                    continue
                
                logging.info(f"Queueing feature grouping creation for position: {pos}")
                tasks.append(self._create_feature_groupings_for_position(executor, pos, pos_targets))
            
            await asyncio.gather(*tasks, return_exceptions=True)

    def create_feature_groupings(self, position: Optional[str] = None, targets: Optional[list[str]] = None):
        if position:
            positions = [position]
        else:
            positions = list(self.targets.keys())

        try:
            asyncio.run(self._run_all_feature_groupings(positions, targets))
        except Exception as e:
            logging.error(f"An error occurred during feature creation: {e}")
            traceback.print_exc()

        return

    def debug_feature_grouping(self, pid: str, position: str, target: str):
        """
        Builds and inspects feature groupings for a single player game for debugging.

        Args:
            pid: The player's ID (e.g., 'LoveJo03').
            position: The player's position (e.g., 'QB').
            target: The target variable to generate features for (e.g., 'passing_yards').
        
        Returns:
            The populated FeatureEngine instance for inspection.
        """
        player_games = self.data_obj.player_data[self.data_obj.player_data['pid'] == pid].copy()
        player_games = player_games.sort_values('game_date').reset_index(drop=True)

        if len(player_games) < self.min_games + 1:
            logging.error(f"Not enough games for player {pid} to debug. Found {len(player_games)}, need at least {self.min_games + 1}.")
            return None

        # Use the last game as the target row and all games before it as prior_games
        row = player_games.iloc[-1]
        prior_games = player_games.iloc[:-1]
        
        logging.info(f"Debugging FeatureEngine for player {pid} ({position}) targeting '{target}' for game on {row['game_date'].date()}")
        logging.info(f"Using {len(prior_games)} prior games.")

        fe = self._build_feature_engine(
            prior_games=prior_games,
            target=target,
            row=row,
            position=position
        )

        frames = fe.grouped_features_as_dfs
        for group_name, df in frames.items():
            print("-" * 50)
            print(f"Feature Group: '{group_name}'")
            print(f"Shape: {df.shape}")
            print("Head:")
            print(df.head())
            print("-" * 50)
            
        return fe

    def _load_and_merge_features(self, position: str, target: str) -> Optional[pd.DataFrame]:
        """Load and merge all feature groups for a specific position and target.
        
        Args:
            position: Player position
            target: Target variable
            
        Returns:
            Merged DataFrame or None if loading fails
        """
        from sklearn.feature_selection import VarianceThreshold
        
        feature_group_dir = os.path.join(self.features_dir, position, target)
        if not os.path.exists(feature_group_dir):
            logging.debug(f"Feature directory does not exist: {feature_group_dir}. Skipping.")
            return None
        
        feature_files = [f for f in os.listdir(feature_group_dir) if f.endswith('_features.csv')]
        if not feature_files:
            logging.warning(f"No feature files found for Position: {position}, Target: {target}. Skipping.")
            return None
        
        logging.info(f"Loading {len(feature_files)} feature groups for Position: {position}, Target: {target}")
        
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
                # Use format='mixed' to handle both "2025-11-09" and "2025-11-09 12:00:00" formats
                if 'game_date' in features_df.columns:
                    features_df['game_date'] = pd.to_datetime(features_df['game_date'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Ensure pid is string
                if 'pid' in features_df.columns:
                    features_df['pid'] = features_df['pid'].astype(str)
                
                # Check for duplicates in merge keys
                dup_count = features_df.duplicated(subset=['pid', 'game_date', 'target']).sum()
                if dup_count > 0:
                    logging.warning(f"  WARNING: Found {dup_count} duplicate rows in {group_name}, removing duplicates")
                    features_df = features_df.drop_duplicates(subset=['pid', 'game_date', 'target'], keep='first')
                
                logging.info(f"    Feature group {group_name} has {len(features_df)} rows")
                
                # Optimize data types to reduce memory
                for col in features_df.columns:
                    if col in ['pid', 'game_date', 'target']:
                        continue
                    # Convert to float32 instead of float64 to save memory
                    if features_df[col].dtype in ['float64', 'int64']:
                        try:
                            features_df[col] = features_df[col].astype('float32')
                        except:
                            pass
                
                # Rename columns
                features_df.columns = [f"{col}_{group_name}" if col not in ['pid', 'game_date', 'target'] else col for col in features_df.columns]
                
                # Merge incrementally
                if df is None:
                    df = features_df
                    logging.info(f"    Initial dataframe: {df.shape}")
                else:
                    # Normalize merge keys in existing df too
                    if 'game_date' in df.columns:
                        df['game_date'] = pd.to_datetime(df['game_date'], format='mixed').dt.strftime('%Y-%m-%d %H:%M:%S')
                    if 'pid' in df.columns:
                        df['pid'] = df['pid'].astype(str)
                    
                    # Validate merge will work correctly
                    before_rows = len(df)
                    df = pd.merge(df, features_df, on=['pid', 'game_date', 'target'], how='inner', validate='1:1')
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
            logging.warning(f"No features loaded for Position: {position}, Target: {target}.")
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

    def _compute_player_baselines(self, df: pd.DataFrame, target_col: str, prior_weight: int = 5, window: int = 15, use_rolling: bool = True, normal_window_mean: bool = False) -> None:
        """Compute player-level shrinkage baselines using rolling window or total mean.
        
        Args:
            df: DataFrame containing pid, game_date, and target column
            target_col: Name of the target column
            prior_weight: Weight for global mean in shrinkage estimation
            rolling_window: Number of recent games to use for baseline (default: 15, only used if use_rolling=True)
            use_rolling: If True, uses rolling window mean. If False, uses total career mean per player
        """
        self.global_mean = df[target_col].mean()
        
        # Sort by player and date to ensure chronological order
        df_sorted = df.sort_values(['pid', 'game_date']).copy()
        
        if use_rolling:
            # Compute rolling mean for each player
            df_sorted['rolling_mean'] = (
                df_sorted.fillna(0.0).groupby('pid')[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=3).mean())
            )
            df_sorted['rolling_count'] = (
                df_sorted.fillna(0.0).groupby('pid')[target_col]
                .transform(lambda x: x.rolling(window=window, min_periods=3).count())
            )
            
            # Apply shrinkage: blend rolling mean with global mean
            df_sorted['baseline'] = (
                df_sorted['rolling_mean'] * df_sorted['rolling_count'] + 
                self.global_mean * prior_weight
            ) / (df_sorted['rolling_count'] + prior_weight)
            
            # Fill NaN with global mean (for players with < min_periods games)
            df_sorted['baseline'] = df_sorted['baseline'].fillna(self.global_mean)
            
            # Create baseline dictionary keyed by (pid, game_date)
            # This allows time-varying baselines
            self.player_baselines = df_sorted.set_index(['pid', 'game_date'])['baseline'].to_dict()
            
            logging.info(f"Computed rolling player baselines (window={window}) - "
                        f"Global mean: {self.global_mean:.4f}, "
                        f"Unique player-game baselines: {len(self.player_baselines)}")
        elif normal_window_mean:
        
            baselines = []
    
            for pid, group in df_sorted.groupby('pid'):
                group = group.sort_values('game_date').reset_index(drop=True)
                
                for idx in range(len(group)):
                    # Get last N games (not including current game)
                    if idx == 0:
                        # First game - use global mean
                        baseline = self.global_mean
                    else:
                        # Get previous games (up to last N)
                        start_idx = max(0, idx - window)
                        last_n = group.iloc[start_idx:idx][target_col].dropna()
                        
                        if len(last_n) == 0:
                            baseline = self.global_mean
                        else:
                            # Simple mean with shrinkage
                            player_mean = last_n.mean()
                            count = len(last_n)
                            baseline = (player_mean * count + self.global_mean * prior_weight) / (count + prior_weight)
                    
                    baselines.append({
                        'pid': pid,
                        'game_date': group.iloc[idx]['game_date'],
                        'baseline': baseline
                    })
    
            # Convert to dictionary
            baselines_df = pd.DataFrame(baselines)
            self.player_baselines = baselines_df.set_index(['pid', 'game_date'])['baseline'].to_dict()

        else:
            # Compute total career mean for each player
            player_stats = df.groupby("pid")[target_col].agg(["mean", "count"])
            player_stats["baseline"] = (
                player_stats["mean"] * player_stats["count"] + self.global_mean * prior_weight
            ) / (player_stats["count"] + prior_weight)
            self.player_baselines = player_stats["baseline"].to_dict()
            
            logging.info(f"Computed total career player baselines - "
                        f"Global mean: {self.global_mean:.4f}, "
                        f"Unique players: {len(self.player_baselines)}")
        return

    def _compute_player_baselines_ewm(self, df: pd.DataFrame, target_col: str, prior_weight: int = 3, span: int = 20) -> None:
        """Compute player baselines using exponentially weighted moving average.
        
        Args:
            span: Controls decay rate (higher = more history, lower = more recent emphasis)
        """
        self.global_mean = df[target_col].mean()
        
        df_sorted = df.sort_values(['pid', 'game_date']).copy()
        
        # Compute EWM for each player (more weight on recent games)
        df_sorted['ewm_mean'] = (
            df_sorted.groupby('pid')[target_col]
            .transform(lambda x: x.ewm(span=span, min_periods=3).mean())
        )
        
        # Apply shrinkage to global mean
        effective_count = span  # Approximate effective sample size for EWM
        df_sorted['baseline'] = (
            df_sorted['ewm_mean'] * effective_count + self.global_mean * prior_weight
        ) / (effective_count + prior_weight)
        
        df_sorted['baseline'] = df_sorted['baseline'].fillna(self.global_mean)
        
        self.player_baselines = df_sorted.set_index(['pid', 'game_date'])['baseline'].to_dict()
        
        logging.info(f"Computed EWM player baselines (span={span}) - "
                    f"Global mean: {self.global_mean:.4f}")
        return

    def _train_and_evaluate_model(self, X: pd.DataFrame, y: pd.Series, position: str, target: str, df_with_pid: pd.DataFrame = None):
        """Train and evaluate a model for the given features and target.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            position: Player position
            target: Target variable
            df_with_pid: Original DataFrame with 'pid' column (required for residual training)
        """
        logging.info(f"Final feature shape for modeling: {X.shape}")
        
        # Handle residual target training for non-classification targets
        is_classification = target in self.classification_targets
        use_residuals = self.use_residual_training and not is_classification
        
        if use_residuals:
            if df_with_pid is None or 'pid' not in df_with_pid.columns:
                logging.warning(f"Residual training enabled but no pid column available. Falling back to regular training.")
                use_residuals = False
            else:
                logging.info(f"Using residual target training for {target}")
                # Compute player baselines with exponentially weighted moving average
                self._compute_player_baselines_ewm(df_with_pid, 'target', prior_weight=1, span=10)
                
                # Add player baseline (handle both time-varying and static baselines)
                if isinstance(self.player_baselines, dict):
                    # Check if it's time-varying (tuple keys) or static (string keys)
                    sample_key = next(iter(self.player_baselines.keys()))
                    if isinstance(sample_key, tuple):
                        # Time-varying baselines (pid, game_date)
                        df_with_pid['player_baseline'] = df_with_pid.apply(
                            lambda row: self.player_baselines.get((row['pid'], row['game_date']), self.global_mean),
                            axis=1
                        )
                    else:
                        # Static baselines (pid only)
                        df_with_pid['player_baseline'] = df_with_pid['pid'].map(self.player_baselines).fillna(self.global_mean)
                
                df_with_pid['residual_target'] = df_with_pid['target'] - df_with_pid['player_baseline']
                
                # Use residual target for training
                y = df_with_pid['residual_target'].copy().fillna(0.0)
                logging.info(f"Residual target stats - Mean: {y.mean():.4f}, Std: {y.std():.4f}, Min: {y.min():.4f}, Max: {y.max():.4f}")
        
        # Log top correlations
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        logging.info(f"Top 10 feature correlations with target:\n{correlations.head(10)}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logging.info("Training model...")
        if target in self.classification_targets:
            model = HistGradientBoostingClassifier()
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
        logging.info(f"Model performance for Position: {position}, Target: {target} -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, Score: {sk_score:.4f}")
        
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
        self._save_model_metrics(position, target, metrics)
        
        # Save model, scaler, and feature names together
        model_dir = os.path.join(self.models_dir, position)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{target}_model.pkl")
        
        model_bundle = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }
        
        # Add residual training baseline data if used
        if use_residuals:
            model_bundle['global_mean'] = self.global_mean
            model_bundle['player_baselines'] = self.player_baselines
            model_bundle['use_residual_training'] = True
            logging.info(f"Saved residual training baseline data (global_mean, player_baselines) to model bundle")
        else:
            model_bundle['use_residual_training'] = False
        
        joblib.dump(model_bundle, model_path)
        logging.info(f"Saved model bundle (model, scaler, feature_names) to {model_path}")
        
        # Clean up memory
        del X_train, X_test, X_train_scaled, X_test_scaled

    def _save_model_metrics(self, position: str, target: str, metrics: dict):
        """Save model metrics to model_metrics.json file.
        
        Args:
            position: Player position
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
        
        # Add new metrics with position_target key
        model_key = f"{position}_{target}"
        all_metrics[model_key] = metrics
        
        # Sort by R2 score (best models first)
        sorted_metrics = dict(sorted(
            all_metrics.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        ))
        
        # Save back to file
        with open(metrics_file, 'w') as f:
            json.dump(sorted_metrics, f, indent=2)
        
        logging.info(f"Saved metrics for {model_key} to {metrics_file}")

    def train_models(self, max_features: int = 3000):
        """Load feature groupings and train models with memory-efficient approach.
        
        Args:
            max_features: Maximum number of features to use (helps prevent memory issues)
        """
        for position, targets in self.targets.items():
            for target in targets:
                # Load and merge all feature groups
                df = self._load_and_merge_features(position, target)
                if df is None:
                    continue
                
                # Prepare data for modeling
                logging.info(f"Total data shape before feature selection: {df.shape}")
                
                # Extract features and target
                X = df.drop(columns=['pid', 'game_date', 'target']).select_dtypes(include=[np.number])
                y = df['target'].fillna(0.0)
                
                # Perform feature selection
                X = self._select_features(X, y, max_features)
                
                # Train and evaluate model (pass df with pid for residual training)
                self._train_and_evaluate_model(X, y, position, target, df_with_pid=df[['pid', 'game_date', 'target']].copy())
                
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
                    s3_path = str(os.path.join('nfl', 'players', s3_path)).replace("\\", "/")
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
    trainer = PlayerModelTrainer(data_obj, use_residual_training=True)

    # trainer.create_feature_groupings(position='QB')
    
    # Example for debugging a single player
    # fe_instance = trainer.debug_feature_grouping(pid='LoveJo03', position='QB', target='passing_yards')

    # trainer.train_models()

    # trainer.upload_models_to_s3()
