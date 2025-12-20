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
from datetime import datetime
from dotenv import load_dotenv
from rapidfuzz import process
from mp_sportsipy.nfl.constants import SIMPLE_POSITION_MAPPINGS

load_dotenv()

try:
    from ..data_object import DataObject
    from ..const import BOVADA_BOXSCORE_MAPPINGS_NFL
    from ..props_and_outcomes import PropsAndOutcomes
except ImportError:
    # Get the absolute path of the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(current_dir))
    from data_object import DataObject
    from const import BOVADA_BOXSCORE_MAPPINGS_NFL
    from props_and_outcomes import PropsAndOutcomes

s3 = boto3.client('s3')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OutcomesTrainer:

    def __init__(self, data_obj: DataObject, local_path: Optional[str] = "./"):
        self.data_obj = data_obj
        self.local_path = local_path
        self.features_dir = os.path.join(sys.path[0], "features/")
        self.models_dir = os.path.join(sys.path[0], "models/")
        if self.local_path:
            os.makedirs(self.features_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)
            self.player_train_features_path = os.path.join(sys.path[0], "..", "players/features/")

        self.skill_positions = ['QB', 'RB', 'WR', 'TE']

        self.player_snaps = self.data_obj.player_snaps

        self.PLAYERS = self.player_snaps['player'].unique()
        self.pid_position_mappings = self.player_snaps[['pid', 'pos']].drop_duplicates().set_index('pid').to_dict()['pos']

        return
    
    def best_match(self, name: str):
        try:
            match, score, idx = process.extractOne(name, self.PLAYERS, score_cutoff=80)  # tweak cutoff
            return self.player_snaps.loc[self.player_snaps['player']==match, 'pid'].values[0]
        except:
            logging.error(f"Could not find pid for player_name: {name}")
            return None
        
    def add_player_ids_nfl(self, df: pd.DataFrame):
        tqdm.pandas(desc="Matching player IDs")

        # Get unique player names
        temp_df = df[['player_name']].drop_duplicates().copy()
        temp_df['player_id'] = temp_df['player_name'].progress_apply(self.best_match)
        name_to_pid = dict(zip(temp_df['player_name'], temp_df['player_id']))

        # map player names to player IDs
        df['player_id'] = df['player_name'].map(name_to_pid)
        return df
    
    def get_outcomes(self):
        if self.local_path:
            df = pd.read_csv(f"{self.local_path}nfl_outcomes.csv")
        else: # load from s3
            pass

        df['bovada_date'] = df['bovada_date'].apply(datetime.fromisoformat)
        df['date_collected'] = df['parent_path'].apply(lambda x: datetime.strptime(x, "bovada_data/%y-%m-%d/%H/nfl"))
        df = df.sort_values(by=['bovada_date', 'date_collected'])
        df = self.add_player_ids_nfl(df)
        df['position'] = df['player_id'].map(self.pid_position_mappings).fillna('UNK')
        df['position'] = df['position'].map(SIMPLE_POSITION_MAPPINGS)
        return df

    def write_merged_groupings(self, outcomes: pd.DataFrame):
        
        FEATURE_GROUP_MERGE_COLS = ['pid', 'game_date', 'target']

        first_date = min(outcomes['bovada_date']).date()
        outcome_pids = outcomes['player_id'].unique()

        for bovada_stat, group_df in outcomes.groupby('stat'):
            group_df = group_df[group_df['position'].isin(self.skill_positions)]
            positions = group_df['position'].dropna().unique()

            if bovada_stat in BOVADA_BOXSCORE_MAPPINGS_NFL:
                normal_stats = BOVADA_BOXSCORE_MAPPINGS_NFL[bovada_stat]
            else:
                logging.warning(f"No normal stats found for BOVADA_STAT: {bovada_stat}")
                continue

            if normal_stats:
                logging.info(f"Writing merged feature groups for: {bovada_stat}, {positions}, {normal_stats}")
                for normal_stat in normal_stats:
                    for pos in positions:
                        features_path = os.path.join(self.player_train_features_path, pos, normal_stat)
                        if os.path.exists(features_path):
                            merged_df = pd.DataFrame()
                            for fn in os.listdir(features_path):
                                temp_df = pd.read_csv(f"{features_path}/{fn}")
                                temp_df['date'] = temp_df['game_date'].apply(lambda x: datetime.fromisoformat(x).date())
                                temp_df: pd.DataFrame = temp_df[(temp_df['date'] >= first_date) & (temp_df['pid'].isin(outcome_pids))]
                                temp_df = temp_df.drop(columns=['date'])
                                if merged_df.empty:
                                    merged_df = temp_df
                                else:
                                    merged_df = merged_df.merge(
                                        temp_df,
                                        on=FEATURE_GROUP_MERGE_COLS,
                                        how='left'
                                    )
                            if not merged_df.empty:
                                bovada_features_path = os.path.join(self.features_dir, bovada_stat)
                                os.makedirs(bovada_features_path, exist_ok=True)
                                merged_df = merged_df.drop(columns=['target'], errors='ignore')
                                merged_df.to_csv(f"{bovada_features_path}/{normal_stat}-{pos}-merged_features.csv", index=False)

        return
    
    def add_player_and_stat_outcome_distributions(self, df: pd.DataFrame, bovada_stat: str):
        player_outcome_distributions = pd.read_csv(os.path.join(self.features_dir, "props_and_outcomes/all_player_outcome_distributions.csv"))
        player_outcome_distributions['date'] = player_outcome_distributions['date'].apply(lambda x: datetime.fromisoformat(x).date())
        stat_outcome_distributions = pd.read_csv(os.path.join(self.features_dir, "props_and_outcomes/all_stat_outcome_distributions.csv"))
        stat_outcome_distributions['date'] = stat_outcome_distributions['date'].apply(lambda x: datetime.fromisoformat(x).date())

        player_od = player_outcome_distributions[player_outcome_distributions['stat'] == bovada_stat]
        stat_od = stat_outcome_distributions[stat_outcome_distributions['stat'] == bovada_stat]

        if 'date' not in df.columns:
            df['date'] = df['game_date'].apply(lambda x: datetime.fromisoformat(x).date())

        df = df.merge(
            player_od.rename(columns={'player_id': 'pid'}).drop(columns=['pos', 'stat']),
            on=['pid', 'date'],
            how='left'
        )
        df = df.merge(
            stat_od.drop(columns=['stat']),
            on=['date'],
            how='left',
            suffixes=('_player_od', '_stat_od')
        )
        df = df.drop(columns=['date'])
        return df

    def concat_and_merge_train_data(self, outcomes: pd.DataFrame):

        temp_outcomes = outcomes[['player_id', 'bovada_date', 'stat', 'over_odds', 'under_odds', 'line_value', 'outcome']].copy()
        temp_outcomes = temp_outcomes.rename(columns={ 'player_id': 'pid' }).drop_duplicates()
        temp_outcomes['date'] = temp_outcomes['bovada_date'].apply(lambda x: x.date())

        player_outcome_distributions = pd.read_csv(os.path.join(self.features_dir, "props_and_outcomes/all_player_outcome_distributions.csv"))
        player_outcome_distributions['date'] = player_outcome_distributions['date'].apply(lambda x: datetime.fromisoformat(x).date())
        stat_outcome_distributions = pd.read_csv(os.path.join(self.features_dir, "props_and_outcomes/all_stat_outcome_distributions.csv"))
        stat_outcome_distributions['date'] = stat_outcome_distributions['date'].apply(lambda x: datetime.fromisoformat(x).date())

        for bovada_stat in os.listdir(self.features_dir):
            if bovada_stat.startswith('train') or bovada_stat.startswith('props_and_outcomes'):
                continue

            groupings_path = os.path.join(self.features_dir, bovada_stat)
            if not os.path.isdir(groupings_path):
                continue
                
            files = os.listdir(groupings_path)
            if not files:
                continue
            
            # Check if this is a combined stat
            is_combined = '&' in bovada_stat or 'and' in bovada_stat.lower()

            player_od = player_outcome_distributions[player_outcome_distributions['stat'] == bovada_stat]
            if player_od.empty:
                logging.warning(f"No player outcome distributions found for BOVADA_STAT: {bovada_stat}")
                continue

            stat_od = stat_outcome_distributions[stat_outcome_distributions['stat'] == bovada_stat]
            if stat_od.empty:
                logging.warning(f"No stat outcome distributions found for BOVADA_STAT: {bovada_stat}")
                continue
            
            if is_combined:
                # For combined stats: concatenate by normal_stat, then merge
                normal_stat_groups = {}
                
                for fn in files:
                    normal_stat, position = fn.split("-")[0], fn.split("-")[1]
                    temp_df = pd.read_csv(os.path.join(groupings_path, fn), low_memory=False)
                    temp_df['date'] = temp_df['game_date'].apply(lambda x: datetime.fromisoformat(x).date())
                    
                    # Add outcome data
                    for (pid, date), outcome_row in temp_outcomes[temp_outcomes['stat'] == bovada_stat].groupby(['pid', 'date']):
                        mask = (temp_df['pid'] == pid) & (temp_df['date'] == date)
                        if mask.any():
                            temp_df.loc[mask, 'over_odds'] = outcome_row['over_odds'].values[0]
                            temp_df.loc[mask, 'under_odds'] = outcome_row['under_odds'].values[0]
                            temp_df.loc[mask, 'line_value'] = outcome_row['line_value'].values[0]
                            temp_df.loc[mask, 'outcome'] = outcome_row['outcome'].values[0]
                    
                    # Group by normal_stat
                    if normal_stat not in normal_stat_groups:
                        normal_stat_groups[normal_stat] = []
                    normal_stat_groups[normal_stat].append(temp_df)
                
                # Concatenate files with same normal_stat
                concatenated_dfs = {}
                for normal_stat, dfs in normal_stat_groups.items():
                    concatenated_dfs[normal_stat] = pd.concat(dfs, ignore_index=True)
                    logging.info(f"Concatenated {len(dfs)} files for {bovada_stat} - {normal_stat}")

                # Merge the concatenated dataframes
                if len(concatenated_dfs) > 1:
                    merge_keys = ['pid', 'game_date']
                    merged_df = None
                    
                    for idx, (normal_stat, df) in enumerate(concatenated_dfs.items()):
                        if merged_df is None:
                            # Rename target column for first dataframe
                            df = df.rename(columns={'target': f'target_{normal_stat}'})
                            merged_df: pd.DataFrame = df
                            logging.info(f"Starting merge with {normal_stat} ({len(df)} rows)")
                        else:
                            # Rename target column and drop duplicate outcome columns
                            df = df.rename(columns={'target': f'target_{normal_stat}'})
                            df_to_merge = df.drop(columns=['over_odds', 'under_odds', 'line_value', 'outcome'], errors='ignore')
                            merged_df: pd.DataFrame = merged_df.merge(df_to_merge, on=merge_keys, how='outer', suffixes=('', f'_{normal_stat}'))
                            logging.info(f"Merged with {normal_stat} (result: {len(merged_df)} rows)")

                    # Save merged result
                    train_path = os.path.join(self.features_dir, 'train')
                    os.makedirs(train_path, exist_ok=True)
                    output_path = os.path.join(train_path, f"{bovada_stat}_train_data.csv")
                    merged_df = self.add_player_and_stat_outcome_distributions(merged_df, bovada_stat)
                    merged_df.to_csv(output_path, index=False)
                    logging.info(f"Saved combined stat training data: {output_path}")
                else:
                    # Only one normal_stat, save it directly
                    train_path = os.path.join(self.features_dir, 'train')
                    os.makedirs(train_path, exist_ok=True)
                    output_path = os.path.join(train_path, f"{bovada_stat}_train_data.csv")
                    merged_df = list(concatenated_dfs.values())[0]
                    merged_df = self.add_player_and_stat_outcome_distributions(merged_df, bovada_stat)
                    merged_df.to_csv(output_path, index=False)
                    logging.info(f"Saved single stat training data: {output_path}")
            
            else:
                # For simple stats: concatenate all files across positions
                all_dfs = []
                
                for fn in files:
                    normal_stat, position = fn.split("-")[0], fn.split("-")[1]
                    temp_df = pd.read_csv(os.path.join(groupings_path, fn), low_memory=False)
                    temp_df['date'] = temp_df['game_date'].apply(lambda x: datetime.fromisoformat(x).date())
                    
                    # Add outcome data
                    for (pid, date), outcome_row in temp_outcomes[temp_outcomes['stat'] == bovada_stat].groupby(['pid', 'date']):
                        mask = (temp_df['pid'] == pid) & (temp_df['date'] == date)
                        if mask.any():
                            temp_df.loc[mask, 'over_odds'] = outcome_row['over_odds'].values[0]
                            temp_df.loc[mask, 'under_odds'] = outcome_row['under_odds'].values[0]
                            temp_df.loc[mask, 'line_value'] = outcome_row['line_value'].values[0]
                            temp_df.loc[mask, 'outcome'] = outcome_row['outcome'].values[0]

                    all_dfs.append(temp_df)
                
                # Concatenate all position files
                if all_dfs:
                    concatenated_df: pd.DataFrame = pd.concat(all_dfs, ignore_index=True)
                    train_path = os.path.join(self.features_dir, 'train')
                    os.makedirs(train_path, exist_ok=True)
                    output_path = os.path.join(train_path, f"{bovada_stat}_train_data.csv")
                    concatenated_df = self.add_player_and_stat_outcome_distributions(concatenated_df, bovada_stat)
                    concatenated_df.to_csv(output_path, index=False)
                    logging.info(f"Concatenated {len(all_dfs)} files for {bovada_stat} ({len(concatenated_df)} total rows)")

        return

    def _save_model_metrics(self, bovada_stat: str, metrics: dict):
        """Save model metrics to model_metrics.json file.
        
        Args:
            bovada_stat: Bovada stat name
            metrics: Dictionary containing model metrics
        """
        metrics_file = os.path.join(self.models_dir, "model_metrics.json")
        
        # Load existing metrics if file exists
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Add new metrics
        all_metrics[bovada_stat] = metrics
        
        # Sort by R2 score (best models first)
        sorted_metrics = dict(sorted(
            all_metrics.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        ))
        
        # Save back to file
        with open(metrics_file, 'w') as f:
            json.dump(sorted_metrics, f, indent=2)
        
        logging.info(f"Saved metrics for {bovada_stat} to {metrics_file}")

    def _train_and_evaluate_model(self, X: pd.DataFrame, y: pd.Series, bovada_stat: str):
        """Train and evaluate a model for the given features and target.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            bovada_stat: Bovada stat name
        """
        logging.info(f"Final feature shape for modeling: {X.shape}")
        
        # Remove zero-variance features to avoid division by zero in correlation calculation
        X_filled = X.fillna(0)
        y_filled = y.fillna(0)
        
        # Filter out features with zero standard deviation
        feature_stds = X_filled.std()
        non_zero_var_features = feature_stds[feature_stds > 0].index.tolist()
        X_filtered = X_filled[non_zero_var_features]
        
        # Log top correlations (suppress numpy warnings about NaN in correlations)
        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = X_filtered.corrwith(y_filled).abs()
        correlations = correlations.dropna().sort_values(ascending=False)
        logging.info(f"Top 10 feature correlations with target:\n{correlations.head(10)}")
        
        # Use filtered features for training
        X = X_filtered
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logging.info("Training model...")
        model = HistGradientBoostingClassifier()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        sk_score = model.score(X_test_scaled, y_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"Model performance for {bovada_stat} -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, Score: {sk_score:.4f}")
        
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
        self._save_model_metrics(bovada_stat, metrics)
        
        # Save model, scaler, and feature names together
        model_path = os.path.join(self.models_dir, f"{bovada_stat}_model.pkl")
        
        model_bundle = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }
        
        joblib.dump(model_bundle, model_path)
        logging.info(f"Saved model bundle (model, scaler, feature_names) to {model_path}")
        
        # Clean up memory
        del X_train, X_test, X_train_scaled, X_test_scaled

    def train_models(self):
        outcomes = self.get_outcomes()
        # outcomes.to_csv("nfl_outcomes_with_pids.csv", index=False)

        # outcomes = pd.read_csv("nfl_outcomes_with_pids.csv")
        # outcomes['bovada_date'] = outcomes['bovada_date'].apply(datetime.fromisoformat)
        # outcomes['date_collected'] = outcomes['date_collected'].apply(datetime.fromisoformat)

        props_and_outcomes_dir = os.path.join(self.features_dir, "props_and_outcomes/")
        os.makedirs(props_and_outcomes_dir, exist_ok=True)

        props_and_outcomes = PropsAndOutcomes(
            _dir=props_and_outcomes_dir,
            league="nfl",
            props_df=None,
            outcomes_df=outcomes,
            data_obj=self.data_obj
        )

        props_and_outcomes.get_past_player_outcome_distributions()
        props_and_outcomes.get_past_stat_outcome_distributions()

        self.write_merged_groupings(outcomes)

        self.concat_and_merge_train_data(outcomes)

        for bovada_stat in outcomes['stat'].unique():
            train_data_path = os.path.join(self.features_dir, "train/", f"{bovada_stat}_train_data.csv")
            if not os.path.exists(train_data_path):
                logging.warning(f"No training data found for bovada_stat: {bovada_stat}")
                continue

            train_df = pd.read_csv(train_data_path, low_memory=False)
            train_df = train_df.drop(columns=['primary_key', 'props_primary_key', 'parent_path', 'team_abbr', 'id', 'actual'], errors='ignore')
            
            # Drop target columns (they may have suffixes from merging)
            target_cols = [col for col in train_df.columns if col.startswith('target')]
            train_df = train_df.drop(columns=target_cols, errors='ignore')
            
            train_df['outcome'] = train_df['outcome'].map({'UNDER': 0, 'OVER': 1}).fillna(0).astype(int)
            
            # Define columns to exclude from features
            exclude_cols = ['pid', 'game_date', 'player_name', 'player_id', 'stat', 'bovada_date', 'date_collected', 'pos', 'date', 'over_odds', 'under_odds', 'line_value', 'outcome']
            feature_cols = [col for col in train_df.columns if col not in exclude_cols]

            X = train_df[feature_cols]
            X = X.select_dtypes(include=[np.number]).fillna(0)
            y = train_df['outcome'].fillna(0)

            # Train and evaluate model
            self._train_and_evaluate_model(X, y, bovada_stat)

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
                    s3_path = str(os.path.join('nfl', 'outcomes', s3_path)).replace("\\", "/")
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
    
    local_path = "/Users/michaelpaul/Library/CloudStorage/GoogleDrive-michaelandrewpaul97@gmail.com/My Drive/python/winsight_api/dynamo_to_s3/data/"
    trainer = OutcomesTrainer(
        data_obj=data_obj,
        local_path=local_path
    )
    
    # trainer.train_models()
    trainer.upload_models_to_s3()