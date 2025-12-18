from __future__ import annotations

import awswrangler as wr
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
from rapidfuzz import process

from mp_sportsipy.nfl.constants import SIMPLE_POSITION_MAPPINGS

load_dotenv()

try:
    from ..data_object import DataObject
    from ..const import BOVADA_BOXSCORE_MAPPINGS_NFL
except ImportError:
    # Get the absolute path of the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(current_dir))
    from data_object import DataObject
    from const import BOVADA_BOXSCORE_MAPPINGS_NFL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

resource = boto3.resource("dynamodb")

def get_dynamo_table_dataframe(table_name: str):
    return wr.dynamodb.read_partiql_query(
        query=f"""
            SELECT * 
            FROM {table_name}
        """
    )

class OutcomesPredictor:
    def __init__(self, data_obj: DataObject, root_dir: str = "./", local_path: Optional[str] = None, player_features_dir: Optional[str] = None):
        self.data_obj = data_obj
        self.root_dir = root_dir
        self.local_path = local_path
        self.player_features_dir = player_features_dir
        self.features_dir = os.path.join(self.root_dir, "predicted_features/")
        self.models_dir = os.path.join(self.root_dir, "models/")
        self.outcome_predictions_dir = os.path.join(self.root_dir, "outcome_predictions/")
        if self.local_path:
            os.makedirs(self.features_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.outcome_predictions_dir, exist_ok=True)
            self.player_train_features_path = os.path.join(self.root_dir, "players/predicted_features/")
        else:
            self.player_train_features_path = os.path.join(self.player_features_dir, "predicted_features/")

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

    def get_props(self):
        if self.local_path:
            df = pd.read_csv(f"{self.local_path}nfl_props.csv")
        else: # load from s3
            df = get_dynamo_table_dataframe(table_name='nfl_props')

        df['bovada_date'] = df['bovada_date'].apply(datetime.fromisoformat)
        df['date_collected'] = df['parent_path'].apply(lambda x: datetime.strptime(x, "bovada_data/%y-%m-%d/%H/nfl"))
        df = df.sort_values(by=['bovada_date', 'date_collected'])
        df = self.add_player_ids_nfl(df)
        df['position'] = df['player_id'].map(self.pid_position_mappings).fillna('UNK')
        df['position'] = df['position'].map(SIMPLE_POSITION_MAPPINGS)
        return df
    
    def write_merged_groupings(self, props: pd.DataFrame):
        """Write merged feature groupings for predictions, similar to trainer."""
        
        FEATURE_GROUP_MERGE_COLS = ['pid', 'game_date', 'abbr', 'key']

        first_date = min(props['bovada_date']).date()
        prop_pids = props['player_id'].unique()

        for bovada_stat, group_df in props.groupby('stat'):
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
                                temp_df: pd.DataFrame = temp_df[(temp_df['date'] >= first_date) & (temp_df['pid'].isin(prop_pids))]
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
                                merged_df.to_csv(f"{bovada_features_path}/{normal_stat}-{pos}-merged_features.csv", index=False)

        return
    
    def concat_and_merge_predict_data(self, props: pd.DataFrame):
        """Concatenate and merge prediction data, similar to trainer."""

        temp_props = props[['player_id', 'bovada_date', 'stat', 'over_odds', 'under_odds', 'line_value']].copy()
        temp_props = temp_props.rename(columns={ 'player_id': 'pid' }).drop_duplicates()
        temp_props['date'] = temp_props['bovada_date'].apply(lambda x: x.date())

        for bovada_stat in os.listdir(self.features_dir):
            if bovada_stat.startswith('predict'):
                continue
            
            groupings_path = os.path.join(self.features_dir, bovada_stat)
            if not os.path.isdir(groupings_path):
                continue
                
            files = os.listdir(groupings_path)
            if not files:
                continue
            
            # Check if this is a combined stat
            is_combined = '&' in bovada_stat or 'and' in bovada_stat.lower()
            
            if is_combined:
                # For combined stats: concatenate by normal_stat, then merge
                normal_stat_groups = {}
                
                for fn in files:
                    normal_stat, position = fn.split("-")[0], fn.split("-")[1]
                    temp_df = pd.read_csv(os.path.join(groupings_path, fn), low_memory=False)
                    temp_df['date'] = temp_df['game_date'].apply(lambda x: datetime.fromisoformat(x).date())
                    
                    # Add prop data (odds and line values)
                    for (pid, date), prop_row in temp_props[temp_props['stat'] == bovada_stat].groupby(['pid', 'date']):
                        mask = (temp_df['pid'] == pid) & (temp_df['date'] == date)
                        if mask.any():
                            temp_df.loc[mask, 'over_odds'] = prop_row['over_odds'].values[0]
                            temp_df.loc[mask, 'under_odds'] = prop_row['under_odds'].values[0]
                            temp_df.loc[mask, 'line_value'] = prop_row['line_value'].values[0]
                    
                    temp_df = temp_df.drop(columns=['date'])
                    
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
                            merged_df = df
                            logging.info(f"Starting merge with {normal_stat} ({len(df)} rows)")
                        else:
                            df_to_merge = df.drop(columns=['over_odds', 'under_odds', 'line_value'], errors='ignore')
                            merged_df = merged_df.merge(df_to_merge, on=merge_keys, how='outer', suffixes=('', f'_{normal_stat}'))
                            logging.info(f"Merged with {normal_stat} (result: {len(merged_df)} rows)")
                    
                    # Save merged result
                    predict_path = os.path.join(self.features_dir, 'predict')
                    os.makedirs(predict_path, exist_ok=True)
                    output_path = os.path.join(predict_path, f"{bovada_stat}_predict_data.csv")
                    merged_df.to_csv(output_path, index=False)
                    logging.info(f"Saved combined stat prediction data: {output_path}")
                else:
                    # Only one normal_stat, save it directly
                    predict_path = os.path.join(self.features_dir, 'predict')
                    os.makedirs(predict_path, exist_ok=True)
                    output_path = os.path.join(predict_path, f"{bovada_stat}_predict_data.csv")
                    list(concatenated_dfs.values())[0].to_csv(output_path, index=False)
                    logging.info(f"Saved single stat prediction data: {output_path}")
            
            else:
                # For simple stats: concatenate all files across positions
                all_dfs = []
                
                for fn in files:
                    normal_stat, position = fn.split("-")[0], fn.split("-")[1]
                    temp_df = pd.read_csv(os.path.join(groupings_path, fn), low_memory=False)
                    temp_df['date'] = temp_df['game_date'].apply(lambda x: datetime.fromisoformat(x).date())
                    
                    # Add prop data (odds and line values)
                    for (pid, date), prop_row in temp_props[temp_props['stat'] == bovada_stat].groupby(['pid', 'date']):
                        mask = (temp_df['pid'] == pid) & (temp_df['date'] == date)
                        if mask.any():
                            temp_df.loc[mask, 'over_odds'] = prop_row['over_odds'].values[0]
                            temp_df.loc[mask, 'under_odds'] = prop_row['under_odds'].values[0]
                            temp_df.loc[mask, 'line_value'] = prop_row['line_value'].values[0]
                    
                    temp_df = temp_df.drop(columns=['date'])
                    all_dfs.append(temp_df)
                
                # Concatenate all position files
                if all_dfs:
                    concatenated_df = pd.concat(all_dfs, ignore_index=True)
                    predict_path = os.path.join(self.features_dir, 'predict')
                    os.makedirs(predict_path, exist_ok=True)
                    output_path = os.path.join(predict_path, f"{bovada_stat}_predict_data.csv")
                    concatenated_df.to_csv(output_path, index=False)
                    logging.info(f"Concatenated {len(all_dfs)} files for {bovada_stat} ({len(concatenated_df)} total rows)")

        return

    def create_predictions(self, target_props: pd.DataFrame) -> pd.DataFrame:

        predictions = []

        for bovada_stat in target_props['stat'].unique():
            predict_data_path = os.path.join(self.features_dir, "predict/", f"{bovada_stat}_predict_data.csv")
            if not os.path.exists(predict_data_path):
                logging.warning(f"No prediction data found for bovada_stat: {bovada_stat}")
                continue

            # Load model bundle
            model_path = os.path.join(self.models_dir, f"{bovada_stat}_model.pkl")
            if not os.path.exists(model_path):
                logging.warning(f"No model found for bovada_stat: {bovada_stat}")
                continue

            model_bundle = joblib.load(model_path)
            model = model_bundle['model']
            scaler = model_bundle['scaler']
            feature_names = model_bundle['feature_names']

            # Load prediction data
            predict_df = pd.read_csv(predict_data_path, low_memory=False)
            
            # Ensure required columns exist
            if 'pid' not in predict_df.columns or 'game_date' not in predict_df.columns:
                logging.error(f"Missing required columns (pid, game_date) in prediction data for {bovada_stat}")
                continue
            
            # Save identifying columns (only those that exist)
            potential_id_cols = ['pid', 'game_date', 'over_odds', 'under_odds', 'line_value']
            id_cols = [col for col in potential_id_cols if col in predict_df.columns]
            identify_df = predict_df[id_cols].copy()
            
            # Add missing odds/line columns with NaN if they don't exist
            for col in ['over_odds', 'under_odds', 'line_value']:
                if col not in identify_df.columns:
                    identify_df[col] = np.nan
            
            # Drop target columns and non-feature columns
            target_cols = [col for col in predict_df.columns if col.startswith('target')]
            predict_df = predict_df.drop(columns=target_cols, errors='ignore')
            
            # Define columns to exclude from features
            exclude_cols = ['pid', 'game_date', 'player_name', 'player_id', 'stat', 'bovada_date', 'date_collected', 'pos', 'date', 'over_odds', 'under_odds', 'line_value']
            feature_cols = [col for col in predict_df.columns if col not in exclude_cols]

            # Prepare features
            X = predict_df[feature_cols]
            X = X.select_dtypes(include=[np.number]).fillna(0)
            
            # Align features with model's expected feature names
            missing_cols = set(feature_names) - set(X.columns)
            extra_cols = set(X.columns) - set(feature_names)
            
            if missing_cols:
                logging.warning(f"Missing features for {bovada_stat}: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0
                    
            if extra_cols:
                logging.info(f"Extra features (will be dropped) for {bovada_stat}: {extra_cols}")
                X = X.drop(columns=list(extra_cols))
            
            # Reorder to match training feature order
            X = X[feature_names]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
            
            # Add predictions to identifying data
            identify_df['predicted_outcome'] = ['OVER' if pred == 1 else 'UNDER' for pred in y_pred]
            identify_df['stat'] = bovada_stat
            
            if y_pred_proba is not None:
                identify_df['over_probability'] = y_pred_proba[:, 1]
                identify_df['under_probability'] = y_pred_proba[:, 0]
            
            predictions.append(identify_df)
            logging.info(f"Generated {len(identify_df)} predictions for {bovada_stat}")

        # Combine all predictions
        if predictions:
            all_predictions: pd.DataFrame = pd.concat(predictions, ignore_index=True)
            
            all_predictions = all_predictions.dropna(subset=['over_odds', 'under_odds', 'line_value'], how='all')
            all_predictions = all_predictions.drop_duplicates(subset=['pid', 'game_date', 'stat'])

            # Merge back with original props data for player names
            all_predictions = all_predictions.merge(
                target_props[['player_id', 'player_name', 'bovada_date', 'stat']].rename(columns={'player_id': 'pid'}),
                on=['pid', 'stat'],
                how='left'
            )
            
            # Save predictions
            for stat, stat_df in all_predictions.groupby('stat'):
                predictions_path = os.path.join(self.outcome_predictions_dir, f"{stat}_prop_predictions.csv")
                stat_df = stat_df.drop_duplicates().sort_values(by=['over_probability'], ascending=False)
                stat_df = stat_df.round(3)
                stat_df.to_csv(predictions_path, index=False)
                logging.info(f"Saved all predictions to {predictions_path}")
            
            return all_predictions
        else:
            logging.warning("No predictions were generated")
            return pd.DataFrame()
    
    def predict_next_props(self):
        props = self.get_props()

        # Filter to recent/upcoming props (1 day buffer)
        target_props = props[props['bovada_date'] >= datetime.now() - timedelta(days=1)].copy()

        self.write_merged_groupings(target_props)

        self.concat_and_merge_predict_data(target_props)

        self.create_predictions(target_props)

        return
    
if __name__ == "__main__":
    # data_obj = DataObject(
    #     league='nfl',
    #     storage_mode='local',
    #     local_root=os.path.join(sys.path[0], "..", "..", "..", "..", "sports-data-storage-copy/")
    # )
    
    # local_path = "/Users/michaelpaul/Library/CloudStorage/GoogleDrive-michaelandrewpaul97@gmail.com/My Drive/python/winsight_api/dynamo_to_s3/data/"
    # predictor = OutcomesPredictor(
    #     data_obj=data_obj,
    #     local_path=local_path
    # )

    data_obj = DataObject(
        storage_mode='s3',
        s3_bucket=os.getenv('SPORTS_DATA_BUCKET_NAME')
    )

    predictor = OutcomesPredictor(
        data_obj=data_obj,
        root_dir="./",
        player_features_dir="../players/predicted_features/"
    )
    
    predictor.predict_next_props()
