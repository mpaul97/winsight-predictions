import pandas as pd
import numpy as np
import os
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import os
import json
import time
import logging
from tqdm import tqdm

try:
    from .const import DATETIME_FORMAT
    from .helpers import get_dynamo_table_dataframe, add_player_ids_nfl
    from .data_object import DataObject
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from const import DATETIME_FORMAT
    from helpers import get_dynamo_table_dataframe, add_player_ids_nfl
    from data_object import DataObject

load_dotenv()

class PropsAndOutcomes:
    def __init__(self, _dir: str, league: str = "nfl", props_df: pd.DataFrame|None = None, outcomes_df: pd.DataFrame|None = None, data_obj: DataObject|None = None):
        self._dir = _dir
        self.league = league
        if props_df is None:
            logging.info(f"Getting ENTIRE {league}_props from dynamo...")
        self.props_df: pd.DataFrame = get_dynamo_table_dataframe(f'{self.league}_props') if props_df is None else props_df
        if outcomes_df is None:
            logging.info(f"Getting ENTIRE {league}_outcomes from dynamo...")
        self.outcomes_df: pd.DataFrame = get_dynamo_table_dataframe(f'{self.league}_outcomes') if outcomes_df is None else outcomes_df
        self.start_date = datetime.now().date() - timedelta(days=4)
        # self.start_date = datetime.strptime("20/05/2025, 20:00:00", DATETIME_FORMAT)
        if self.props_df['bovada_date'].dtype == 'datetime64[ns]':
            self.props_df['bovada_datetime_obj'] = self.props_df['bovada_date']

        if self.props_df['bovada_date'].dtype == 'string':
            self.props_df['bovada_datetime_obj'] = self.props_df['bovada_date'].apply(lambda x: datetime.strptime(x, DATETIME_FORMAT))

        if 'bovada_datetime_obj' not in self.props_df.columns:
            self.props_df['bovada_datetime_obj'] = self.props_df['bovada_date'].apply(lambda x: datetime.strptime(x, DATETIME_FORMAT))

        self.props_df['bovada_date_obj'] = self.props_df['bovada_datetime_obj'].apply(lambda x: x.date())
        # ONLY upcoming props/games
        self.props_df = self.props_df[self.props_df['bovada_date_obj']>=self.start_date]
        # data object
        if data_obj is not None:
            self.data_obj = data_obj
        else:
            self.data_obj = DataObject(
                league='nfl',
                storage_mode='s3',
                s3_bucket=os.getenv("SPORTS_DATA_BUCKET_NAME")
            )
        self.player_snaps = self.data_obj.player_snaps
        self.PLAYERS = self.player_snaps['player'].unique()
        if self.league == 'nfl':
            self.props_df = add_player_ids_nfl(self.props_df, PLAYERS=self.PLAYERS, player_snaps=self.player_snaps)
            self.outcomes_df = add_player_ids_nfl(self.outcomes_df, PLAYERS=self.PLAYERS, player_snaps=self.player_snaps)
        self.players_df = self.props_df[['player_name', 'player_id']].drop_duplicates()
        self.pid_position_mappings = self.player_snaps[['pid', 'pos']].drop_duplicates().set_index('pid').to_dict()['pos']
        return
    def get_props(self):
        return get_dynamo_table_dataframe(f'{self.league}_props')
    def get_outcomes(self):
        return get_dynamo_table_dataframe(f'{self.league}_outcomes')
    def get_past_player_outcome_distributions(self):
        df: pd.DataFrame = self.outcomes_df.copy()[['id', 'bovada_date', 'player_id', 'stat', 'outcome']].drop_duplicates()

        if df['bovada_date'].dtype == 'datetime64[ns]':
            df['bovada_date'] = df['bovada_date'].apply(lambda x: x.date())

        if df['bovada_date'].dtype == 'string':
            df['bovada_date'] = df['bovada_date'].apply(lambda x: datetime.strptime(x, DATETIME_FORMAT).date())

        dates = df['bovada_date'].drop_duplicates().values
        dates.sort()
        all_dfs = []
        for date in dates:
            df_list = []
            pids = df[df['bovada_date']==date]['player_id'].drop_duplicates().values
            logging.info(f"Getting player_outcome_distributions for: {date}, with pids length: {len(pids)}")
            for pid in pids:
                temp_df: pd.DataFrame = df.copy()[(df['player_id']==pid)&(df['bovada_date']<date)]
                # Get counts and group totals
                outcome_counts = temp_df.groupby(by=['stat'])['outcome'].value_counts().reset_index(name='count')
                group_totals = temp_df.groupby(by=['stat']).size().reset_index(name='group_total')
                # Merge and calculate proportion
                outcome_counts = outcome_counts.merge(group_totals, on='stat')
                outcome_counts['proportion'] = round((outcome_counts['count'] / outcome_counts['group_total']) * 100.0, 2)
                outcome_counts['weighted_proportion'] = round((outcome_counts['count'] * outcome_counts['proportion']) / 100.0, 2)
                # Add name and id
                outcome_counts.insert(0, 'player_id', pid)
                df_list.append(outcome_counts)
            new_df = pd.concat(df_list)
            new_df['pos'] = new_df['player_id'].apply(lambda x: self.pid_position_mappings[x])
            # Sort and save
            new_df = new_df.sort_values(by=['weighted_proportion'], ascending=False)
            new_df = new_df.reset_index(drop=True)
            new_df['min_weighted_proportion'] = min(new_df['weighted_proportion']) if not new_df['weighted_proportion'].empty else np.nan
            new_df['max_weighted_proportion'] = max(new_df['weighted_proportion']) if not new_df['weighted_proportion'].empty else np.nan
            new_df['rank'] = new_df.index + 1
            new_df['max_rank'] = max(new_df['rank']) if not new_df['rank'].empty else np.nan
            new_df.insert(0, 'date', date)
            all_dfs.append(new_df)
        total_df = pd.concat(all_dfs)
        pivoted_df = total_df.pivot(
            index=['date', 'player_id', 'pos', 'stat'],
            columns='outcome',
            values=[
                'count', 'group_total', 'proportion', 'weighted_proportion', 
                'min_weighted_proportion', 'max_weighted_proportion', 
                'rank', 'max_rank'
            ]
        )
        # Flatten the multi-level column index
        pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]
        # Reset the index to make the pivoted columns regular columns
        pivoted_df = pivoted_df.reset_index()
        pivoted_df.to_csv(f"{self._dir}all_player_outcome_distributions.csv", index=False)
        return
    def get_past_stat_outcome_distributions(self):
        df: pd.DataFrame = self.outcomes_df.copy()[['id', 'bovada_date', 'player_id', 'stat', 'outcome']].drop_duplicates()
        
        if df['bovada_date'].dtype == 'datetime64[ns]':
            df['bovada_date'] = df['bovada_date'].apply(lambda x: x.date())

        if df['bovada_date'].dtype == 'string':
            df['bovada_date'] = df['bovada_date'].apply(lambda x: datetime.strptime(x, DATETIME_FORMAT).date())
            
        dates = df['bovada_date'].drop_duplicates().values
        dates.sort()
        all_dfs = []
        for date in dates:
            temp_df: pd.DataFrame = df.copy()[df['bovada_date']<date]
            # Get counts and group totals
            outcome_counts = temp_df.groupby(by=['stat'])['outcome'].value_counts().reset_index(name='count')
            group_totals = temp_df.groupby(by=['stat']).size().reset_index(name='group_total')
            # Merge and calculate proportion
            outcome_counts = outcome_counts.merge(group_totals, on='stat')
            outcome_counts['proportion'] = round((outcome_counts['count'] / outcome_counts['group_total']) * 100.0, 2)
            outcome_counts['weighted_proportion'] = round((outcome_counts['count'] * outcome_counts['proportion']) / 100.0, 2)
            # Sort and save
            outcome_counts = outcome_counts.sort_values(by=['weighted_proportion'], ascending=False)
            outcome_counts = outcome_counts.reset_index(drop=True)
            outcome_counts['min_weighted_proportion'] = min(outcome_counts['weighted_proportion']) if not outcome_counts['weighted_proportion'].empty else np.nan
            outcome_counts['max_weighted_proportion'] = max(outcome_counts['weighted_proportion']) if not outcome_counts['weighted_proportion'].empty else np.nan
            outcome_counts['rank'] = outcome_counts.index + 1
            outcome_counts['max_rank'] = max(outcome_counts['rank']) if not outcome_counts['rank'].empty else np.nan
            outcome_counts.insert(0, 'date', date)
            all_dfs.append(outcome_counts)
        total_df = pd.concat(all_dfs)
        pivoted_df = total_df.pivot(
            index=['date', 'stat'],
            columns='outcome',
            values=[
                'count', 'group_total', 'proportion', 'weighted_proportion', 
                'min_weighted_proportion', 'max_weighted_proportion', 
                'rank', 'max_rank'
            ]
        )

        # Flatten the multi-level column index
        pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]
        # Reset the index to make the pivoted columns regular columns
        pivoted_df = pivoted_df.reset_index()
        pivoted_df.to_csv(f"{self._dir}all_stat_outcome_distributions.csv", index=False)
        return
    def get_player_outcome_distributions(self):
        df: pd.DataFrame = self.outcomes_df.copy()[['id', 'player_id', 'stat', 'outcome']].drop_duplicates()
        df_list = []
        for _, row in tqdm(self.players_df.iterrows(), total=len(self.players_df), desc="Processing player outcome distributions"):
            player_name, player_id = row[['player_name', 'player_id']]
            temp_df: pd.DataFrame = df.copy()[df['player_id']==player_id]
            # Get counts and group totals
            outcome_counts = temp_df.groupby(by=['stat'])['outcome'].value_counts().reset_index(name='count')
            group_totals = temp_df.groupby(by=['stat']).size().reset_index(name='group_total')
            # Merge and calculate proportion
            outcome_counts = outcome_counts.merge(group_totals, on='stat')
            outcome_counts['proportion'] = round((outcome_counts['count'] / outcome_counts['group_total']) * 100.0, 2)
            outcome_counts['weighted_proportion'] = round((outcome_counts['count'] * outcome_counts['proportion']) / 100.0, 2)
            # Add name and id
            outcome_counts.insert(0, 'player_name', player_name)
            outcome_counts.insert(1, 'player_id', player_id)
            df_list.append(outcome_counts)
        if len(df_list)==0:
            logging.info(f"No UPCOMING props found for {self.league}")
            return 0
        new_df: pd.DataFrame = pd.concat(df_list)
        # Sort and save
        new_df: pd.DataFrame = new_df.sort_values(by=['weighted_proportion'], ascending=False)
        new_df = new_df.reset_index(drop=True)
        new_df['min_weighted_proportion'] = min(new_df['weighted_proportion'])
        new_df['max_weighted_proportion'] = max(new_df['weighted_proportion'])
        new_df['rank'] = new_df.index + 1
        new_df['max_rank'] = max(new_df['rank'])
        new_df.to_csv(f"{self._dir}{self.league}_player_outcome_distributions.csv", index=False)
        pivoted_df = new_df.pivot(
            index=['player_id', 'player_name', 'stat'],
            columns='outcome',
            values=[
                'count', 'group_total', 'proportion', 'weighted_proportion', 
                'min_weighted_proportion', 'max_weighted_proportion', 
                'rank', 'max_rank'
            ]
        )

        # Flatten the multi-level column index
        pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]
        # Reset the index to make the pivoted columns regular columns
        pivoted_df = pivoted_df.reset_index()
        pivoted_df.to_csv(f"{self._dir}pivoted_player_outcome_distributions.csv", index=False)
        return
    def get_stat_outcome_distributions(self):
        df: pd.DataFrame = self.outcomes_df.copy()[['id', 'player_id', 'stat', 'outcome']].drop_duplicates()
        # Get counts and group totals
        outcome_counts = df.groupby(by=['stat'])['outcome'].value_counts().reset_index(name='count')
        group_totals = df.groupby(by=['stat']).size().reset_index(name='group_total')
        # Merge and calculate proportion
        outcome_counts = outcome_counts.merge(group_totals, on='stat')
        outcome_counts['proportion'] = round((outcome_counts['count'] / outcome_counts['group_total']) * 100.0, 2)
        outcome_counts['weighted_proportion'] = round((outcome_counts['count'] * outcome_counts['proportion']) / 100.0, 2)
        # Sort and save
        outcome_counts = outcome_counts.sort_values(by=['weighted_proportion'], ascending=False)
        outcome_counts = outcome_counts.reset_index(drop=True)
        outcome_counts['min_weighted_proportion'] = min(outcome_counts['weighted_proportion'])
        outcome_counts['max_weighted_proportion'] = max(outcome_counts['weighted_proportion'])
        outcome_counts['rank'] = outcome_counts.index + 1
        outcome_counts['max_rank'] = max(outcome_counts['rank'])
        outcome_counts.to_csv(f"{self._dir}{self.league}_stat_outcome_distributions.csv", index=False)
        pivoted_df = outcome_counts.pivot(
            index=['stat'],
            columns='outcome',
            values=[
                'count', 'group_total', 'proportion', 'weighted_proportion', 
                'min_weighted_proportion', 'max_weighted_proportion', 
                'rank', 'max_rank'
            ]
        )

        # Flatten the multi-level column index
        pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]
        # Reset the index to make the pivoted columns regular columns
        pivoted_df = pivoted_df.reset_index()
        pivoted_df.to_csv(f"{self._dir}pivoted_stat_outcome_distributions.csv", index=False)
        return
# END PropsAndOutcomes

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pao = PropsAndOutcomes('./', 'nfl')
    pao.get_player_outcome_distributions()