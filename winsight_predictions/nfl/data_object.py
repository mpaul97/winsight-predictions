import os
import logging
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Iterable
import regex as re
import json
from io import StringIO
try:
    import boto3  # optional; only needed for S3 mode
except ImportError:  # pragma: no cover
    boto3 = None

try:
    from .const import TEAM_TO_PLAYER_ABBR_MAPPINGS, CONFERENCE_MAPPINGS, DIVISION_MAPPINGS
except ImportError:
    from const import TEAM_TO_PLAYER_ABBR_MAPPINGS, CONFERENCE_MAPPINGS, DIVISION_MAPPINGS

from mp_sportsipy.nfl.constants import SIMPLE_POSITION_MAPPINGS

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class DataObject:
    """
    Centralized data management for NFL predictors.
    Handles loading and caching of all data sources including:
    - Game schedules and boxscores
    - Player statistics and snap counts
    - Team and player rankings
    - Play-by-play features
    - Injury reports, previews, standings
    """
    
    def __init__(
        self,
        league: str = "nfl",
        storage_mode: str = "local",  # "local" or "s3"
        local_root: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_client: Optional[object] = None,
    ):
        """Create a DataObject that can operate on local files or S3 objects.

        Args:
            league: League key (e.g. nfl, nba, mlb).
            storage_mode: One of "local" or "s3".
            local_root: Root directory for local storage (defaults to ../../sports-data-storage-copy).
            s3_bucket: Bucket name for S3 mode. Defaults to env SPORTS_DATA_BUCKET_NAME.
            s3_client: Optional pre-configured boto3 client (for testing / dependency injection).
        """
        self.league = league
        self.storage_mode = storage_mode.lower()
        if self.storage_mode not in {"local", "s3"}:
            raise ValueError("storage_mode must be 'local' or 's3'")

        # --- Backends ---
        if self.storage_mode == "local":
            base_root = local_root or "../../sports-data-storage-copy"
            # Normalize paths and append league segment
            self.local_data_dir = os.path.join(base_root, league, "html_tables") + "/"
            self.local_schedules_dir = os.path.join(base_root, league, "schedules") + "/"
            self.local_pbp_features_dir = os.path.join(base_root, league, "processed", "features") + "/"
            self.local_injury_dir = os.path.join(base_root, league, "injury_reports") + "/"
            self.previews_dir = os.path.join(base_root, league, "previews") + "/"
            self.starters_dir = os.path.join(base_root, league, "starters") + "/"
            self.ranks_dir = os.path.join(base_root, league, "ranks") + "/"
            self.player_ratings_dir = os.path.join(base_root, league, "player_ratings") + "/"
            self.local_officials_dir = os.path.join(base_root, league, "officials") + "/"
            self.local_game_predictions_dir = os.path.join(base_root, league, "game_predictions") + "/"
            self.s3_bucket = None
            self._s3 = None
        else:  # S3 mode
            if boto3 is None:
                raise ImportError("boto3 is required for S3 storage_mode")
            # Use prefixes analogous to local directory layout
            self.local_data_dir = f"{league}/html_tables/"
            self.local_schedules_dir = f"{league}/schedules/"
            self.local_pbp_features_dir = f"{league}/processed/features/"
            self.local_injury_dir = f"{league}/injury_reports/"
            self.previews_dir = f"{league}/previews/"
            self.starters_dir = f"{league}/starters/"
            self.ranks_dir = f"{league}/ranks/"
            self.player_ratings_dir = f"{league}/player_ratings/"
            self.local_officials_dir = f"{league}/officials/"
            self.local_game_predictions_dir = f"{league}/game_predictions/"
            self.s3_bucket = s3_bucket or os.getenv("SPORTS_DATA_BUCKET_NAME", "")
            if not self.s3_bucket:
                logging.warning("S3 bucket name not provided or env SPORTS_DATA_BUCKET_NAME missing.")
            self._s3 = s3_client or boto3.client("s3")

        # --- Unified helpers ---
        def _read_csv(path: str, **kwargs) -> pd.DataFrame:
            if self.storage_mode == "local":
                return pd.read_csv(path, **kwargs)
            # S3 path treated as key
            try:
                resp = self._s3.get_object(Bucket=self.s3_bucket, Key=path)
                content = resp["Body"].read().decode("utf-8")
                return pd.read_csv(StringIO(content), **kwargs)
            except Exception as e:
                logging.error(f"Error reading S3 CSV {path}: {e}")
                return pd.DataFrame()

        def _listdir(prefix_or_path: str) -> List[str]:
            if self.storage_mode == "local":
                if not os.path.exists(prefix_or_path):
                    return []
                return sorted(os.listdir(prefix_or_path))
            # S3 listing
            keys: List[str] = []
            continuation: Optional[str] = None
            while True:
                params = {"Bucket": self.s3_bucket, "Prefix": prefix_or_path}
                if continuation:
                    params["ContinuationToken"] = continuation
                try:
                    resp = self._s3.list_objects_v2(**params)
                except Exception as e:
                    logging.error(f"Error listing S3 prefix {prefix_or_path}: {e}")
                    break
                for obj in resp.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/"):  # skip 'directories'
                        continue
                    # Extract filename portion relative to prefix
                    filename = key[len(prefix_or_path):] if key.startswith(prefix_or_path) else key
                    if filename:
                        keys.append(filename)
                continuation = resp.get("NextContinuationToken")
                if not continuation:
                    break
            return sorted(keys)

        self._read_csv = _read_csv
        self._listdir = _listdir
        
        # Rank configurations
        self.team_rank_configs = [
            {'mode': 'season', 'split': None},
            {'mode': 'season', 'split': 'home'},
            {'mode': 'season', 'split': 'away'},
            {'mode': 'last_n', 'last_n': 3, 'split': None},
            {'mode': 'last_n', 'last_n': 5, 'split': None},
        ]

        self.player_group_rank_configs = [
            {'mode': 'season'},
            {'mode': 'last_n', 'last_n': 3},
            {'mode': 'last_n', 'last_n': 5},
        ]
        
        # String columns for ranks (non-numeric)
        self.team_ranks_str_cols = ['key', 'game_date', 'abbr', 'week', 'year']

        # big play stat columns
        self.big_play_stat_columns: List[str] = ["big_play_count_10", "big_play_count_20", "big_play_count_30", "big_play_count_40", "big_play_count_50"]
        
        # Data containers (lazy loaded)
        self._schedules: pd.DataFrame | None = None
        self._standings: pd.DataFrame | None = None
        self._previews: pd.DataFrame | None = None
        self._starters: pd.DataFrame | None = None
        self._starters_new: pd.DataFrame | None = None
        self._boxscores: pd.DataFrame | None = None
        self._player_data: pd.DataFrame | None = None
        self._player_snaps: pd.DataFrame | None = None
        self._advanced_stats: Dict[str, pd.DataFrame] = {}
        self._pbp_features: Dict[str, pd.DataFrame] = {}
        self._team_ranks: Dict[str, pd.DataFrame] = {}
        self._player_group_ranks: Dict[str, pd.DataFrame] = {}
        self._game_predictions: pd.DataFrame | None = None
        self._next_game_predictions: pd.DataFrame | None = None
        self._cache: Dict[str, pd.DataFrame] = {}  # General cache for utility methods

        # Advanced stat columns (populated during load)
        self.advanced_stat_cols: Dict[str, List[str]] = {}
        self.redzone_columns: List[str] = []
        self.team_epa_columns: List[str] = []

        # New PBP feature columns (populated during load)
        self.play_type_columns: List[str] = []  # play types per down
        self.yards_togo_columns: List[str] = []  # avg yards to go per down
        self.yards_gained_columns: List[str] = []  # avg yards gained per down
        self.big_play_position_columns: List[str] = []  # big plays by position
        self.player_epa_position_columns: List[str] = []  # player EPAs by position

        self.team_name_mappings: Dict[str, str] = {}
        self.player_key_abbr_mappings: Dict[tuple, str] = {}

        self.DIVISION_MAPPINGS = DIVISION_MAPPINGS
        self.CONFERENCE_MAPPINGS = CONFERENCE_MAPPINGS

        logging.info(f"Initialized DataObject for {league.upper()}")
    
    # ====================================================================
    # SCHEDULES
    # ====================================================================
    
    @property
    def schedules(self) -> pd.DataFrame:
        """Load and cache schedules from local directory."""
        if self._schedules is None:
            self._schedules = self._load_schedules()
        return self._schedules
    
    def _load_schedules(self) -> pd.DataFrame:
        """Load all schedule files from local directory."""
        logging.info("Loading schedules...")
        df = pd.DataFrame()
        
        if not os.path.exists(self.local_schedules_dir):
            logging.warning(f"Schedules directory not found: {self.local_schedules_dir}")
            return df
        
        for fn in self._listdir(self.local_schedules_dir):
            try:
                temp_df = self._read_csv(f"{self.local_schedules_dir}{fn}")
                df = pd.concat([df, temp_df], ignore_index=True)
            except Exception as e:
                logging.error(f"Error loading schedule file {fn}: {e}")
        
        if not df.empty:
            # Apply team abbreviation mappings
            df['abbr'] = df['abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['abbr'])
            df['opp_abbr'] = df['opp_abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['opp_abbr'])
            
            # Convert and sort by date
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                df = df.sort_values('game_date')
        
        logging.info(f"Loaded {len(df)} schedule records")
        return df
    
    # ====================================================================
    # STANDINGS
    # ====================================================================
    
    @property
    def standings(self) -> pd.DataFrame:
        """Load and cache standings data."""
        if self._standings is None:
            self._standings = self._load_standings()
        return self._standings
    
    def _load_standings(self) -> pd.DataFrame:
        """Load standings from CSV."""
        logging.info("Loading standings...")
        try:
            df = self._read_csv(f"{self.local_data_dir}standings.csv")
            if 'game_ids' in df.columns:
                df = df.drop(columns=['game_ids'])
            logging.info(f"Loaded {len(df)} standings records")
            return df
        except Exception as e:
            logging.error(f"Error loading standings: {e}")
            return pd.DataFrame()
    
    # ====================================================================
    # PREVIEWS
    # ====================================================================
    
    @property
    def previews(self) -> pd.DataFrame:
        """Load and cache preview data."""
        if self._previews is None:
            self._previews = self._load_previews()
        return self._previews
    
    def _load_previews(self) -> pd.DataFrame:
        """Load most recent preview file."""
        logging.info("Loading previews...")
        
        if not os.path.exists(self.previews_dir):
            logging.warning(f"Previews directory not found: {self.previews_dir}")
            return pd.DataFrame()
        
        fns = sorted(self._listdir(self.previews_dir), reverse=True)
        if not fns:
            logging.warning("No preview files found")
            return pd.DataFrame()
        
        fn = f"{self.previews_dir}/{fns[0]}"
        try:
            df = self._read_csv(fn)
            df = df.rename(columns={'ou': 'over_under'})
            
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
            
            # Parse weather if available
            if set(['temperature', 'humidity', 'wind']).issubset(df.columns):
                df['weather'] = df['temperature'].astype(str) + '|' + df['humidity'].astype(str) + '|' + df['wind'].astype(str)
            
            if 'game_date' in df.columns:
                df = df.sort_values('game_date')

            df['abbr'] = df['abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['abbr'])
            df['opp_abbr'] = df['opp_abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['opp_abbr'])
            
            temp_df = df.copy()[df['is_home']==1].rename(columns={'abbr': 'home_abbr', 'opp_abbr': 'away_abbr'})
            temp_df = self.add_spread_info(temp_df, is_prediction=True)

            df = df.merge(
                temp_df[['game_id', 'spread', 'home_is_favorite', 'covered_spread', 'mov_of_favorite', 'spread_favorite_abbr']],
                on='game_id',
                how='left'
            )

            logging.info(f"Loaded previews from {fn}")
            return df
        except Exception as e:
            logging.error(f"Error loading previews from {fn}: {e}")
            return pd.DataFrame()
    
    # ====================================================================
    # STARTERS
    # ====================================================================
    
    @property
    def starters(self) -> pd.DataFrame:
        """Load and cache historical starters data."""
        if self._starters is None:
            self._starters = self._load_starters()
        return self._starters
    
    def _load_starters(self) -> pd.DataFrame:
        """Load historical starters from CSV files."""
        logging.info("Loading historical starters...")
        try:
            home_df = self._read_csv(f"{self.local_data_dir}home_starters.csv")
            home_df['is_home'] = 1
            vis_df = self._read_csv(f"{self.local_data_dir}vis_starters.csv")
            vis_df['is_home'] = 0
            df = pd.concat([
                home_df,
                vis_df
            ])
            logging.info(f"Loaded {len(df)} historical starter records")
            return df
        except Exception as e:
            logging.error(f"Error loading historical starters: {e}")
            return pd.DataFrame()
    
    @property
    def starters_new(self) -> pd.DataFrame:
        """Load and cache new/upcoming starters data."""
        if self._starters_new is None:
            self._starters_new = self._load_starters_new()
        return self._starters_new
    
    def _load_starters_new(self) -> pd.DataFrame:
        """Load most recent starters file."""
        logging.info("Loading new starters...")
        
        if not os.path.exists(self.starters_dir):
            logging.warning(f"Starters directory not found: {self.starters_dir}")
            return pd.DataFrame()
        
        fns = sorted(self._listdir(self.starters_dir), reverse=True)
        if not fns:
            logging.warning("No starters files found")
            return pd.DataFrame()
        
        fn = f"{self.starters_dir}/{fns[0]}"
        try:
            df = self._read_csv(fn)
            logging.info(f"Loaded new starters from {fn}")
            return df
        except Exception as e:
            logging.error(f"Error loading new starters from {fn}: {e}")
            return pd.DataFrame()

    # ====================================================================
    # BOXSCORES
    # ====================================================================
    
    @property
    def boxscores(self) -> pd.DataFrame:
        """Load and cache boxscore data."""
        if self._boxscores is None:
            self._boxscores = self._load_boxscores()
        return self._boxscores
    
    def _load_boxscores(self) -> pd.DataFrame:
        """Load boxscore data with team mappings."""
        logging.info("Loading boxscores...")
        try:
            df = self._read_csv(f"{self.local_data_dir}boxscore.csv")
            
            # Apply team abbreviation mappings
            for col in ['home_abbr', 'away_abbr', 'winning_abbr', 'losing_abbr']:
                if col in df.columns:
                    df[col] = df[col].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df[col])
            
            # Convert and sort by date
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                df = df.sort_values('game_date')
            
            # Calculate home win
            if 'home_points' in df.columns and 'away_points' in df.columns:
                df['home_win'] = (df['home_points'] > df['away_points']).astype(int)
            
            # Add division and conference mappings
            df['home_division'] = df['home_abbr'].map(DIVISION_MAPPINGS).fillna('Unknown')
            df['home_conference'] = df['home_abbr'].map(CONFERENCE_MAPPINGS).fillna('Unknown')
            df['away_division'] = df['away_abbr'].map(DIVISION_MAPPINGS).fillna('Unknown')
            df['away_conference'] = df['away_abbr'].map(CONFERENCE_MAPPINGS).fillna('Unknown')
            
            # map team names
            self.team_name_mappings = df[['losing_abbr', 'losing_name']].drop_duplicates().set_index('losing_abbr')['losing_name'].to_dict()

            logging.info(f"Loaded {len(df)} boxscore records")
            return df
        except Exception as e:
            logging.error(f"Error loading boxscores: {e}")
            return pd.DataFrame()
    
    # ===================================================================
    # PLAYER SNAPS
    # ===================================================================

    @property
    def player_snaps(self) -> pd.DataFrame:
        """Load and cache player snap counts."""
        if self._player_snaps is None:
            self._player_snaps = self._load_player_snaps()
        return self._player_snaps
    
    def _load_player_snaps(self) -> pd.DataFrame:
        """Load player snap count data."""
        logging.info("Loading player snap counts...")
        try:
            df = pd.concat([
                self._read_csv(f"{self.local_data_dir}home_snap_counts.csv"),
                self._read_csv(f"{self.local_data_dir}vis_snap_counts.csv")
            ])
            
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                df = df.sort_values('game_date')
            
            logging.info(f"Loaded {len(df)} snap count records")
            return df
        except Exception as e:
            logging.error(f"Error loading player snaps: {e}")
            return pd.DataFrame()

    # ====================================================================
    # PLAYER DATA
    # ====================================================================
    
    @property
    def player_data(self) -> pd.DataFrame:
        """Load and cache complete player data with all merges."""
        if self._player_data is None:
            self._player_data = self._load_player_data()
        return self._player_data
    
    def _add_dfk_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add DraftKings fantasy points calculations to player data."""
        DFK_TARGETS = {
            'over_under_attempted_passes_34+': 34,
            'over_under_completed_passes_22+': 22,
            'over_under_passing_yards_250+': 250,
            'over_under_passing_touchdowns_2+': 2,
            'over_under_interceptions_thrown_1+': 1,
            'over_under_rush_attempts_16+': 16,
            'over_under_rush_yards_60+': 60,
            'over_under_rush_touchdowns_1+': 1,
            'over_under_receptions_5+': 5,
            'over_under_receiving_yards_60+': 60,
            'over_under_receiving_touchdowns_1+': 1,
            'over_under_rush_yards_&_receiving_yards_100+': 100,
            'over_under_rush_touchdowns_&_receiving_touchdowns_1+': 1,
        }

        for stat, value in DFK_TARGETS.items():
            data_stat = '_'.join(stat.split('_')[:-1]).replace('over_under_', '')  # Remove the '+' at the end
            if '&' not in stat:
                stat_cols = [data_stat]
            else:
                stat_cols = data_stat.replace('over_under_', '').split('_&_')
            df[stat] = (df[stat_cols].sum(axis=1) >= value).astype(int)
        return df
    
    def get_qb_points(self, row: pd.Series):
        points = 0
        # passing_touchdowns
        points += round(row['passing_touchdowns'], 0)*4
        # passing_yards
        points += round(row['passing_yards'], 0)*0.04
        points += 3 if row['passing_yards'] > 300 else 0
        # interceptions
        points -= round(row['interceptions_thrown'], 0)
        # rush_yards
        points += round(row['rush_yards'], 0)*0.1
        points += 3 if row['rush_yards'] > 100 else 0
        # rush_touchdowns
        points += round(row['rush_touchdowns'], 0)*6
        return round(points, 2)
    
    def get_skill_points(self, row: pd.Series):
        points = 0
        # rush_yards
        if 'rush_yards' in row.index:
            points += round(row['rush_yards'], 0)*0.1
            points += 3 if row['rush_yards'] > 100 else 0
        # rush_touchdowns
        if 'rush_touchdowns' in row.index:
            points += round(row['rush_touchdowns'], 0)*6
        # receptions
        points += round(row['receptions'], 0)
        # receiving_yards
        points += round(row['receiving_yards'], 0)*0.1
        points += 3 if row['receiving_yards'] > 100 else 0
        # receiving_touchdowns
        points += round(row['receiving_touchdowns'], 0)*6
        return round(points, 2)

    def _load_player_data(self) -> pd.DataFrame:
        """Load complete player data with all necessary merges."""
        logging.info("Loading comprehensive player data...")
        
        # Load base player stats
        try:
            df = pd.concat([
                self._read_csv(f"{self.local_data_dir}home_players.csv"),
                self._read_csv(f"{self.local_data_dir}away_players.csv")
            ])
            df['abbr'] = df['abbr'].str.upper()
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                df = df.sort_values('game_date')
            # Populate mapping early to avoid circular dependency
            if not hasattr(self, 'player_key_abbr_mappings') or not self.player_key_abbr_mappings:
                self.player_key_abbr_mappings = df[['key', 'pid', 'abbr']].drop_duplicates().set_index(['key', 'pid'])['abbr'].to_dict()
        except Exception as e:
            logging.error(f"Error loading player data: {e}")
            return pd.DataFrame()
        
        # Merge snap counts
        snaps_df = self.player_snaps
        if not snaps_df.empty:
            df = df.merge(
                snaps_df.drop(columns=['player'], errors='ignore'),
                on=['key', 'year', 'game_date', 'pid'],
                how='left'
            )
        # Merge boxscores (use basic boxscores to avoid circular dependency)
        boxscores_df = self.add_spread_info(self.boxscores).copy()
        if not boxscores_df.empty:
            df = df.merge(
                boxscores_df.drop(columns=['game_date', 'week'], errors='ignore'),
                on=['key', 'year'],
                how='left'
            )
        
        # Add is_home indicator
        if 'home_abbr' in df.columns and 'abbr' in df.columns:
            df['is_home'] = (df['abbr'] == df['home_abbr']).astype(int)
        
        # Merge PBP player EPAs
        pbp_features = self.pbp_features
        if 'player_epas' in pbp_features and not pbp_features['player_epas'].empty:
            player_epas = pbp_features['player_epas'].copy()
            if 'pid' in player_epas.columns:
                player_epas['pid'] = player_epas['pid'].str.split("|")
                player_epas = player_epas.explode('pid').drop(columns=['pos'], errors='ignore')
                player_epas = player_epas.groupby(by=['key', 'pid']).mean()
                df = df.merge(player_epas, on=['key', 'pid'], how='left')
        
        # Merge player injuries
        if 'player_injuries' in pbp_features and not pbp_features['player_injuries'].empty:
            df = df.merge(pbp_features['player_injuries'], on=['key', 'pid'], how='left')
        
        # Merge starters
        starters_df = self.starters[['key', 'pid']]
        if not starters_df.empty:
            df = df.merge(
                starters_df.assign(starter=1),
                on=['key', 'pid'],
                how='left'
            )
            df['starter'] = df['starter'].fillna(0).astype(int)
        
        # Add week column from schedules
        schedules_df = self.schedules
        if not schedules_df.empty and 'week' in schedules_df.columns:
            df = df.merge(
                schedules_df[['game_id', 'week']].drop_duplicates(),
                left_on='key',
                right_on='game_id',
                how='left'
            )
            # Add last_week for standings lookup
            weeks_df = schedules_df[['week']].drop_duplicates().copy()
            weeks_df['last_week'] = weeks_df['week'].shift(1)
            df = df.merge(weeks_df, on='week', how='left')
        
        # Merge big plays
        if 'big_plays' in pbp_features and not pbp_features['big_plays'].empty:
            df = df.merge(
                pbp_features['big_plays'].drop(columns=['pos'], errors='ignore'),
                left_on=['key', 'abbr', 'pid'],
                right_on=['key', 'possession_team', 'pid'],
                how='left'
            )
        
        # Add is_win indicator
        if 'home_points' in df.columns and 'away_points' in df.columns and 'is_home' in df.columns:
            df['is_win'] = (
                (df['home_points'] > df['away_points']) & (df['is_home'] == 1)
            ).fillna(False)
        
        # Add DraftKings fantasy targets
        df = self._add_dfk_targets(df)

        # Advanced stats
        for stat_type in ['passing', 'rushing', 'receiving']:
            adv_df = self.get_advanced_stats(stat_type)
            adv_df = adv_df.drop(columns=['game_date', 'player', 'tm'], errors='ignore').rename(columns={
                col: f"adv_{stat_type}_{col}" for col in adv_df.columns if col not in ['key', 'year', 'pid']
            })
            if not adv_df.empty:
                df = df.merge(
                    adv_df,
                    on=['key', 'year', 'pid'],
                    how='left'
                )

        # Add fantasy points calculations
        df['fantasy_points'] = df.apply(
            lambda row: self.get_qb_points(row) if row.get('pos') == 'QB' else self.get_skill_points(row),
            axis=1
        )

        df = df.sort_values('game_date')
        logging.info(f"Loaded complete player data: {len(df)} records")
        return df
    
    # ====================================================================
    # ADVANCED STATS
    # ====================================================================
    
    def get_advanced_stats(self, stat_type: str) -> pd.DataFrame:
        """Load advanced stats for a specific type (passing, rushing, receiving)."""
        if stat_type not in self._advanced_stats:
            self._advanced_stats[stat_type] = self._load_advanced_stats(stat_type)
        return self._advanced_stats[stat_type]
    
    def _load_advanced_stats(self, stat_type: str) -> pd.DataFrame:
        """Load advanced stats from CSV."""
        logging.info(f"Loading advanced {stat_type} stats...")
        try:
            # Files are named like "passing_advanced.csv", not "home_advanced_passing.csv"
            df = self._read_csv(f"{self.local_data_dir}{stat_type}_advanced.csv")
            
            # Rename columns to include stat_type prefix (except standard columns)
            exclude_cols = ['key', 'year', 'pid', 'player', 'game_date', 'tm']
            df = df.rename(columns={
                col: f"{stat_type}_{col}" 
                for col in df.columns 
                if col not in exclude_cols
            })
            
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                df = df.sort_values('game_date')
            
            # Store column names (excluding standard columns)
            self.advanced_stat_cols[stat_type] = [
                col for col in df.columns if col not in exclude_cols
            ]
            
            logging.info(f"Loaded {len(df)} advanced {stat_type} records")
            return df
        except Exception as e:
            logging.error(f"Error loading advanced {stat_type} stats: {e}")
            return pd.DataFrame()
    
    # ====================================================================
    # PLAY-BY-PLAY FEATURES
    # ====================================================================
    
    @property
    def pbp_features(self) -> Dict[str, pd.DataFrame]:
        """Load and cache all PBP features."""
        if not self._pbp_features:
            self._pbp_features = self._load_pbp_features()
        return self._pbp_features
    
    def _load_pbp_features(self) -> Dict[str, pd.DataFrame]:
        """Load all PBP feature files from directory."""
        logging.info("Loading PBP features...")
        pbp_dict = {}
        
        if not os.path.exists(self.local_pbp_features_dir):
            logging.warning(f"PBP features directory not found: {self.local_pbp_features_dir}")
            return pbp_dict
        
        pbp_string_cols = ['key', 'team', 'game_date', 'possession_team', 'pid', 'home_abbr', 'away_abbr']
        
        for fn in self._listdir(self.local_pbp_features_dir):
            if fn.endswith(".csv"):
                key = fn.replace(".csv", "")
                try:
                    df = self._read_csv(f"{self.local_pbp_features_dir}{fn}")
                    
                    # Apply team mappings
                    if 'team' in df.columns:
                        df['team'] = df['team'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['team'])
                    if 'possession_team' in df.columns:
                        df['possession_team'] = df['possession_team'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['possession_team'])
                    if 'home_abbr' in df.columns:
                        df['home_abbr'] = df['home_abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['home_abbr'])
                    if 'away_abbr' in df.columns:
                        df['away_abbr'] = df['away_abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['away_abbr'])
                    
                    # Convert dates
                    if 'game_date' in df.columns:
                        df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                        df = df.sort_values('game_date')
                    
                    # Store feature columns for specific types
                    if key == 'team_redzone_epa_and_outcomes':
                        self.redzone_columns = [
                            col for col in df.columns if col not in pbp_string_cols
                        ]
                    elif key == 'team_possession_epas':
                        self.team_epa_columns = [
                            col for col in df.columns if col not in ['key', 'home_abbr', 'away_abbr', 'game_date']
                        ]
                    elif key == 'play_types_per_down':
                        # Will be populated when game data is loaded (dynamically created columns)
                        pass
                    elif key == 'yards_togo_per_down':
                        # Will be populated when game data is loaded (dynamically created columns)
                        pass
                    elif key == 'yards_gained_per_down':
                        # Will be populated when game data is loaded (dynamically created columns)
                        pass
                    elif key == 'big_plays':
                        # Will be populated when game data is loaded (dynamically created columns)
                        pass
                    elif key == 'player_epas':
                        # Will be populated when game data is loaded (dynamically created columns)
                        pass
                    
                    pbp_dict[key] = df
                    logging.info(f"Loaded PBP feature: {key} ({len(df)} records)")
                except Exception as e:
                    logging.error(f"Error loading PBP feature {fn}: {e}")
        
        return pbp_dict
    
    # ====================================================================
    # TEAM RANKS
    # ====================================================================
    
    @property
    def team_ranks(self) -> Dict[str, pd.DataFrame]:
        """Load and cache all team ranks configurations."""
        if not self._team_ranks:
            self._team_ranks = self._load_all_team_ranks()
        return self._team_ranks
    
    def _load_all_team_ranks(self) -> Dict[str, pd.DataFrame]:
        """Load all team rank configurations."""
        logging.info("Loading all team ranks...")
        ranks_dict = {}
        
        for cfg in self.team_rank_configs:
            cfg_key = f"{cfg['mode']}_{cfg.get('split', '')}_{cfg.get('last_n', '')}".strip("_")
            ranks_dict[cfg_key] = self._load_team_ranks(**cfg)
        
        return ranks_dict
    
    def _load_team_ranks(self, mode: str = 'season', last_n: int = None, split: str = None) -> pd.DataFrame:
        """Load team ranks for specific configuration."""
        # Determine directory
        if mode == 'season' and split is None:
            local_dir = os.path.join(self.ranks_dir, "season_ranks/")
        elif mode == 'season' and split in ('home', 'away'):
            local_dir = os.path.join(self.ranks_dir, f"season_ranks_{split}/")
        elif mode == 'last_n':
            local_dir = os.path.join(self.ranks_dir, f"last_{last_n}_game_ranks/")
        else:
            raise ValueError(f"Invalid mode/split configuration: {mode}, {split}")
        
        df = pd.DataFrame()
        
        if not os.path.exists(local_dir):
            logging.warning(f"Team ranks directory not found: {local_dir}")
            return df
        
        for fn in self._listdir(local_dir):
            try:
                temp_df = self._read_csv(f"{local_dir}{fn}")
                df = pd.concat([df, temp_df], ignore_index=True)
            except Exception as e:
                logging.error(f"Error loading team ranks file {fn}: {e}")
        
        if not df.empty:
            # Apply team mappings
            if 'abbr' in df.columns:
                df['abbr'] = df['abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['abbr'])
            
            # Convert dates
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                df = df.sort_values('game_date')
        
        logging.info(f"Loaded team ranks {mode}_{split}_{last_n}: {len(df)} records")
        return df
    
    # ====================================================================
    # PLAYER GROUP RANKS
    # ====================================================================
    
    @property
    def player_group_ranks(self) -> Dict[str, pd.DataFrame]:
        """Load and cache all player group ranks configurations."""
        if not self._player_group_ranks:
            self._player_group_ranks = self._load_all_player_group_ranks()
        return self._player_group_ranks
    
    def _load_all_player_group_ranks(self) -> Dict[str, pd.DataFrame]:
        """Load all player group rank configurations."""
        logging.info("Loading all player group ranks...")
        ranks_dict = {}
        
        for cfg in self.player_group_rank_configs:
            cfg_key = f"{cfg['mode']}_{cfg.get('last_n', '')}".strip("_")
            ranks_dict[cfg_key] = self._load_player_group_ranks(**cfg)
        
        return ranks_dict
    
    def _load_player_group_ranks(self, mode: str = 'season', last_n: int = None) -> pd.DataFrame:
        """Load player group ranks for specific configuration."""
        # Determine directory
        if mode == 'season':
            local_dir = os.path.join(self.ranks_dir, "player_season_ranks/")
        elif mode == 'last_n':
            local_dir = os.path.join(self.ranks_dir, f"player_last_{last_n}_game_ranks/")
        else:
            raise ValueError(f"Invalid mode configuration: {mode}")
        
        df = pd.DataFrame()
        
        if not os.path.exists(local_dir):
            logging.warning(f"Player group ranks directory not found: {local_dir}")
            return df
        
        for fn in self._listdir(local_dir):
            try:
                temp_df = self._read_csv(f"{local_dir}{fn}")
                df = pd.concat([df, temp_df], ignore_index=True)
            except Exception as e:
                logging.error(f"Error loading player group ranks file {fn}: {e}")
        
        if not df.empty:
            # Apply team mappings
            if 'abbr' in df.columns:
                df['abbr'] = df['abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df['abbr'])
            
            # Convert dates
            if 'game_date' in df.columns:
                df['game_date'] = df['game_date'].apply(datetime.fromisoformat)
                df = df.sort_values('game_date')
        
        logging.info(f"Loaded player group ranks {mode}_{last_n}: {len(df)} records")
        return df
    
    # ====================================================================
    # GAME PREDICTIONS
    # ====================================================================
    
    @property
    def game_predictions(self) -> pd.DataFrame:
        """Load and cache past game predictions."""
        if self._game_predictions is None:
            self._game_predictions = self._load_game_predictions()
        return self._game_predictions
    
    def _load_game_predictions(self) -> pd.DataFrame:
        """Load past game predictions from CSV."""
        logging.info("Loading game predictions...")
        try:
            # Game predictions typically kept local; allow S3 access if storage_mode == 's3'
            df = self._read_csv(self.local_game_predictions_dir + "all_past_game_predictions.csv")
            logging.info(f"Loaded {len(df)} game prediction records")
            df['game_date'] = pd.to_datetime(df['game_date'])
            df = df.sort_values('game_date')
            return df
        except Exception as e:
            logging.error(f"Error loading game predictions: {e}")
            return pd.DataFrame()
    
    @property
    def next_game_predictions(self) -> pd.DataFrame:
        """Load and cache next game predictions."""
        if self._next_game_predictions is None:
            self._next_game_predictions = self._load_next_game_predictions()
        return self._next_game_predictions
    
    def _load_next_game_predictions(self) -> pd.DataFrame:
        """Load next game predictions from CSV."""
        logging.info("Loading next game predictions...")
        if not self.previews.empty:
            self.previews  # Ensure previews are loaded for team mappings
        try:
            week, year = self.previews['week'].mode()[0], self.previews['year'].mode()[0]
            game_predictions_dir = os.path.join(self.local_game_predictions_dir, f"{year}_week_{week}/")
            files = self._listdir(game_predictions_dir)
            if len(files) == 0:
                logging.warning(f"No next game prediction files found in {game_predictions_dir}")
                return pd.DataFrame()
            filename = sorted(files)[-1]  # Get the latest file
            df = self._read_csv(os.path.join(game_predictions_dir, filename))
            logging.info(f"Loaded {len(df)} next game prediction records from {filename}")
            df['game_date'] = pd.to_datetime(df['game_date'])
            df = df.sort_values('game_date')
            return df
        except Exception as e:
            logging.error(f"Error loading next game predictions: {e}")
            return pd.DataFrame()
    
    # ====================================================================
    # PLAYER RATINGS
    # ====================================================================

    @property
    def player_ratings(self) -> Dict[str, pd.DataFrame]:
        """Load and cache player ratings data."""
        if not hasattr(self, '_player_ratings'):
            self._player_ratings = self._load_player_ratings()
        return self._player_ratings

    def _load_player_ratings(self) -> Dict[str, pd.DataFrame]:
        """Load player ratings from CSV."""
        logging.info("Loading player ratings...")
        if self.boxscores.empty:
            self.boxscores  # Ensure boxscores are loaded for team mappings
        try:
            _dict = {}
            for fn in self._listdir(self.player_ratings_dir):
                if fn.endswith(".csv"):
                    temp_df = self._read_csv(f"{self.player_ratings_dir}{fn}")
                    temp_df = temp_df.merge(self.boxscores[['key', 'year']], on='key', how='left')
                    _dict[fn.split("_")[0]] = temp_df
            logging.info(f"Loaded player rating records")
            return _dict
        except Exception as e:
            logging.error(f"Error loading player ratings: {e}")
            return {}
    
    def get_team_position_ratings(self, year: int = None, positions: List[str] = None) -> pd.DataFrame:
        """
        Calculate average overall ratings for team positions based on starters.
        
        Aggregates player ratings for starters grouped by game, position, and team to produce
        team-level position ratings. This shows how strong each team's position group was
        for each game.
        
        Args:
            year: Year to filter starters data. If None, uses all years.
            positions: List of positions to include (e.g., ['QB', 'RB', 'WR']). 
                      If None, uses all available positions.
        
        Returns:
            pd.DataFrame with columns:
                - key: Game identifier
                - pos: Position group (QB, RB, WR, etc.)
                - abbr: Team abbreviation
                - avg_overall_rating: Average rating for that team's position group in that game
        
        Example:
            >>> data_obj = DataObject()
            >>> ratings = data_obj.get_team_position_ratings(year=2025, positions=['QB', 'WR'])
            >>> top_qb_ratings = ratings[ratings['pos']=='QB'].sort_values('avg_overall_rating', ascending=False)
        """
        from mp_sportsipy.nfl.constants import SIMPLE_POSITION_MAPPINGS
        
        # Load required data
        starters_df = self.starters.copy()
        boxscores_df = self.boxscores[['key', 'home_abbr', 'away_abbr']].copy()
        player_ratings_dict = self.player_ratings
        
        if starters_df.empty or not player_ratings_dict:
            logging.warning("Starters or player_ratings data is empty")
            return pd.DataFrame()
        
        # Merge boxscores once (instead of per group)
        starters_df = starters_df.merge(boxscores_df, on='key', how='left')
        
        # Map positions to simple positions once
        starters_df['pos'] = starters_df['pos'].map(SIMPLE_POSITION_MAPPINGS).fillna(starters_df['pos'])
        
        # Filter by year if specified
        if year is not None:
            starters_df = starters_df[starters_df['year'] == year]
        
        # Filter by positions if specified
        if positions is not None:
            starters_df = starters_df[starters_df['pos'].isin(positions)]
        
        # Add team abbreviation based on is_home
        starters_df['abbr'] = starters_df.apply(
            lambda row: row['home_abbr'] if row['is_home'] == 1 else row['away_abbr'], 
            axis=1
        )
        
        # Combine all ratings into single DataFrame for efficient merging
        all_ratings = []
        for pos, ratings_df in player_ratings_dict.items():
            rating_col = f'overall_{pos.lower()}_rating'
            if rating_col in ratings_df.columns:
                # Only keep necessary columns
                temp = ratings_df[['key', 'pid', rating_col]].copy()
                temp['pos'] = pos
                temp = temp.rename(columns={rating_col: 'rating'})
                all_ratings.append(temp)
        
        if not all_ratings:
            logging.warning("No rating columns found in player_ratings")
            return pd.DataFrame()
        
        # Concatenate all ratings
        combined_ratings = pd.concat(all_ratings, ignore_index=True)
        
        # Merge starters with ratings
        merged = starters_df.merge(
            combined_ratings,
            on=['key', 'pid', 'pos'],
            how='inner'
        )
        
        # Group and calculate averages using vectorized operations
        result_df = merged.groupby(['key', 'year', 'game_date', 'pos', 'abbr'], as_index=False)['rating'].mean()
        result_df = result_df.rename(columns={'rating': 'avg_overall_rating'})
        
        # Ensure game_date is datetime
        result_df['game_date'] = pd.to_datetime(result_df['game_date'])
        
        # Sort results
        result_df = result_df.sort_values(by=['key', 'game_date', 'abbr', 'pos'])
        
        logging.info(f"Calculated team position ratings: {len(result_df)} records")
        return result_df[['key', 'year', 'game_date', 'abbr', 'pos', 'avg_overall_rating']]

    # ====================================================================
    # OFFICIALS
    # ====================================================================

    @property
    def officials(self) -> pd.DataFrame:
        """Load and cache officials data."""
        if not hasattr(self, '_officials'):
            self._officials = self._load_officials()
        return self._officials
    
    def _load_officials(self) -> pd.DataFrame:
        """Load officials from CSV."""
        logging.info("Loading officials data...")
        try:
            df = self._read_csv(f"{self.local_data_dir}officials.csv")
            logging.info(f"Loaded {len(df)} official records")
            return df
        except Exception as e:
            logging.error(f"Error loading officials data: {e}")
            return pd.DataFrame()

    @property
    def new_officials(self) -> pd.DataFrame:
        """Load and cache new officials data."""
        if not hasattr(self, '_new_officials'):
            self._new_officials = self._load_new_officials()
        return self._new_officials
    
    @property
    def new_officials_with_features(self) -> pd.DataFrame:
        """Load and cache processed new officials features."""
        if not hasattr(self, '_new_officials_with_features'):
            new_officials_df = self.new_officials
            if new_officials_df.empty:
                self._new_officials_with_features = pd.DataFrame()
            else:
                self._new_officials_with_features = self.get_new_officials_with_features(new_officials_df, cache=True)
        return self._new_officials_with_features
    
    def _load_new_officials(self) -> pd.DataFrame:
        """Load new officials from CSV."""
        logging.info("Loading new officials data...")
        week, year = self.previews['week'].mode().values[0], self.previews['year'].mode().values[0]
        try:
            df = self._read_csv(f"{self.local_officials_dir}officials_{year}_week{week}.csv")
            logging.info(f"Loaded {len(df)} new official records")
            return df
        except Exception as e:
            logging.error(f"Error loading new officials data: {e}")
            return pd.DataFrame()

    # ====================================================================
    # UTILITY METHODS
    # ====================================================================
    
    def reload_all(self):
        """Force reload all cached data."""
        logging.info("Reloading all data...")
        self._schedules = None
        self._standings = None
        self._previews = None
        self._starters = None
        self._starters_new = None
        self._boxscores = None
        self._player_data = None
        self._player_snaps = None
        self._advanced_stats = {}
        self._pbp_features = {}
        self._team_ranks = {}
        self._player_group_ranks = {}
        self._game_predictions = None
        self._next_game_predictions = None
        logging.info("All data cleared from cache")

    def get_officials_with_features(self, cache: bool = True) -> pd.DataFrame:
        """
        Generate officials statistics with features for referees.
        Includes season and all-time statistics relative to league averages.
        
        Features calculated:
        - Games, playoff games, position
        - Home/visitor penalties, home win percentage
        - Total penalties, penalty yards, per-game averages
        - League averages for comparison
        - Relative stats (difference from league average)
        
        Returns:
            DataFrame with official statistics by referee_pid, year, and cumulative
        """
        cache_key = 'officials_with_features' if cache else None
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load officials, boxscore, and schedule data
        officials = self.officials.copy()
        boxscores = self.boxscores.copy()
        schedules = self.schedules.copy().rename(columns={'game_id': 'key'})
        
        if officials.empty or boxscores.empty:
            logging.warning("Officials or boxscores data is empty")
            return pd.DataFrame()
        
        # Filter to only Referees
        officials = officials[officials['type'] == 'Referee'].copy()
        
        # Ensure game_date is datetime
        if 'game_date' in officials.columns:
            officials['game_date'] = pd.to_datetime(officials['game_date'])
        if 'game_date' in boxscores.columns:
            boxscores['game_date'] = pd.to_datetime(boxscores['game_date'])
        if 'game_date' in schedules.columns:
            schedules['game_date'] = pd.to_datetime(schedules['game_date'])
        
        # Get week, game_num, is_playoffs from schedules
        # Schedules has rows for both home and away teams, so we need to get unique game info
        schedule_info = schedules[schedules['is_home'] == 1][['key', 'week', 'is_playoffs']].drop_duplicates('key')
        
        # Merge officials with boxscore data to get penalty and game outcome info
        merged = officials.drop(columns=['game_date']).merge(
            boxscores[['key', 'year', 'game_date', 'home_abbr', 'away_abbr', 
                      'home_points', 'away_points', 'home_penalties', 'away_penalties',
                      'home_yards_from_penalties', 'away_yards_from_penalties']],
            on=['key', 'year'],
            how='left'
        )

        # Merge with schedule info to get week, game_num, is_playoffs
        merged = merged.merge(
            schedule_info,
            on='key',
            how='left'
        )
        
        # Fill missing is_playoffs with 0 (regular season)
        merged['is_playoffs'] = merged['is_playoffs'].fillna(0).astype(int)
        
        # Calculate game-level stats
        merged['home_win'] = (merged['home_points'] > merged['away_points']).astype(int)
        merged['total_penalties'] = merged['home_penalties'].fillna(0) + merged['away_penalties'].fillna(0)
        merged['total_pen_yards'] = merged['home_yards_from_penalties'].fillna(0) + merged['away_yards_from_penalties'].fillna(0)
        
        # Sort by date for cumulative calculations
        merged = merged.sort_values(['pid', 'year', 'week']).reset_index(drop=True)
        
        results = []

        # Group by referee
        for pid, ref_games in merged.groupby('pid'):
            ref_name = ref_games.iloc[0]['name']
            
            # Process each game to calculate stats up to that point
            for idx, (game_idx, game) in enumerate(ref_games.iterrows()):
                year = game['year']
                week = game.get('week', idx + 1)  # Use schedule week or fallback
                game_num = game.get('game_num', idx + 1)  # Use schedule game_num or fallback
                is_playoffs = game.get('is_playoffs', 0)
                
                # Get all games in current season
                season_games = ref_games[ref_games['year'] == year]
                
                # For first game of season, use all of previous season; otherwise use current season up to this game
                if week == 1 and year > ref_games['year'].min():
                    # Use all of previous season
                    prior_games = ref_games[ref_games['year'] == (year - 1)]
                else:
                    # Use current season games before this one (by week order)
                    prior_games = season_games[
                        (season_games['week'] < week) | 
                        ((season_games['week'] == week) & (season_games['game_date'] < game['game_date']))
                    ]
                
                # All-time stats (up to but not including this game)
                all_time_games = ref_games[
                    (ref_games['year'] < year) |
                    ((ref_games['year'] == year) & (ref_games['week'] < week)) |
                    ((ref_games['year'] == year) & (ref_games['week'] == week) & (ref_games['game_date'] < game['game_date']))
                ]
                
                # Calculate league averages for this year (all games up to and including this week)
                league_games = merged[
                    (merged['year'] == year) & 
                    (merged['week'] <= week)
                ]
                
                # Season stats
                season_g = len(prior_games)
                season_g_playoffs = int(prior_games['is_playoffs'].sum()) if season_g > 0 else 0
                season_home_pen = prior_games['home_penalties'].fillna(0).sum()
                season_away_pen = prior_games['away_penalties'].fillna(0).sum()
                season_home_pct = season_home_pen / (season_home_pen + season_away_pen) * 100 if (season_home_pen + season_away_pen) > 0 else 0
                season_home_wins = prior_games['home_win'].sum()
                season_home_wpct = season_home_wins / season_g * 100 if season_g > 0 else 0
                season_total_pen = prior_games['total_penalties'].sum()
                season_total_yds = prior_games['total_pen_yards'].sum()
                season_pen_per_g = season_total_pen / season_g if season_g > 0 else 0
                season_yds_per_g = season_total_yds / season_g if season_g > 0 else 0
                
                # All-time stats
                all_g = len(all_time_games)
                all_g_playoffs = int(all_time_games['is_playoffs'].sum()) if all_g > 0 else 0
                all_home_pen = all_time_games['home_penalties'].fillna(0).sum()
                all_away_pen = all_time_games['away_penalties'].fillna(0).sum()
                all_home_pct = all_home_pen / (all_home_pen + all_away_pen) * 100 if (all_home_pen + all_away_pen) > 0 else 0
                all_home_wins = all_time_games['home_win'].sum()
                all_home_wpct = all_home_wins / all_g * 100 if all_g > 0 else 0
                all_total_pen = all_time_games['total_penalties'].sum()
                all_total_yds = all_time_games['total_pen_yards'].sum()
                all_pen_per_g = all_total_pen / all_g if all_g > 0 else 0
                all_yds_per_g = all_total_yds / all_g if all_g > 0 else 0
                
                # League averages
                lg_total_games = len(league_games)
                if lg_total_games > 0:
                    lg_home_pen = league_games['home_penalties'].fillna(0).sum()
                    lg_away_pen = league_games['away_penalties'].fillna(0).sum()
                    lg_home_pct = lg_home_pen / (lg_home_pen + lg_away_pen) * 100 if (lg_home_pen + lg_away_pen) > 0 else 0
                    lg_home_wins = league_games['home_win'].sum()
                    lg_home_wpct = lg_home_wins / lg_total_games * 100
                    lg_total_pen = league_games['total_penalties'].sum()
                    lg_total_yds = league_games['total_pen_yards'].sum()
                    lg_pen_per_g = lg_total_pen / lg_total_games
                    lg_yds_per_g = lg_total_yds / lg_total_games
                else:
                    lg_home_pct = lg_home_wpct = lg_pen_per_g = lg_yds_per_g = 0
                
                # Relative stats (season vs league)
                rel_home_pct = season_home_pct - lg_home_pct
                rel_home_wpct = season_home_wpct - lg_home_wpct
                rel_pen_per_g = season_pen_per_g - lg_pen_per_g
                rel_yds_per_g = season_yds_per_g - lg_yds_per_g
                
                # Relative stats (all-time vs league)
                all_rel_home_pct = all_home_pct - lg_home_pct
                all_rel_home_wpct = all_home_wpct - lg_home_wpct
                all_rel_pen_per_g = all_pen_per_g - lg_pen_per_g
                all_rel_yds_per_g = all_yds_per_g - lg_yds_per_g
                
                results.append({
                    'referee_pid': pid,
                    'referee_name': ref_name,
                    'game_id': game['key'],
                    'year': year,
                    'week': week,
                    'game_num': game_num,
                    'is_playoffs': is_playoffs,
                    'game_date': game['game_date'],
                    'position': 'Referee',
                    
                    # Season stats
                    'season_g': season_g,
                    'season_g_playoffs': season_g_playoffs,
                    'season_home_penalties': season_home_pen,
                    'season_away_penalties': season_away_pen,
                    'season_home_pct': season_home_pct,
                    'season_home_wpct': season_home_wpct,
                    'season_total_penalties': season_total_pen,
                    'season_total_yards': season_total_yds,
                    'season_pen_per_g': season_pen_per_g,
                    'season_yds_per_g': season_yds_per_g,
                    
                    # All-time stats
                    'all_g': all_g,
                    'all_g_playoffs': all_g_playoffs,
                    'all_home_penalties': all_home_pen,
                    'all_away_penalties': all_away_pen,
                    'all_home_pct': all_home_pct,
                    'all_home_wpct': all_home_wpct,
                    'all_total_penalties': all_total_pen,
                    'all_total_yards': all_total_yds,
                    'all_pen_per_g': all_pen_per_g,
                    'all_yds_per_g': all_yds_per_g,
                    
                    # League averages
                    'lg_home_pct': lg_home_pct,
                    'lg_home_wpct': lg_home_wpct,
                    'lg_pen_per_g': lg_pen_per_g,
                    'lg_yds_per_g': lg_yds_per_g,
                    
                    # Relative stats (season)
                    'season_rel_home_pct': rel_home_pct,
                    'season_rel_home_wpct': rel_home_wpct,
                    'season_rel_pen_per_g': rel_pen_per_g,
                    'season_rel_yds_per_g': rel_yds_per_g,
                    
                    # Relative stats (all-time)
                    'all_rel_home_pct': all_rel_home_pct,
                    'all_rel_home_wpct': all_rel_home_wpct,
                    'all_rel_pen_per_g': all_rel_pen_per_g,
                    'all_rel_yds_per_g': all_rel_yds_per_g,
                })
        
        result_df = pd.DataFrame(results).sort_values(['year', 'week', 'game_id'])
        
        if cache:
            self._cache[cache_key] = result_df
        
        logging.info(f"Generated officials features for {len(result_df)} game records")
        return result_df

    def get_new_officials_with_features(self, new_officials_df: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
        """
        Generate features for new officials (upcoming games) using their most recent historical data.
        
        Args:
            new_officials_df: DataFrame with columns [game_id, year, week, home_abbr, away_abbr, referee_name, referee_pid]
            cache: Whether to use cached historical features
            
        Returns:
            DataFrame with same structure as get_officials_with_features() but using latest available stats
        """
        if new_officials_df.empty:
            logging.warning("New officials DataFrame is empty")
            return pd.DataFrame()
        
        # Get all historical features
        historical_features = self.get_officials_with_features(cache=cache)
        
        if historical_features.empty:
            logging.warning("No historical officials features available")
            return pd.DataFrame()
        
        # Get the most recent stats for each referee
        latest_stats = historical_features.sort_values(['referee_pid', 'year', 'week']).groupby('referee_pid').last().reset_index()
        
        results = []
        
        for _, new_game in new_officials_df.iterrows():
            referee_pid = new_game['referee_pid']
            referee_name = new_game['referee_name']
            game_id = new_game['game_id']
            year = new_game['year']
            week = new_game['week']
            
            # Find this referee's latest stats
            ref_stats = latest_stats[latest_stats['referee_pid'] == referee_pid]
            
            if ref_stats.empty:
                # New referee with no history - use league averages
                logging.warning(f"No historical data for referee {referee_name} ({referee_pid})")
                
                # Calculate current season league averages
                current_season_stats = historical_features[historical_features['year'] == year]
                if not current_season_stats.empty:
                    lg_home_pct = current_season_stats['lg_home_pct'].mean()
                    lg_home_wpct = current_season_stats['lg_home_wpct'].mean()
                    lg_pen_per_g = current_season_stats['lg_pen_per_g'].mean()
                    lg_yds_per_g = current_season_stats['lg_yds_per_g'].mean()
                else:
                    # Fallback to all-time averages
                    lg_home_pct = historical_features['lg_home_pct'].mean()
                    lg_home_wpct = historical_features['lg_home_wpct'].mean()
                    lg_pen_per_g = historical_features['lg_pen_per_g'].mean()
                    lg_yds_per_g = historical_features['lg_yds_per_g'].mean()
                
                results.append({
                    'referee_pid': referee_pid,
                    'referee_name': referee_name,
                    'game_id': game_id,
                    'year': year,
                    'week': week,
                    'game_num': None,
                    'is_playoffs': 0,
                    'game_date': None,
                    'position': 'Referee',
                    
                    # Season stats - all zeros for new referee
                    'season_g': 0,
                    'season_g_playoffs': 0,
                    'season_home_penalties': 0,
                    'season_away_penalties': 0,
                    'season_home_pct': lg_home_pct,  # Use league average
                    'season_home_wpct': lg_home_wpct,
                    'season_total_penalties': 0,
                    'season_total_yards': 0,
                    'season_pen_per_g': lg_pen_per_g,
                    'season_yds_per_g': lg_yds_per_g,
                    
                    # All-time stats - all zeros
                    'all_g': 0,
                    'all_g_playoffs': 0,
                    'all_home_penalties': 0,
                    'all_away_penalties': 0,
                    'all_home_pct': lg_home_pct,
                    'all_home_wpct': lg_home_wpct,
                    'all_total_penalties': 0,
                    'all_total_yards': 0,
                    'all_pen_per_g': lg_pen_per_g,
                    'all_yds_per_g': lg_yds_per_g,
                    
                    # League averages
                    'lg_home_pct': lg_home_pct,
                    'lg_home_wpct': lg_home_wpct,
                    'lg_pen_per_g': lg_pen_per_g,
                    'lg_yds_per_g': lg_yds_per_g,
                    
                    # Relative stats - all zeros
                    'season_rel_home_pct': 0,
                    'season_rel_home_wpct': 0,
                    'season_rel_pen_per_g': 0,
                    'season_rel_yds_per_g': 0,
                    'all_rel_home_pct': 0,
                    'all_rel_home_wpct': 0,
                    'all_rel_pen_per_g': 0,
                    'all_rel_yds_per_g': 0,
                })
            else:
                # Use the most recent stats for this referee
                latest = ref_stats.iloc[0]
                
                # Calculate current year/week league averages
                current_stats = historical_features[
                    (historical_features['year'] == year) & 
                    (historical_features['week'] <= week)
                ]
                
                if not current_stats.empty:
                    lg_home_pct = current_stats['lg_home_pct'].mean()
                    lg_home_wpct = current_stats['lg_home_wpct'].mean()
                    lg_pen_per_g = current_stats['lg_pen_per_g'].mean()
                    lg_yds_per_g = current_stats['lg_yds_per_g'].mean()
                else:
                    # Use latest known league averages from the referee's last game
                    lg_home_pct = latest['lg_home_pct']
                    lg_home_wpct = latest['lg_home_wpct']
                    lg_pen_per_g = latest['lg_pen_per_g']
                    lg_yds_per_g = latest['lg_yds_per_g']
                
                # If this is a new season, reset season stats
                if year > latest['year']:
                    season_stats = {
                        'season_g': 0,
                        'season_g_playoffs': 0,
                        'season_home_penalties': 0,
                        'season_away_penalties': 0,
                        'season_home_pct': lg_home_pct,
                        'season_home_wpct': lg_home_wpct,
                        'season_total_penalties': 0,
                        'season_total_yards': 0,
                        'season_pen_per_g': lg_pen_per_g,
                        'season_yds_per_g': lg_yds_per_g,
                        'season_rel_home_pct': 0,
                        'season_rel_home_wpct': 0,
                        'season_rel_pen_per_g': 0,
                        'season_rel_yds_per_g': 0,
                    }
                else:
                    # Same season - use latest season stats
                    season_stats = {
                        'season_g': latest['season_g'],
                        'season_g_playoffs': latest['season_g_playoffs'],
                        'season_home_penalties': latest['season_home_penalties'],
                        'season_away_penalties': latest['season_away_penalties'],
                        'season_home_pct': latest['season_home_pct'],
                        'season_home_wpct': latest['season_home_wpct'],
                        'season_total_penalties': latest['season_total_penalties'],
                        'season_total_yards': latest['season_total_yards'],
                        'season_pen_per_g': latest['season_pen_per_g'],
                        'season_yds_per_g': latest['season_yds_per_g'],
                        'season_rel_home_pct': latest['season_home_pct'] - lg_home_pct,
                        'season_rel_home_wpct': latest['season_home_wpct'] - lg_home_wpct,
                        'season_rel_pen_per_g': latest['season_pen_per_g'] - lg_pen_per_g,
                        'season_rel_yds_per_g': latest['season_yds_per_g'] - lg_yds_per_g,
                    }
                
                results.append({
                    'referee_pid': referee_pid,
                    'referee_name': referee_name,
                    'game_id': game_id,
                    'year': year,
                    'week': week,
                    'game_num': None,
                    'is_playoffs': 0,
                    'game_date': None,
                    'position': 'Referee',
                    
                    # Season stats
                    **season_stats,
                    
                    # All-time stats (always use latest)
                    'all_g': latest['all_g'],
                    'all_g_playoffs': latest['all_g_playoffs'],
                    'all_home_penalties': latest['all_home_penalties'],
                    'all_away_penalties': latest['all_away_penalties'],
                    'all_home_pct': latest['all_home_pct'],
                    'all_home_wpct': latest['all_home_wpct'],
                    'all_total_penalties': latest['all_total_penalties'],
                    'all_total_yards': latest['all_total_yards'],
                    'all_pen_per_g': latest['all_pen_per_g'],
                    'all_yds_per_g': latest['all_yds_per_g'],
                    
                    # League averages
                    'lg_home_pct': lg_home_pct,
                    'lg_home_wpct': lg_home_wpct,
                    'lg_pen_per_g': lg_pen_per_g,
                    'lg_yds_per_g': lg_yds_per_g,
                    
                    # Relative stats (all-time)
                    'all_rel_home_pct': latest['all_home_pct'] - lg_home_pct,
                    'all_rel_home_wpct': latest['all_home_wpct'] - lg_home_wpct,
                    'all_rel_pen_per_g': latest['all_pen_per_g'] - lg_pen_per_g,
                    'all_rel_yds_per_g': latest['all_yds_per_g'] - lg_yds_per_g,
                })
        
        result_df = pd.DataFrame(results)
        
        logging.info(f"Generated features for {len(result_df)} new official assignments")
        return result_df
    
    def _get_spread_and_favorite(self, row: pd.Series, abbr: str) -> tuple:
        """Parse spread and favorite indicator from vegas_line string."""
        vegas_line_str = row.get('vegas_line', '')  

        if pd.isna(vegas_line_str) or not isinstance(vegas_line_str, str) or vegas_line_str.strip() == '':
            return np.nan, np.nan
        elif 'winning_name' in row.index and 'losing_name' in row.index:
            # Boxscore format: has winning/losing team names (full names like "Atlanta Falcons")
            winning_name = row['winning_name']
            losing_name = row['losing_name']
            winning_abbr = row['winning_abbr']
            losing_abbr = row['losing_abbr']
            
            # Extract spread value by removing team names
            spread_str = vegas_line_str.replace(winning_name, '').replace(losing_name, '').strip()
            spread = float(spread_str) if spread_str and spread_str != 'None' else np.nan
            
            # Determine if current team is favorite
            if winning_name in vegas_line_str:
                is_favorite = True if abbr == winning_abbr else False
            elif losing_name in vegas_line_str:
                is_favorite = True if abbr == losing_abbr else False
            else:
                is_favorite = np.nan
            return spread, is_favorite
        else:
            # Preview format: uses short team names (like "Falcons", "Bills")
            # Parse format like "Falcons -3.5" or "Bills -5.5"
            import re
            
            # Extract spread value using regex (looks for +/- followed by number)
            spread_match = re.search(r'([+-]?\d+\.?\d*)', vegas_line_str.replace('49ers', ''))
            if spread_match:
                spread = float(spread_match.group(1))
            else:
                spread = np.nan

            # Determine which team is the favorite
            # The team name appears before the spread in vegas_line
            # Get full team names from mappings and extract last word (team nickname)
            team_full_name = self.team_name_mappings.get(abbr, '')
            opp_full_name = self.team_name_mappings.get(row['away_abbr'], '')

            if team_full_name and opp_full_name:
                # Extract nickname (last word) from full name
                team_nickname = team_full_name.split()[-1] if isinstance(team_full_name, str) else ''
                opp_nickname = opp_full_name.split()[-1] if isinstance(opp_full_name, str) else ''

                if team_nickname == '':
                    raise ValueError(f"Could not extract abbr from team_full_name '{team_full_name}' in DataObject._get_spread_and_favorite")
                
                if opp_nickname == '':
                    raise ValueError(f"Could not extract abbr from opp_full_name '{opp_full_name}' in DataObject._get_spread_and_favorite")

                # Check which team name appears in vegas_line
                if team_nickname and team_nickname in vegas_line_str:
                    is_favorite = True
                elif opp_nickname and opp_nickname in vegas_line_str:
                    is_favorite = False
                else:
                    is_favorite = np.nan
            else:
                is_favorite = np.nan
            return spread, is_favorite

    def add_spread_info(self, df: pd.DataFrame, is_prediction: bool = False) -> pd.DataFrame:
        """Add spread information to a given DataFrame of games.

        Args:
            df: DataFrame containing game data.
            is_prediction: If True, handles preview data for predictions.
        """
        def func(row: pd.Series) -> pd.Series:
            home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
            try:
                spread, home_is_favorite = self._get_spread_and_favorite(row, home_abbr)
                if pd.notna(spread) and pd.notna(home_is_favorite):
                    if not is_prediction:
                        # Training mode: Calculate movement of favorite and spread coverage
                        if home_is_favorite:
                            mov_of_favorite = row['home_points'] - row['away_points']
                        else:
                            mov_of_favorite = row['away_points'] - row['home_points']
                        cover_spread = mov_of_favorite > abs(spread)
                        return pd.Series([
                            spread,
                            home_is_favorite,
                            cover_spread,
                            mov_of_favorite,
                            home_abbr if home_is_favorite else away_abbr
                        ])
                    else:
                        # Prediction mode: No movement or coverage calculation
                        return pd.Series([
                            spread,
                            home_is_favorite,
                            np.nan,  # Placeholder for covered_spread
                            np.nan,  # Placeholder for mov_of_favorite
                            home_abbr if home_is_favorite else away_abbr
                        ])
                elif row.get('vegas_line', '') == 'Pick':
                    return pd.Series([0.0, True, np.nan, np.nan, np.nan])
                else:
                    logging.error(f"Could not determine spread or favorite for {row['away_abbr']} vs {row['home_abbr']}, vegas_line='{row.get('vegas_line', '')}'")
                    return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
            except Exception as e:
                if row.get('vegas_line', '') == 'Pick':
                    return pd.Series([0.0, True, np.nan, np.nan, np.nan])
                logging.debug(f"Error parsing spread for {row['away_abbr']} vs {row['home_abbr']}, vegas_line='{row.get('vegas_line', '')}': {str(e)}")
                return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

        self.boxscores  # Ensure boxscores data is loaded for team name mappings

        # Apply the function to the DataFrame
        df[[
            'spread', 'home_is_favorite', 'covered_spread', 
            'mov_of_favorite', 'spread_favorite_abbr'
        ]] = df.apply(func, axis=1, result_type='expand')
        return df

    def add_pbp_features_to_game_data(self, df: pd.DataFrame, pbp: dict) -> pd.DataFrame:
        """Add PBP features to boxscores DataFrame."""
        logging.info("Adding PBP features to boxscores...")

        # Merge redzone EPA and outcomes
        if 'team_redzone_epa_and_outcomes' in pbp:
            redzone_df = pbp['team_redzone_epa_and_outcomes'].copy()
            redzone_df['team'] = redzone_df['team'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(redzone_df['team'])
            
            # Home team redzone
            home_redzone = df[['key', 'home_abbr']].merge(
                redzone_df,
                left_on=['key', 'home_abbr'],
                right_on=['key', 'team'],
                how='left'
            ).drop(columns=['team', 'game_date'], errors='ignore').add_prefix('home_')
            home_redzone = home_redzone.rename(columns={"home_key": "key", "home_home_abbr": "home_abbr"})
            
            # Away team redzone
            away_redzone = df[['key', 'away_abbr']].merge(
                redzone_df,
                left_on=['key', 'away_abbr'],
                right_on=['key', 'team'],
                how='left'
            ).drop(columns=['team', 'game_date'], errors='ignore').add_prefix('away_')
            away_redzone = away_redzone.rename(columns={"away_key": "key", "away_away_abbr": "away_abbr"})
            
            redzone_data = home_redzone.merge(away_redzone, on='key', how='left')
            df = df.merge(redzone_data, on=['key', 'home_abbr', 'away_abbr'], how='left')
        
        # Merge team EPAs
        if 'team_possession_epas' in pbp:
            team_epas = pbp['team_possession_epas'].copy()
            if not team_epas.empty:
                team_epas = team_epas.drop(columns=['game_date'], errors='ignore')
                team_epas['home_abbr'] = team_epas['home_abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(team_epas['home_abbr'])
                team_epas['away_abbr'] = team_epas['away_abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(team_epas['away_abbr'])
                df = df.merge(team_epas, on=['key', 'home_abbr', 'away_abbr'], how='left')

        # play types per down
        if 'play_types_per_down' in pbp:
            # cols: key,possession_team,down,play_type,count
            play_types = pbp['play_types_per_down'].copy()
            if not play_types.empty:
                play_types['possession_team'] = play_types['possession_team'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(play_types['possession_team'])
                for down in [1, 2, 3, 4]:
                    down_df = play_types[play_types['down'] == down][['key', 'possession_team', 'play_type', 'count']]
                    down_pivot = down_df.pivot_table(index=['key', 'possession_team'], columns='play_type', values='count', fill_value=0).reset_index()
                    down_pivot = down_pivot.add_prefix(f'down{down}_')
                    down_pivot = down_pivot.rename(columns={f'down{down}_key': 'key', f'down{down}_possession_team': 'possession_team'})
                    
                    # Merge for home team
                    home_merge = df[['key', 'home_abbr']].merge(
                        down_pivot,
                        left_on=['key', 'home_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix('home_')
                    home_merge = home_merge.rename(columns={"home_key": "key", "home_home_abbr": "home_abbr"})
                    
                    # Merge for away team
                    away_merge = df[['key', 'away_abbr']].merge(
                        down_pivot,
                        left_on=['key', 'away_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix('away_')
                    away_merge = away_merge.rename(columns={"away_key": "key", "away_away_abbr": "away_abbr"})
                    
                    play_type_data = home_merge.merge(away_merge, on='key', how='left')
                    df = df.merge(play_type_data, on=['key', 'home_abbr', 'away_abbr'], how='left')

        # yards togo per down
        if 'yards_togo_per_down' in pbp:
            # cols: key,possession_team,down,avg_yards
            yards_to_go: pd.DataFrame = pbp['yards_togo_per_down'].copy()
            if not yards_to_go.empty:
                yards_to_go['possession_team'] = yards_to_go['possession_team'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(yards_to_go['possession_team'])
                for down in [1, 2, 3, 4]:
                    down_df = yards_to_go[yards_to_go['down'] == down][['key', 'possession_team', 'avg_yards']]
                    down_df = down_df.rename(columns={'avg_yards': f'avg_yards_togo_per_down{down}'})
                    
                    # Merge for home team
                    home_merge = df[['key', 'home_abbr']].merge(
                        down_df,
                        left_on=['key', 'home_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix('home_')
                    home_merge = home_merge.rename(columns={"home_key": "key", "home_home_abbr": "home_abbr"})
                    
                    # Merge for away team
                    away_merge = df[['key', 'away_abbr']].merge(
                        down_df,
                        left_on=['key', 'away_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix('away_')
                    away_merge = away_merge.rename(columns={"away_key": "key", "away_away_abbr": "away_abbr"})
                    
                    yards_data = home_merge.merge(away_merge, on='key', how='left')
                    df = df.merge(yards_data, on=['key', 'home_abbr', 'away_abbr'], how='left')

        # yards gained per down
        if 'yards_gained_per_down' in pbp:
            # cols: key,possession_team,down,avg_yards
            yards_gained: pd.DataFrame = pbp['yards_gained_per_down'].copy()
            if not yards_gained.empty:
                yards_gained['possession_team'] = yards_gained['possession_team'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(yards_gained['possession_team'])
                for down in [1, 2, 3, 4]:
                    down_df = yards_gained[yards_gained['down'] == down][['key', 'possession_team', 'avg_yards']]
                    down_df = down_df.rename(columns={'avg_yards': f'avg_yards_gained_per_down{down}'})
                    
                    # Merge for home team
                    home_merge = df[['key', 'home_abbr']].merge(
                        down_df,
                        left_on=['key', 'home_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix('home_')
                    home_merge = home_merge.rename(columns={"home_key": "key", "home_home_abbr": "home_abbr"})
                    
                    # Merge for away team
                    away_merge = df[['key', 'away_abbr']].merge(
                        down_df,
                        left_on=['key', 'away_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix('away_')
                    away_merge = away_merge.rename(columns={"away_key": "key", "away_away_abbr": "away_abbr"})
                    
                    yards_data = home_merge.merge(away_merge, on='key', how='left')
                    df = df.merge(yards_data, on=['key', 'home_abbr', 'away_abbr'], how='left')

        # big plays per team position
        if 'big_plays' in pbp:
            # cols: key, possession_team, pid, pos, big_play_count_10,big_play_count_20,big_play_count_30,big_play_count_40,big_play_count_50
            big_plays: pd.DataFrame = pbp['big_plays'].copy()
            if not big_plays.empty:
                big_plays['possession_team'] = big_plays['possession_team'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(big_plays['possession_team'])
                
                big_plays = big_plays.drop(columns=['pid'], errors='ignore')
                big_plays['pos'] = big_plays['pos'].map(SIMPLE_POSITION_MAPPINGS)
                big_plays = big_plays.dropna(subset=['pos'])

                # Aggregate by team and position
                big_plays = big_plays.groupby(['key', 'possession_team', 'pos'], as_index=False).sum()

                for pos in big_plays['pos'].unique():
                    pos_df = big_plays[big_plays['pos'] == pos].reset_index(drop=True).drop(columns=['pos'], errors='ignore')
                    
                    # Merge for home team
                    home_merge = df[['key', 'home_abbr']].merge(
                        pos_df,
                        left_on=['key', 'home_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix(f'home_{pos}_')
                    home_merge = home_merge.rename(columns={f"home_{pos}_key": "key", f"home_{pos}_home_abbr": "home_abbr"})
                    
                    # Merge for away team
                    away_merge = df[['key', 'away_abbr']].merge(
                        pos_df,
                        left_on=['key', 'away_abbr'],
                        right_on=['key', 'possession_team'],
                        how='left'
                    ).drop(columns=['possession_team'], errors='ignore').add_prefix(f'away_{pos}_')
                    away_merge = away_merge.rename(columns={f"away_{pos}_key": "key", f"away_{pos}_away_abbr": "away_abbr"})

                    big_play_data = home_merge.merge(away_merge, on='key', how='left')
                    df = df.merge(big_play_data, on=['key', 'home_abbr', 'away_abbr'], how='left')

        # player epas by position
        if 'player_epas' in pbp:
            # cols: key,pid,epa,epa_added,pos
            player_epas: pd.DataFrame = pbp['player_epas'].copy()
            if not player_epas.empty:
                # Ensure player_key_abbr_mappings is populated before using it
                if not self.player_key_abbr_mappings:
                    try:
                        temp_df = pd.concat([
                            self._read_csv(f"{self.local_data_dir}home_players.csv"),
                            self._read_csv(f"{self.local_data_dir}away_players.csv")
                        ])
                        temp_df['abbr'] = temp_df['abbr'].str.upper()
                        self.player_key_abbr_mappings = temp_df[['key', 'pid', 'abbr']].drop_duplicates().set_index(['key', 'pid'])['abbr'].to_dict()
                    except Exception as e:
                        logging.error(f"Error loading player mappings: {e}")
                
                # add abbr for pid
                player_epas['abbr'] = player_epas.set_index(['key', 'pid']).index.map(self.player_key_abbr_mappings)
                player_epas['abbr'] = player_epas['abbr'].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(player_epas['abbr'])

                player_epas = player_epas.drop(columns=['pid'], errors='ignore')

                player_epas['pos'] = player_epas['pos'].map(SIMPLE_POSITION_MAPPINGS)
                player_epas = player_epas.dropna(subset=['pos'])
                
                player_epas = player_epas.groupby(['key', 'abbr', 'pos'], as_index=False).sum()

                for pos in player_epas['pos'].unique():
                    pos_df = player_epas[player_epas['pos'] == pos].reset_index(drop=True).drop(columns=['pos'], errors='ignore')
                    
                    # Merge for home team
                    home_merge = df[['key', 'home_abbr']].merge(
                        pos_df,
                        left_on=['key', 'home_abbr'],
                        right_on=['key', 'abbr'],
                        how='left'
                    ).drop(columns=['abbr'], errors='ignore').add_prefix(f'home_{pos}_')
                    home_merge = home_merge.rename(columns={f"home_{pos}_key": "key", f"home_{pos}_home_abbr": "home_abbr"})

                    # Merge for away team
                    away_merge = df[['key', 'away_abbr']].merge(
                        pos_df,
                        left_on=['key', 'away_abbr'],
                        right_on=['key', 'abbr'],
                        how='left'
                    ).drop(columns=['abbr'], errors='ignore').add_prefix(f'away_{pos}_')
                    away_merge = away_merge.rename(columns={f"away_{pos}_key": "key", f"away_{pos}_away_abbr": "away_abbr"})

                    epa_data = home_merge.merge(away_merge, on='key', how='left')
                    df = df.merge(epa_data, on=['key', 'home_abbr', 'away_abbr'], how='left')
        
        # Populate column lists after merging all features
        # Play types per down columns
        self.play_type_columns = [col for col in df.columns if col.startswith(('home_down', 'away_down')) and any(pt in col for pt in ['pass', 'run', 'punt', 'field_goal'])]
        
        # Yards to go per down columns
        self.yards_togo_columns = [col for col in df.columns if 'avg_yards_togo_per_down' in col]
        
        # Yards gained per down columns
        self.yards_gained_columns = [col for col in df.columns if 'avg_yards_gained_per_down' in col]
        
        # Big plays by position columns
        self.big_play_position_columns = [col for col in df.columns if any(f'_{pos}_big_play_count_' in col for pos in ['QB', 'RB', 'WR', 'TE'])]
        
        # Player EPA by position columns
        self.player_epa_position_columns = [col for col in df.columns if any(f'_{pos}_epa' in col for pos in ['QB', 'RB', 'WR', 'TE'])]

        return df

    def get_game_data_with_features(self) -> pd.DataFrame:
        """
        Get game-level data with all merged features for GamePredictorNFL.
        Includes boxscores + schedules + previews + PBP features.
        """
        logging.info("Building game data with features...")
        df = self.boxscores.copy()
        
        if df.empty:
            return df
        
        # Merge schedules for week info
        if not self.schedules.empty:
            df = df.merge(
                self.schedules[['game_id', 'week']].drop_duplicates(),
                left_on='key',
                right_on='game_id',
                how='left'
            )
            # Add last_week for standings lookup
            weeks_df = self.schedules[['week']].drop_duplicates().copy()
            weeks_df['last_week'] = weeks_df['week'].shift(1)
            df = df.merge(weeks_df, on='week', how='left')
        

        # PBP features
        pbp = self.pbp_features
        df = self.add_pbp_features_to_game_data(df, pbp)

        # spread features
        df = self.add_spread_info(df)

        # map abbr columns
        for col in df.columns:
            if 'abbr' in col:
                df[col] = df[col].map(TEAM_TO_PLAYER_ABBR_MAPPINGS).fillna(df[col])
        
        logging.info(f"Built game data with features: {len(df)} records")
        return df
    
    def get_game_data_with_features_and_predictions(self) -> pd.DataFrame:
        """
        Get game-level data with all merged features and model predictions for GamePredictorNFL.
        Includes boxscores + schedules + previews + PBP features + predictions.
        """
        logging.info("Building game data with features and predictions...")
        df = self.get_game_data_with_features()
        
        if df.empty:
            return df
        
        # Merge game predictions
        predictions = self.game_predictions.copy()
        if not predictions.empty:
            df = df.merge(
                predictions.drop(columns=['year', 'home_abbr', 'away_abbr', 'game_date'], errors='ignore').rename(columns={'game_id': 'key'}),
                on='key',
                how='left'
            )
        logging.info(f"Built game data with features and predictions: {len(df)} records")
        return df

    # ====================================================================
    # GENERIC FILE ACCESS (LOCAL OR S3)
    # ====================================================================

    def get_file(self, path: str, file_type: Optional[str] = None):
        """Fetch a file from local filesystem or S3 depending on storage_mode.

        Args:
            path: Local path or S3 key. For local mode, absolute or relative paths
                  work. For S3 mode, pass the key (e.g. f"{self.league}/html_tables/file.csv").
            file_type: Override automatic type detection ("csv", "json", "text").

        Returns:
            If CSV -> pandas.DataFrame
            If JSON -> parsed Python object
            Else -> str (decoded text) or None on error/missing.
        """
        # Infer file type from extension if not provided
        if file_type is None:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                file_type = 'csv'
            elif ext == '.json':
                file_type = 'json'
            else:
                file_type = 'text'

        if self.storage_mode == 'local':
            # Allow relative paths anchored at current working directory
            if not os.path.exists(path):
                # Support concatenating with root dirs when caller passes a directory alias
                candidate = path
                if not os.path.exists(candidate):
                    logging.debug(f"Local file not found: {path}")
                    return None
            try:
                if file_type == 'csv':
                    return pd.read_csv(path)
                elif file_type == 'json':
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
            except Exception as e:
                logging.error(f"Error reading local file {path}: {e}")
                return None
        else:  # S3 mode
            if not self._s3 or not self.s3_bucket:
                logging.error("S3 client or bucket not configured")
                return None
            try:
                resp = self._s3.get_object(Bucket=self.s3_bucket, Key=path)
                body = resp['Body'].read().decode('utf-8')
                if file_type == 'csv':
                    return pd.read_csv(StringIO(body))
                elif file_type == 'json':
                    return json.loads(body)
                else:
                    return body
            except Exception as e:
                logging.error(f"Error reading S3 file {path}: {e}")
                return None

    def list_files(self, directory_or_prefix: str) -> List[str]:
        """List files in a local directory or S3 prefix (no recursion)."""
        return self._listdir(directory_or_prefix)

if __name__ == "__main__":
    # Example usage
    data_obj = DataObject(
        league='nfl',
        storage_mode='local',
        local_root=os.path.join(sys.path[0], "..", "..", "..", 'sports-data-storage-copy')
    )
    df = data_obj.previews
    print(df)
    # position, target = 'QB', 'passing_yards'
    # pos_df = df[df['pos'] == position]
    # total_players = len(pos_df['pid'].unique())
    
    # logging.info(f"Training {position}_{target} with {total_players} players")