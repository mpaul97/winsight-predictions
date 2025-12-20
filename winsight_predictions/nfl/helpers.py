import boto3
import awswrangler as wr
import pandas as pd
from rapidfuzz import process
import logging
from tqdm import tqdm

s3 = boto3.client('s3')

def upload_file_to_s3(file_path: str, bucket_name: str, s3_key: str):
    """Uploads a file to an S3 bucket."""
    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")

def get_file_from_s3(bucket_name: str, s3_key: str, download_path: str):
    """Downloads a file from an S3 bucket."""
    try:
        s3.download_file(bucket_name, s3_key, download_path)
        print(f"Successfully downloaded s3://{bucket_name}/{s3_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def get_dynamo_table_dataframe(table_name: str):
    return wr.dynamodb.read_partiql_query(
        query=f"""
            SELECT * 
            FROM {table_name}
        """
    )

def best_match(name: str, PLAYERS: list = None, player_snaps: pd.DataFrame = None):
    try:
        match, score, idx = process.extractOne(name, PLAYERS, score_cutoff=80)  # tweak cutoff
        return player_snaps.loc[player_snaps['player']==match, 'pid'].values[0]
    except:
        logging.error(f"Could not find pid for player_name: {name}")
        return None
        
def add_player_ids_nfl(df: pd.DataFrame, PLAYERS: list = None, player_snaps: pd.DataFrame = None):
    tqdm.pandas(desc="Matching player IDs")

    # Get unique player names
    temp_df = df[['player_name']].drop_duplicates().copy()
    temp_df['player_id'] = temp_df.copy()['player_name'].progress_apply(lambda x: best_match(x, PLAYERS, player_snaps))
    name_to_pid = dict(zip(temp_df['player_name'], temp_df['player_id']))

    # map player names to player IDs
    df['player_id'] = df['player_name'].map(name_to_pid)
    return df