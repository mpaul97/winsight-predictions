from winsight_predictions.nfl.players.predictor import PlayerPredictor
from winsight_predictions.nfl.data_object import DataObject
import os
import sys
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Example usage
    data_obj = DataObject(
        league='nfl',
        storage_mode='s3',
        s3_bucket=os.getenv("LEAGUE_PREDICTIONS_BUCKET_NAME")
    )
    
    predictor = PlayerPredictor(data_obj=data_obj)
    
    # Example 1: Predict for a single player (all targets)
    # predictor.predict_single_player(pid='LoveJo03', position='QB', show=True)
    
    # Example 2: Predict for all starters
    predictor.predict_next_players()
    
    # Example 3: Predict for specific positions and save feature groupings (like trainer)
    # predictor.predict_next_players(positions=['QB', 'RB'], save_results=True, save_features=True)