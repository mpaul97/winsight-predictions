from winsight_predictions.nfl.games.predictor import GamePredictor
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
        s3_bucket=os.getenv("SPORTS_DATA_BUCKET_NAME")
    )
    
    run_dir = "./test_run"

    game_predictor = GamePredictor(
        data_obj=data_obj,
        root_dir=run_dir,
        predictions_bucket_name="LEAGUE_PREDICTIONS_BUCKET_NAME"
    )

    game_predictor.update_all_past_predictions(save_to_file=True, upload_to_s3=True)
    game_predictor.predict_all_next_games(save_to_file=True, upload_to_s3=True)

    predictor = PlayerPredictor(
        data_obj=data_obj,
        root_dir=run_dir,
        predictions_bucket_name="LEAGUE_PREDICTIONS_BUCKET_NAME"
    )

    predictor.predict_next_players()