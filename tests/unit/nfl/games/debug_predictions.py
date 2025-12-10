"""Debug script to analyze why predictions are off."""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, PROJECT_ROOT)

from winsight_predictions.nfl.data_object import DataObject
from winsight_predictions.nfl.games.predictor import GamePredictor

def analyze_prediction_issue():
    """Analyze why predictions are so far off from means."""
    
    print("="*80)
    print("PREDICTION DEBUG ANALYSIS")
    print("="*80)
    
    # Load data
    data_obj = DataObject()
    game_data = data_obj.get_game_data_with_features()
    
    print(f"\n1. Game Data Stats:")
    print(f"   Total games: {len(game_data)}")
    
    targets = ['home_points', 'away_points', 'home_pass_yards', 'home_rush_yards', 
               'home_pass_attempts', 'home_rush_attempts']
    
    print(f"\n2. Target Variable Statistics (from game_data):")
    for target in targets:
        if target in game_data.columns:
            mean_val = game_data[target].mean()
            std_val = game_data[target].std()
            min_val = game_data[target].min()
            max_val = game_data[target].max()
            print(f"   {target:25s}: mean={mean_val:7.2f} std={std_val:6.2f} min={min_val:6.2f} max={max_val:6.2f}")
    
    # Load predictor
    predictor = GamePredictor(
        data_obj=data_obj,
        models={},
        use_saved_scalers=True,
    )
    
    # Load models
    predictor.load_models_from_dir(targets)
    
    print(f"\n3. Loaded Models:")
    for target in targets:
        if target in predictor.models:
            print(f"   ✓ {target}")
        else:
            print(f"   ✗ {target} - NOT LOADED")
    
    print(f"\n4. Feature Scaler Info:")
    for target, scaler in predictor.scalers.items():
        if scaler is not None:
            if hasattr(scaler, 'center_'):
                print(f"   {target}:")
                print(f"      Center (median): {scaler.center_[:5]} ... (first 5)")
                print(f"      Scale (IQR): {scaler.scale_[:5]} ... (first 5)")
    
    # Get a recent game for prediction
    recent_game = game_data.iloc[-1].copy()
    
    print(f"\n5. Test Prediction on Recent Game:")
    print(f"   Game: {recent_game.get('game_id')} - {recent_game['home_abbr']} vs {recent_game['away_abbr']}")
    print(f"   Date: {recent_game['game_date']}")
    
    # Make prediction
    predictions = predictor.predict_game(
        game_data=game_data.iloc[:-1],  # Use all but last game for history
        upcoming_row=recent_game,
        target_subset=targets,
    )
    
    print(f"\n6. Predictions vs Actual:")
    for target in targets:
        if target in predictions:
            pred = predictions[target]
            actual = recent_game.get(target, np.nan)
            error = pred - actual if not np.isnan(actual) else np.nan
            pct_error = (error / actual * 100) if not np.isnan(actual) and actual != 0 else np.nan
            print(f"   {target:25s}: pred={pred:7.2f} actual={actual:7.2f} error={error:+7.2f} ({pct_error:+6.1f}%)")
    
    print(f"\n7. Checking for Feature Scaling Issues:")
    # Build features for the prediction
    from winsight_predictions.nfl.games.features import FeatureEngine
    
    fe = FeatureEngine(
        game_data=game_data.iloc[:-1],
        target_name='home_points',
        row=recent_game,
        predicted_features=None,
        **predictor._fe_common_params
    )
    
    features = fe.features
    print(f"   Total features generated: {len(features)}")
    
    # Check how many are NaN or zero
    feat_series = pd.Series(features)
    nan_count = feat_series.isna().sum()
    zero_count = (feat_series == 0.0).sum()
    
    print(f"   NaN features: {nan_count} ({nan_count/len(features)*100:.1f}%)")
    print(f"   Zero features: {zero_count} ({zero_count/len(features)*100:.1f}%)")
    
    # Check feature value ranges
    print(f"\n8. Feature Value Distribution:")
    print(f"   Min: {feat_series.min():.4f}")
    print(f"   25%: {feat_series.quantile(0.25):.4f}")
    print(f"   50%: {feat_series.median():.4f}")
    print(f"   75%: {feat_series.quantile(0.75):.4f}")
    print(f"   Max: {feat_series.max():.4f}")
    
    print(f"\n9. Checking Missing Feature Handling:")
    target = 'home_points'
    if target in predictor.model_feature_names:
        needed_features = predictor.model_feature_names[target]
        generated_features = list(features.keys())
        missing = [f for f in needed_features if f not in generated_features]
        
        print(f"   Model expects: {len(needed_features)} features")
        print(f"   Generated: {len(generated_features)} features")
        print(f"   Missing (filled with 0.0): {len(missing)} features")
        
        if len(missing) > 0:
            print(f"   ⚠️ WARNING: {len(missing)} features filled with 0.0!")
            print(f"   Sample missing features: {missing[:10]}")

if __name__ == '__main__':
    analyze_prediction_issue()
