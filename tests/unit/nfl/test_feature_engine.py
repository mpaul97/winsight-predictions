import pytest
import pandas as pd
from winsight_predictions.nfl.players.features import FeatureEngine

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    prior_games = pd.DataFrame({
        'game_date': pd.to_datetime(['2025-12-01', '2025-11-24']),
        'abbr': ['SEA', 'SEA'],
        'is_home': [1, 0],
        'off_pct': [0.75, 0.65],
        'spread': ['Pick', -3.5],
        'home_is_favorite': [None, 1],
    })
    row = pd.Series({
        'game_date': pd.Timestamp('2025-12-07'),
        'abbr': 'SEA',
        'is_home': 1,
        'spread': 'Pick',
        'home_is_favorite': None,
    })
    return prior_games, row

def test_spread_and_favorite_feature(sample_data):
    """Test spread and favorite feature computation."""
    prior_games, row = sample_data
    feature_engine = FeatureEngine(prior_games=prior_games, row=row, target_name='off_pct', position='QB')

    # Compute spread and favorite feature
    spread_feature = feature_engine.spread_and_favorite_feature

    # Assertions
    assert spread_feature['spread'] == 0.0, "Spread should be set to 0 for 'Pick'"
    assert spread_feature['is_favorite'] == 1, "Home team should be marked as favorite for 'Pick'"

def test_feature_alignment(sample_data):
    """Test alignment of training and prediction features."""
    prior_games, row = sample_data
    feature_engine = FeatureEngine(prior_games=prior_games, row=row, target_name='off_pct', position='QB')

    # Load features for training and prediction
    training_features = feature_engine.load_features()
    prediction_features = feature_engine.load_features_lightweight()

    # Assertions
    assert set(training_features.keys()) == set(prediction_features.keys()), "Feature keys should align between training and prediction"

if __name__ == "__main__":
    pytest.main(["-v", __file__])