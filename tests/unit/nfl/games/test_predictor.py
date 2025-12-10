"""Unit tests for NFL games GamePredictor and GameModelTrainer.

Tests verify correct prediction ordering, dependency handling, model training,
and integration with FeatureEngine.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# Ensure project root is on path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from winsight_predictions.nfl.games.predictor import (
    GamePredictor,
    GameModelTrainer,
    MeanRegressor,
    default_model_factory,
)
from winsight_predictions.nfl.games.features import FeatureEngine


class MockDataObject:
    """Mock DataObject for testing."""
    
    def __init__(self):
        self.league = 'nfl'
        self.storage_mode = 'local'
        self._game_data = self._create_game_data()
        self._team_ranks = self._create_team_ranks()
        self._standings = self._create_standings()
        self._position_ratings = self._create_position_ratings()
    
    def _create_game_data(self, num_games=30):
        """Create synthetic game data."""
        base_date = datetime(2024, 9, 1)
        rows = []
        teams = ['DAL', 'PHI', 'NYG', 'WAS', 'SF', 'SEA', 'LA', 'ARI']
        
        for i in range(num_games):
            home_team = teams[i % len(teams)]
            away_team = teams[(i + 1) % len(teams)]
            
            rows.append({
                'game_date': base_date + timedelta(days=i * 7),
                'game_id': f'202409{i:02d}0{home_team.lower()}',
                'season': 2024,
                'week': (i // 4) + 1,
                'home_abbr': home_team,
                'away_abbr': away_team,
                'home_division': 'NFC East' if home_team in ['DAL', 'PHI', 'NYG', 'WAS'] else 'NFC West',
                'away_division': 'NFC East' if away_team in ['DAL', 'PHI', 'NYG', 'WAS'] else 'NFC West',
                'home_conference': 'NFC',
                'away_conference': 'NFC',
                'home_points': 24 + i % 10,
                'away_points': 20 + i % 8,
                'home_pass_attempts': 30 + i % 5,
                'away_pass_attempts': 28 + i % 4,
                'home_rush_attempts': 25 + i % 3,
                'away_rush_attempts': 22 + i % 4,
                'home_pass_yards': 250 + i * 10,
                'away_pass_yards': 230 + i * 8,
                'home_rush_yards': 120 + i * 5,
                'away_rush_yards': 110 + i * 4,
                'home_total_yards': 370 + i * 15,
                'away_total_yards': 340 + i * 12,
                'home_win': 1 if i % 2 == 0 else 0,
                'spread': -3.5 + (i % 7) - 3,
                'ou': 44.5 + i % 10,
                'weather': f'{70 + i % 20}|{50 + i % 30}|{5 + i % 10}',
                'home_rest_days': 7,
                'away_rest_days': 7,
            })
        
        return pd.DataFrame(rows)
    
    def _create_team_ranks(self):
        """Create synthetic team rankings."""
        teams = ['DAL', 'PHI', 'NYG', 'WAS', 'SF', 'SEA', 'LA', 'ARI']
        return {
            'offense': pd.DataFrame([
                {'team_abbr': team, 'rank': i + 1, 'points_per_game': 25 - i}
                for i, team in enumerate(teams)
            ]),
            'defense': pd.DataFrame([
                {'team_abbr': team, 'rank': i + 1, 'points_allowed_per_game': 20 - i}
                for i, team in enumerate(teams)
            ])
        }
    
    def _create_standings(self):
        """Create synthetic standings."""
        teams = ['DAL', 'PHI', 'NYG', 'WAS', 'SF', 'SEA', 'LA', 'ARI']
        return pd.DataFrame([
            {
                'team_abbr': team,
                'wins': 8 - i,
                'losses': i,
                'win_pct': (8 - i) / 8,
                'division_rank': (i % 4) + 1,
            }
            for i, team in enumerate(teams)
        ])
    
    def _create_position_ratings(self):
        """Create synthetic position ratings."""
        teams = ['DAL', 'PHI', 'NYG', 'WAS', 'SF', 'SEA', 'LA', 'ARI']
        positions = ['QB', 'RB', 'WR', 'TE']
        
        rows = []
        for team in teams:
            for position in positions:
                rows.append({
                    'team_abbr': team,
                    'position': position,
                    'rating': 85.0 + np.random.uniform(-10, 10),
                })
        
        return pd.DataFrame(rows)
    
    @property
    def game_data(self):
        return self._game_data
    
    @property
    def schedules(self):
        return self._game_data[['game_id', 'game_date', 'home_abbr', 'away_abbr', 'season', 'week']]
    
    @property
    def team_ranks(self):
        return self._team_ranks
    
    @property
    def team_ranks_str_cols(self):
        return ['team_abbr']
    
    @property
    def standings(self):
        return self._standings
    
    @property
    def team_position_ratings(self):
        return self._position_ratings
    
    @property
    def available_positions(self):
        return ['QB', 'RB', 'WR', 'TE']
    
    @property
    def new_officials_with_features(self):
        # Return empty DataFrame as we don't test officials here
        return pd.DataFrame()
    
    def get_team_position_ratings(self):
        return self._position_ratings


# ========== MeanRegressor Tests ==========

def test_mean_regressor_fits_and_predicts():
    """Test MeanRegressor baseline model."""
    model = MeanRegressor()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([10, 20, 30])
    
    model.fit(X, y)
    
    assert model.mean_ == 20.0
    
    predictions = model.predict(X)
    assert len(predictions) == 3
    assert all(p == 20.0 for p in predictions)


def test_mean_regressor_empty_data():
    """Test MeanRegressor with empty data."""
    model = MeanRegressor()
    model.fit(np.array([]), np.array([]))
    
    assert model.mean_ == 0.0


# ========== GamePredictor Tests ==========

def test_game_predictor_initialization():
    """Test GamePredictor initializes correctly."""
    data_obj = MockDataObject()
    models = {
        'home_pass_attempts': MeanRegressor(),
        'away_pass_attempts': MeanRegressor(),
    }
    
    predictor = GamePredictor(data_obj=data_obj, models=models)
    
    assert predictor.data_obj == data_obj
    assert len(predictor.models) == 2
    assert predictor.min_games == 3


def test_game_predictor_get_prediction_order():
    """Test that prediction order respects dependencies."""
    data_obj = MockDataObject()
    models = {
        'home_pass_attempts': MeanRegressor(),
        'away_pass_attempts': MeanRegressor(),
        'home_rush_attempts': MeanRegressor(),
        'away_rush_attempts': MeanRegressor(),
        'home_win': MeanRegressor(),
        'home_points': MeanRegressor(),
        'away_points': MeanRegressor(),
        'home_pass_yards': MeanRegressor(),
    }
    
    predictor = GamePredictor(data_obj=data_obj, models=models)
    order = predictor._get_prediction_order()
    
    # Base volume stats should come before efficiency stats
    base_targets = [t for t in order if t in predictor.base_volume_stats]
    efficiency_targets = [t for t in order if t in predictor.efficiency_stats]
    
    if base_targets and efficiency_targets:
        # Find indices
        max_base_idx = max(order.index(t) for t in base_targets)
        min_eff_idx = min(order.index(t) for t in efficiency_targets)
        
        # All base stats should come before all efficiency stats
        assert max_base_idx < min_eff_idx


def test_game_predictor_predict_single_game():
    """Test predicting a single game."""
    data_obj = MockDataObject()
    
    # Train simple models
    models = {}
    for target in ['home_pass_attempts', 'away_pass_attempts', 'home_points']:
        model = MeanRegressor()
        model.fit(
            np.array([[1], [2], [3]]),
            data_obj.game_data[target].head(3).values
        )
        models[target] = model
    
    predictor = GamePredictor(data_obj=data_obj, models=models)
    
    # Predict a game
    game_row = data_obj.game_data.iloc[-1]
    predictions = predictor.predict_single_game(game_row)
    
    assert isinstance(predictions, dict)
    assert 'home_pass_attempts' in predictions
    assert 'away_pass_attempts' in predictions
    assert 'home_points' in predictions


def test_game_predictor_predict_game_with_dependencies():
    """Test that predictions use dependency values correctly."""
    data_obj = MockDataObject()
    
    # Create models for base and dependent stats
    models = {
        'home_pass_attempts': MeanRegressor(),
        'away_pass_attempts': MeanRegressor(),
        'home_points': MeanRegressor(),  # Depends on attempts
    }
    
    # Train the models
    game_data = data_obj.game_data
    for target, model in models.items():
        y = game_data[target].dropna().values[:10]
        X = np.random.rand(len(y), 5)
        model.fit(X, y)
    
    predictor = GamePredictor(data_obj=data_obj, models=models)
    
    game_row = game_data.iloc[-1]
    predictions = predictor.predict_game(game_row, {})
    
    # Should have predictions for all targets
    assert len(predictions) == 3
    
    # All predictions should be numeric
    for value in predictions.values():
        assert isinstance(value, (int, float, np.number))


def test_game_predictor_skips_insufficient_history():
    """Test that games with insufficient history are skipped."""
    data_obj = MockDataObject()
    data_obj._game_data = data_obj._game_data.head(2)  # Only 2 games
    
    models = {'home_points': MeanRegressor()}
    predictor = GamePredictor(data_obj=data_obj, models=models, min_games=5)
    
    game_row = data_obj.game_data.iloc[-1]
    predictions = predictor.predict_single_game(game_row)
    
    # Should return empty or skip
    assert predictions is None or len(predictions) == 0


# ========== GameModelTrainer Tests ==========

def test_game_model_trainer_initialization():
    """Test GameModelTrainer initializes correctly."""
    data_obj = MockDataObject()
    
    trainer = GameModelTrainer(
        data_obj=data_obj
    )
    
    assert trainer.data_obj == data_obj
    assert trainer.target == 'home_points'
    assert trainer.min_games == 3


def test_game_model_trainer_build_training_rows():
    """Test building training rows from game data."""
    data_obj = MockDataObject()
    
    trainer = GameModelTrainer(
        data_obj=data_obj    
    )
    
    rows = trainer.build_training_rows()
    
    assert isinstance(rows, list)
    assert len(rows) > 0
    
    # Each row should be a tuple (features_dict, target_value)
    if rows:
        features, target_val = rows[0]
        assert isinstance(features, dict)
        assert isinstance(target_val, (int, float, np.number))


def test_game_model_trainer_train_model():
    """Test training a model."""
    data_obj = MockDataObject()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = GameModelTrainer(
            data_obj=data_obj,
            model_dir=tmpdir,
        )
        
        # Train the model
        model, scaler, metrics = trainer.train()
        
        assert model is not None
        assert metrics is not None
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert 'mae' in metrics


def test_game_model_trainer_train_with_custom_factory():
    """Test training with a custom model factory."""
    data_obj = MockDataObject()
    
    def custom_factory():
        return MeanRegressor()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = GameModelTrainer(
            data_obj=data_obj,
            model_factory=custom_factory,
            model_dir=tmpdir,
        )
        
        model, scaler, metrics = trainer.train()
        
        assert isinstance(model, MeanRegressor)


def test_game_model_trainer_saves_artifacts():
    """Test that trainer saves model artifacts."""
    data_obj = MockDataObject()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = GameModelTrainer(
            data_obj=data_obj,
            model_dir=tmpdir,
        )
        
        trainer.train()
        
        # Check that files were created
        model_path = os.path.join(tmpdir, 'home_points.pkl')
        scaler_path = os.path.join(tmpdir, 'scaler_home_points.pkl')
        
        assert os.path.exists(model_path)
        assert os.path.exists(scaler_path)


def test_game_model_trainer_handles_missing_target():
    """Test that trainer handles missing target values gracefully."""
    data_obj = MockDataObject()
    # Set some target values to NaN
    data_obj._game_data.loc[0:5, 'home_points'] = np.nan
    
    trainer = GameModelTrainer(
        data_obj=data_obj
    )
    
    rows = trainer.build_training_rows()
    
    # Should filter out rows with missing targets
    assert isinstance(rows, list)


# ========== Integration Tests ==========

def test_end_to_end_train_and_predict():
    """Test complete workflow: train model, then use for prediction."""
    data_obj = MockDataObject()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train a model
        trainer = GameModelTrainer(
            data_obj=data_obj,
            model_dir=tmpdir,
        )
        
        model, scaler, metrics = trainer.train()
        
        # Create predictor with trained model
        predictor = GamePredictor(
            data_obj=data_obj,
            models={'home_points': model},
            model_dir=tmpdir,
            use_saved_scalers=True,
        )
        
        # Load the scaler
        predictor.scalers = {'home_points': scaler}
        
        # Make a prediction
        game_row = data_obj.game_data.iloc[-1]
        predictions = predictor.predict_single_game(game_row)
        
        assert predictions is not None
        assert 'home_points' in predictions
        assert isinstance(predictions['home_points'], (int, float, np.number))


def test_multiple_targets_dependency_order():
    """Test training and predicting multiple targets in correct order."""
    data_obj = MockDataObject()
    
    targets = ['home_pass_attempts', 'away_pass_attempts', 'home_points', 'away_points']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        models = {}
        
        # Train models for all targets
        for target in targets:
            trainer = GameModelTrainer(
                data_obj=data_obj,
                model_dir=tmpdir,
            )
            model, _, _ = trainer.train()
            models[target] = model
        
        # Predict with all models
        predictor = GamePredictor(
            data_obj=data_obj,
            models=models,
            model_dir=tmpdir,
        )
        
        game_row = data_obj.game_data.iloc[-1]
        predictions = predictor.predict_game(game_row, {})
        
        # Should predict all targets
        assert len(predictions) == len(targets)
        
        # All should be numeric
        for target, value in predictions.items():
            assert isinstance(value, (int, float, np.number))


def test_feature_consistency_across_train_predict():
    """Test that features are consistent between training and prediction."""
    data_obj = MockDataObject()
    
    # Build training features
    trainer = GameModelTrainer(
        data_obj=data_obj
    )
    
    training_rows = trainer.build_training_rows()
    
    # Get feature keys from training
    if training_rows:
        train_feature_keys = set(training_rows[0][0].keys())
        
        # Build prediction features
        predictor = GamePredictor(
            data_obj=data_obj,
            models={'home_points': MeanRegressor()},
        )
        
        game_row = data_obj.game_data.iloc[-1]
        fe = predictor._build_feature_engine('home_points', game_row, {})
        
        pred_feature_keys = set(fe.features.keys())
        
        # Feature keys should match
        assert train_feature_keys == pred_feature_keys


def test_predictor_caches_common_params():
    """Test that predictor properly caches common FeatureEngine parameters."""
    data_obj = MockDataObject()
    
    predictor = GamePredictor(
        data_obj=data_obj,
        models={'home_points': MeanRegressor()},
    )
    
    # Check that common params are cached
    assert hasattr(predictor, '_fe_common_params')
    assert isinstance(predictor._fe_common_params, dict)
    
    # Should contain expected keys
    expected_keys = [
        'schedules', 'standings', 'team_ranks', 'team_ranks_str_cols',
        'team_position_ratings', 'available_positions'
    ]
    
    for key in expected_keys:
        assert key in predictor._fe_common_params


def test_trainer_caches_common_params():
    """Test that trainer properly caches common FeatureEngine parameters."""
    data_obj = MockDataObject()
    
    trainer = GameModelTrainer(
        data_obj=data_obj
    )
    
    # Check that common params are cached
    assert hasattr(trainer, '_fe_common_params')
    assert isinstance(trainer._fe_common_params, dict)


# ========== Edge Cases ==========

def test_predictor_with_empty_models_dict():
    """Test predictor behavior with no models loaded."""
    data_obj = MockDataObject()
    
    predictor = GamePredictor(
        data_obj=data_obj,
        models={},
    )
    
    game_row = data_obj.game_data.iloc[-1]
    predictions = predictor.predict_single_game(game_row)
    
    # Should return empty or None
    assert predictions is None or len(predictions) == 0


def test_trainer_with_insufficient_data():
    """Test trainer with very limited data."""
    data_obj = MockDataObject()
    data_obj._game_data = data_obj._game_data.head(2)
    
    trainer = GameModelTrainer(
        data_obj=data_obj
    )
    
    rows = trainer.build_training_rows()
    
    # Should handle gracefully
    assert isinstance(rows, list)


if __name__ == '__main__':
    # Run a few quick tests
    test_mean_regressor_fits_and_predicts()
    test_game_predictor_initialization()
    test_game_predictor_get_prediction_order()
    test_game_model_trainer_initialization()
    test_game_model_trainer_build_training_rows()
    test_end_to_end_train_and_predict()
    print("All manual tests passed!")
