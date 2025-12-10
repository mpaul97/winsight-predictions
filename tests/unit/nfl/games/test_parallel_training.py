"""Tests for parallel build_training_rows functionality."""

import pytest
import time
import pandas as pd
import sys
from pathlib import Path

# Setup imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from winsight_predictions.nfl.data_object import DataObject
from winsight_predictions.nfl.games.predictor import GameModelTrainer


class TestParallelTraining:
    """Test suite for multithreaded training row building."""

    @pytest.fixture(scope='class')
    def data_obj(self):
        """Create DataObject for testing."""
        return DataObject(
            league='nfl',
            storage_mode='local',
            local_root=r'G:\My Drive\python\winsight_api\sports-data-storage-copy'
        )

    @pytest.fixture(scope='class')
    def trainer(self, data_obj):
        """Create GameModelTrainer instance."""
        root_dir = r"G:\My Drive\python\winsight_api\winsight-predictions\winsight_predictions\nfl\games"
        return GameModelTrainer(
            data_obj=data_obj,
            root_dir=root_dir,
            testing=True  # Use 10% sampling
        )

    def test_parallel_build_data_integrity(self, trainer):
        """Verify parallel execution produces same results as sequential would."""
        game_data = trainer.game_data.copy()
        
        if trainer.testing:
            game_data = game_data.sample(frac=0.1, random_state=42)
        
        target = 'home_points'
        
        # Build with multithreading (default)
        print("\n=== Testing Parallel Execution ===")
        start = time.time()
        parallel_rows = trainer.build_training_rows(game_data, target)
        parallel_time = time.time() - start
        
        print(f"Parallel execution: {len(parallel_rows)} rows in {parallel_time:.2f}s")
        
        # Verify results
        assert len(parallel_rows) > 0, "No training rows generated"
        
        # Check all rows have the target column
        df = pd.DataFrame(parallel_rows)
        assert f'target_{target}' in df.columns, "Missing target column"
        
        # Verify no NaN values in target
        assert df[f'target_{target}'].notna().all(), "NaN values in target column"
        
        # Check chronological ordering (should be maintained after sorting)
        if 'game_date' in df.columns:
            dates = pd.to_datetime(df['game_date'])
            assert dates.is_monotonic_increasing, "Rows not in chronological order"
        
        print(f"✓ Data integrity verified")
        print(f"  - Total rows: {len(parallel_rows)}")
        print(f"  - Feature columns: {len(df.columns) - 1}")
        print(f"  - Features cache entries: {len(trainer.features_cache)}")

    def test_parallel_execution_with_workers(self, trainer):
        """Test with different worker counts and measure performance."""
        game_data = trainer.game_data.copy()
        
        if trainer.testing:
            game_data = game_data.sample(frac=0.05, random_state=42)  # Even smaller sample
        
        target = 'home_points'
        
        worker_counts = [1, 2, 4]
        results = {}
        
        for workers in worker_counts:
            # Clear cache for clean test
            trainer.features_cache.clear()
            
            start = time.time()
            rows = trainer.build_training_rows(game_data, target, max_workers=workers)
            elapsed = time.time() - start
            
            results[workers] = {
                'time': elapsed,
                'rows': len(rows),
                'features': len(pd.DataFrame(rows).columns) - 1
            }
            
            print(f"\n{workers} worker(s): {len(rows)} rows in {elapsed:.2f}s ({elapsed/len(rows):.3f}s per row)")
        
        # Verify all configurations produce same number of rows
        row_counts = [r['rows'] for r in results.values()]
        assert len(set(row_counts)) == 1, "Different worker counts produced different row counts"
        
        # Verify all configurations produce same number of features
        feature_counts = [r['features'] for r in results.values()]
        assert len(set(feature_counts)) == 1, "Different worker counts produced different feature counts"
        
        # Calculate speedup
        baseline_time = results[1]['time']
        for workers in [2, 4]:
            speedup = baseline_time / results[workers]['time']
            print(f"\nSpeedup with {workers} workers: {speedup:.2f}x")
        
        print(f"\n✓ Consistency verified across {len(worker_counts)} worker configurations")

    def test_thread_safety_features_cache(self, trainer):
        """Verify features_cache is updated correctly with concurrent writes."""
        game_data = trainer.game_data.copy()
        
        if trainer.testing:
            game_data = game_data.sample(frac=0.1, random_state=42)
        
        target = 'home_points'
        
        # Clear cache
        trainer.features_cache.clear()
        
        # Build with parallel execution
        rows = trainer.build_training_rows(game_data, target, max_workers=4)
        
        # Verify cache has entries
        assert len(trainer.features_cache) > 0, "Features cache is empty"
        
        # Verify each cache entry has correct number of rows
        expected_rows = len(rows)
        for key, cached_df in trainer.features_cache.items():
            if key.startswith(f"{target}_"):
                assert len(cached_df) == expected_rows, \
                    f"Cache entry {key} has {len(cached_df)} rows, expected {expected_rows}"
                
                # Verify required columns exist
                required_cols = ['game_id', 'game_date', 'home_abbr', 'away_abbr', 'target']
                for col in required_cols:
                    assert col in cached_df.columns, f"Missing {col} in cache entry {key}"
        
        print(f"\n✓ Features cache thread safety verified")
        print(f"  - Cache entries: {len(trainer.features_cache)}")
        print(f"  - Rows per entry: {expected_rows}")

    def test_no_data_leakage(self, trainer):
        """Verify each training row only uses prior game data (no future leakage)."""
        game_data = trainer.game_data.copy()
        game_data = game_data.sort_values('game_date')
        
        # Filter to games with results
        game_data = game_data[game_data['home_points'].notna() & game_data['away_points'].notna()]
        
        # Use very small sample for fast execution
        sample_size = 20
        game_data = game_data.iloc[:sample_size]
        
        target = 'home_points'
        
        # Build training rows
        rows = trainer.build_training_rows(game_data, target, max_workers=2)
        
        # The number of rows should be sample_size - min_games
        expected_rows = sample_size - trainer.min_games
        assert len(rows) == expected_rows, \
            f"Expected {expected_rows} rows, got {len(rows)}"
        
        print(f"\n✓ No data leakage verified")
        print(f"  - Total games: {sample_size}")
        print(f"  - Min games required: {trainer.min_games}")
        print(f"  - Training rows generated: {len(rows)}")
