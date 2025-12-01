"""Lean player prediction and training interfaces.

This module intentionally removes the legacy monolithic `Predictor` class
and all duplicated feature engineering logic. Feature construction is now
exclusively delegated to `FeatureEngine` (see `features.py`). Data loading,
storage mode abstraction (local vs S3), and ancillary metadata are handled
by `DataObject` (see `data_object.py`).

Provided classes:
-----------------
1. PlayerPredictor (inference only)
   - Consumes pre-trained per-target models you supply.
   - Iteratively predicts targets in dependency order so later targets can
     incorporate earlier predictions (e.g. yards depend on attempts &
     completions; fantasy points depend on all positional stat outputs).
   - Uses a private helper to instantiate FeatureEngine consistently.

2. PlayerModelTrainer (simple training helper)
   - Builds per-target training rows from historical player games.
   - Trains a baseline model (RandomForestRegressor if available; else a
     tiny MeanRegressor). Persisted with joblib under `model_dir`.
   - Returns the fitted model object for integration into inference.

Targets & Ordering:
-------------------
Targets are predicted in grouped passes to respect dependencies. Only the
over/under targets present in FeatureEngine's `feature_dependencies` are
kept here (legacy attempted/rush_attempts over/under thresholds removed).

Extending:
----------
Add new target names to the appropriate group lists (or create a new group
if dependency layering changes) and ensure your models dict contains the
`POSITION_target` key with a `.predict(2D_array)` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Sequence, Callable
import os
import logging
import joblib
import numpy as np
import pandas as pd

try:
    from .features import FeatureEngine
    from .data_object import DataObject
except ImportError:
    from features import FeatureEngine
    from data_object import DataObject

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Minimal fallback model (used when a factory isn't provided on training)
# ---------------------------------------------------------------------------
class MeanRegressor:
    """Extremely small baseline: predicts global mean of y."""
    def __init__(self):
        self.mean_: float = 0.0
    def fit(self, X: np.ndarray, y: Sequence[float]):  # type: ignore[override]
        self.mean_ = float(np.mean(y)) if len(y) else 0.0
    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


def default_model_factory() -> Any:
    """Picklable default model factory used by PlayerModelTrainer.

    Avoid using lambdas/closures to keep the trainer instance picklable for
    process-based parallelism on Windows (spawn).
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, random_state=42)
    except Exception:
        return MeanRegressor()


# ---------------------------------------------------------------------------
# PlayerPredictor (Inference)
# ---------------------------------------------------------------------------
@dataclass
class PlayerPredictor:
    data_obj: DataObject
    models: Dict[str, Any]  # key: POSITION_target -> fitted model
    min_games: int = 2
    model_dir: str = "./models"
    use_saved_scalers: bool = False
    scalers: Dict[str, Any] = field(default_factory=dict)  # optional per-target scaler
    model_feature_names: Dict[str, List[str]] = field(default_factory=dict)  # stored feature ordering per model

    # Ordered target groups for iterative prediction
    BASE_VOLUME: List[str] = field(default_factory=lambda: [
        'attempted_passes', 'rush_attempts', 'times_pass_target'
    ])
    DERIVED_VOLUME: List[str] = field(default_factory=lambda: [
        'completed_passes', 'receptions'
    ])
    EFFICIENCY: List[str] = field(default_factory=lambda: [
        'passing_yards', 'passing_touchdowns', 'interceptions_thrown',
        'rush_yards', 'rush_touchdowns', 'receiving_yards', 'receiving_touchdowns'
    ])
    FANTASY: List[str] = field(default_factory=lambda: ['fantasy_points'])
    OVER_UNDER: List[str] = field(default_factory=lambda: [
        'over_under_completed_passes_22+', 'over_under_passing_yards_250+',
        'over_under_passing_touchdowns_2+', 'over_under_interceptions_thrown_1+',
        'over_under_rush_yards_60+', 'over_under_rush_touchdowns_1+',
        'over_under_receptions_5+', 'over_under_receiving_yards_60+',
        'over_under_receiving_touchdowns_1+', 'over_under_rush_yards_&_receiving_yards_100+',
        'over_under_rush_touchdowns_&_receiving_touchdowns_1+'
    ])

    def __post_init__(self):
        os.makedirs(self.model_dir, exist_ok=True)

    def _build_feature_engine(
        self,
        prior_games: pd.DataFrame,
        target: str,
        row: pd.Series,
        position: str,
        predictions: Optional[Dict[str, Any]] = None,
        game_src_df: Optional[pd.DataFrame] = None,
    ) -> FeatureEngine:
        return FeatureEngine(
            prior_games=prior_games,
            target_name=target,
            row=row,
            position=position,
            predicted_features=predictions,
            player_data=self.data_obj.player_data,
            player_data_big_plays=self.data_obj.player_data[
                ['key','game_date','home_abbr','away_abbr','abbr','pos', *self.data_obj.big_play_stat_columns]
            ] if not self.data_obj.player_data.empty else pd.DataFrame(),
            standings=self.data_obj.standings,
            team_ranks=self.data_obj.team_ranks,
            player_group_ranks=self.data_obj.player_group_ranks,
            advanced_stat_cols=self.data_obj.advanced_stat_cols,
            big_play_stat_columns=self.data_obj.big_play_stat_columns,
            game_src_df=game_src_df,
        )

    def predict_player(
        self,
        player_games: pd.DataFrame,
        upcoming_row: pd.Series,
        position: str,
        target_subset: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Predict selected targets for a single upcoming game.

        Parameters
        ----------
        player_games : DataFrame
            Historical games for the player (chronologically sorted or unsorted; will be sorted).
        upcoming_row : Series
            Row describing the upcoming game context.
        position : str
            Player position (QB/RB/WR/TE).
        target_subset : Sequence[str], optional
            If provided, restrict predictions to these targets.
        """
        if len(player_games) < self.min_games:
            raise ValueError(f"Need at least {self.min_games} prior games; got {len(player_games)}")
        player_games = player_games.sort_values('game_date')
        predictions: Dict[str, float] = {}
        active = set(target_subset) if target_subset else None
        ordered_groups = [self.BASE_VOLUME, self.DERIVED_VOLUME, self.EFFICIENCY, self.FANTASY, self.OVER_UNDER]
        for group in ordered_groups:
            for target in group:
                if active and target not in active:
                    continue
                key = f"{position}_{target}"
                model = self.models.get(key)
                if model is None:
                    continue
                # Optional game-level predictions table
                game_src_df = None
                if hasattr(self.data_obj, 'next_game_predictions') and 'game_id' in upcoming_row.index:
                    ngp = getattr(self.data_obj, 'next_game_predictions')
                    if isinstance(ngp, pd.DataFrame) and 'game_id' in ngp.columns:
                        game_src_df = ngp[ngp['game_id'] == upcoming_row.get('game_id')]
                fe = self._build_feature_engine(
                    prior_games=player_games,
                    target=target,
                    row=upcoming_row,
                    position=position,
                    predictions=predictions if predictions else None,
                    game_src_df=game_src_df,
                )
                X_df = pd.DataFrame([fe.features])
                # Ensure numeric-only and align to trained feature set if available
                if key in self.model_feature_names:
                    needed = self.model_feature_names[key]
                    # add any missing needed cols with 0.0
                    for col in needed:
                        if col not in X_df.columns:
                            X_df[col] = 0.0
                    # drop unexpected extra cols
                    X_df = X_df[needed]
                else:
                    X_df = X_df.select_dtypes(include=[np.number])
                scaler = self.scalers.get(key) if self.use_saved_scalers else None
                if scaler is not None:
                    # if scaler has feature names ensure ordering
                    fn = getattr(scaler, 'feature_names_in_', None)
                    if fn is not None:
                        missing = [c for c in fn if c not in X_df.columns]
                        for c in missing:
                            X_df[c] = 0.0
                        X_df = X_df[fn]
                    X_arr = scaler.transform(X_df.values)
                else:
                    X_arr = X_df.values
                pred_val = float(model.predict(X_arr)[0])
                predictions[target] = pred_val
        return predictions

    def load_models_from_dir(self, positions: Sequence[str], targets: Sequence[str]) -> None:
        """Load models from `model_dir` named as POSITION_target.pkl (joblib)."""
        for pos in positions:
            for tgt in targets:
                key = f"{pos}_{tgt}"
                path = os.path.join(self.model_dir, f"{key}.pkl")
                if not os.path.exists(path):
                    continue
                try:
                    obj = joblib.load(path)
                    if isinstance(obj, dict) and 'model' in obj:
                        self.models[key] = obj['model']
                        if 'scaler' in obj and obj['scaler'] is not None:
                            self.scalers[key] = obj['scaler']
                        if 'feature_names' in obj and isinstance(obj['feature_names'], list):
                            self.model_feature_names[key] = obj['feature_names']
                    else:  # backward compatibility: raw model without metadata
                        self.models[key] = obj
                    logging.info(f"Loaded model: {path}")
                except Exception as e:
                    logging.warning(f"Failed loading {path}: {e}")

    def save_model(self, key: str, model: Any, scaler: Any = None) -> None:
        path = os.path.join(self.model_dir, f"{key}.pkl")
        joblib.dump({'model': model, 'scaler': scaler}, path)
        logging.info(f"Saved model {key} -> {path}")


# ---------------------------------------------------------------------------
# PlayerModelTrainer (Training)
# ---------------------------------------------------------------------------
@dataclass
class PlayerModelTrainer:
    data_obj: DataObject
    min_games: int = 2
    model_factory: Optional[Callable[[], Any]] = None
    model_dir: str = "./models"

    def __post_init__(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if self.model_factory is None:
            # Use a top-level function to keep instance picklable in parallel
            self.model_factory = default_model_factory

    def _build_feature_engine(
        self,
        prior_games: pd.DataFrame,
        target: str,
        row: pd.Series,
        position: str,
    ) -> FeatureEngine:
        return FeatureEngine(
            prior_games=prior_games,
            target_name=target,
            row=row,
            position=position,
            player_data=self.data_obj.player_data,
            player_data_big_plays=self.data_obj.player_data[
                ['key','game_date','home_abbr','away_abbr','abbr','pos', *self.data_obj.big_play_stat_columns]
            ] if not self.data_obj.player_data.empty else pd.DataFrame(),
            standings=self.data_obj.standings,
            team_ranks=self.data_obj.team_ranks,
            player_group_ranks=self.data_obj.player_group_ranks,
            advanced_stat_cols=self.data_obj.advanced_stat_cols,
            big_play_stat_columns=self.data_obj.big_play_stat_columns,
        )

    def build_training_rows(self, player_games: pd.DataFrame, position: str, target: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        player_games = player_games.sort_values('game_date')
        for idx in range(self.min_games, len(player_games)):
            prior = player_games.iloc[:idx]
            current = player_games.iloc[idx]
            logging.info(f"Building training row for {current['pid']} on {current['game_date'].date()} target={target}")
            fe = self._build_feature_engine(prior, target, current, position)
            feat = fe.features.copy()
            feat[f"target_{target}"] = current.get(target, np.nan)
            rows.append(feat)
        return rows

    def train_target(self, position: str, target: str) -> Any:
        df = self.data_obj.player_data
        if df.empty:
            raise ValueError("player_data is empty; cannot train")
        pos_df = df[df['pos'] == position]
        all_rows: List[Dict[str, Any]] = []
        for pid, pid_games in pos_df.groupby('pid'):
            if len(pid_games) < self.min_games:
                continue
            all_rows.extend(self.build_training_rows(pid_games, position, target))
        if not all_rows:
            raise ValueError(f"No training rows generated for {position} {target}")
        train_df = pd.DataFrame(all_rows).fillna(0.0)
        y_col = f"target_{target}"
        X_full = train_df.drop(columns=[y_col])
        # keep only numeric columns; drop datetime/object to avoid float conversion errors
        X = X_full.select_dtypes(include=[np.number]).copy()
        y = train_df[y_col].values
        # model_factory may be either a factory function or a class
        model = self.model_factory() if callable(self.model_factory) else default_model_factory()
        model.fit(X.values, y)
        key = f"{position}_{target}"
        joblib.dump({'model': model, 'feature_names': list(X.columns)}, os.path.join(self.model_dir, f"{key}.pkl"))
        logging.info(f"Trained & saved model {key}; rows={len(train_df)} features={X.shape[1]} numeric_only")
        return model

    def _train_one_safe(self, position: str, target: str):
        try:
            model = self.train_target(position, target)
            return (target, model, None)
        except Exception as e:
            logging.warning(f"Skipping {position}_{target}: {e}")
            return (target, None, e)

    def train_many(self, position: str, targets: Sequence[str], n_jobs: int = -1) -> Dict[str, Any]:
        """Train multiple targets, optionally in parallel.

        Parameters
        ----------
        position : str
            Player position (QB/RB/WR/TE).
        targets : Sequence[str]
            Target names to train models for.
        n_jobs : int, default -1
            Number of worker processes. -1 uses all available cores.
        """
        if not targets:
            return {}

        # Fast path for single-core
        if n_jobs == 1:
            out: Dict[str, Any] = {}
            for t in targets:
                key, model, err = self._train_one_safe(position, t)
                if model is not None:
                    out[key] = model
            return out

        # Parallel path (process-based; safe on Windows)
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(PlayerModelTrainer._train_one_safe)(self, position, t) for t in targets
        )
        out: Dict[str, Any] = {k: m for (k, m, e) in results if m is not None}
        return out


if __name__ == "__main__":  # minimal manual smoke example
    data_obj = DataObject(
        league='nfl',
        storage_mode='local',
        local_root='G:/My Drive/python/winsight_api/sports-data-storage-copy/'
    )
    trainer = PlayerModelTrainer(data_obj)
    # Example: train a couple QB volume stats if data present
    sample_targets = ['attempted_passes', 'completed_passes']
    try:
        trainer.train_many('QB', sample_targets)
    except Exception as e:
        logging.warning(f"Training skipped (sample): {e}")
    predictor = PlayerPredictor(data_obj=data_obj, models={})
    predictor.load_models_from_dir(['QB'], sample_targets)
    sample_qb = data_obj.player_data[data_obj.player_data['pos']=='QB']
    if not sample_qb.empty:
        gid = sample_qb['pid'].iloc[0]
        games: pd.DataFrame = sample_qb[sample_qb['pid']==gid]
        upcoming = games.iloc[-1].copy()
        upcoming['game_date'] = upcoming['game_date'] + pd.Timedelta(days=7)
        preds = predictor.predict_player(games, upcoming, 'QB', target_subset=sample_targets)
        logging.info(f"Smoke predictions: {preds}")