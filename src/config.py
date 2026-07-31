from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


PRODUCTION_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_matches_expanded_clean.csv"
)

LEGACY_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_matches.csv"
)

PRODUCTION_MODEL = (
    PROJECT_ROOT
    / "saved_models"
    / "match_model.joblib"
)

EXPANDED_CANDIDATE_MODEL = (
    PROJECT_ROOT
    / "saved_models"
    / "expanded_match_model_candidate.joblib"
)