# Changelog

## v0.2 — Improvement (complete after evaluating)
- Candidate: StandardScaler + Ridge(alpha=10.0).
- Replaced the Linear Regression baseline with Ridge Regression.
   - Maintained the same dataset, random state (42), and test split (20%) for   comparison.
    - Improved test RMSE from 53.8534 (v0.1) to 53.6263 (v0.2), approximately a 0.42% reduction.
    - Updated the model version to v0.2.
    - Evaluation
        v0.1: Linear Regression — RMSE 53.8534
        v0.2: Ridge Regression — RMSE 53.6263
    - Conclusion
        Ridge Regression achieved a modest improvement in prediction accuracy compared with the baseline model.

## v0.1 — Baseline
- StandardScaler + LinearRegression, fixed 80/20 split (random_state=42).
- FastAPI health and prediction endpoints with input validation.
- Docker and automated CI/release pipeline.
- RMSE: 53.8534 in an initial local run (recheck in pinned CI environment).


