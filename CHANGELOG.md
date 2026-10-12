# Changelog

## v0.1 — Baseline
- StandardScaler + LinearRegression, fixed 80/20 split (random_state=42).
- FastAPI health and prediction endpoints with input validation.
- Docker and automated CI/release pipeline.
- RMSE: 53.8534 in an initial local run (recheck in pinned CI environment).

## v0.2 — Improvement (complete after evaluating)
- Candidate: StandardScaler + Ridge(alpha=10.0).
- Initial local RMSE: v0.1 = 53.8534; v0.2 = 53.6263; reduction = 0.2272 (about 0.42%). Confirm these values in CI.
- Ridge adds L2 regularization, which slightly reduced held-out RMSE in the initial local run. Confirm before release.
