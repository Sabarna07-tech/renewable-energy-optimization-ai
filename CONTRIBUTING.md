# Contributing

Thanks for your interest in improving Energy Load Forecasting! We welcome pull requests and issue reports that help harden the serving stack and surrounding tooling.

## Prerequisites

- Python 3.11
- Docker and Docker Compose
- `pip install -r serving/requirements.txt`
- `pip install -r requirements.txt` (for pipeline extras)
- `pip install pre-commit` and run `pre-commit install`

## Local development

1. Create a feature branch from `main`.
2. Install dependencies: `pip install -r serving/requirements.txt`.
3. Run unit tests: `pytest -q`.
4. Build and run the API locally:
   ```bash
   docker compose -f docker-compose.min.yml up --build -d
   curl -s localhost:8000/health
   ```
5. Stop the stack when you are done: `docker compose -f docker-compose.min.yml down`.

## Coding standards

- Python code is formatted with **Black** and linted with **Ruff**.
- Run `pre-commit run --all-files` before pushing.
- Keep dependencies pinned or bounded as in the sample requirements files.
- Avoid committing secrets or large binary artifacts.

## Opening a pull request

- Rebase on the latest `main` and resolve conflicts locally.
- Fill out the pull request template checklist.
- Explain any noteworthy design decisions in self-review comments on the diff.
- Ensure the CI workflow is passing before requesting review.

We appreciate every contribution and look forward to collaborating with you!
