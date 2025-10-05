# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Reframed documentation and metadata to focus on electric load/demand forecasting while keeping the API contract unchanged.
- Harden the full streaming and orchestration stack.
- Document deployment targets beyond Docker Compose.

## [1.0.0] - 2025-08-18
### Added
- Production-ready FastAPI serving image with Docker Compose quickstart.
- Automated tests covering health and prediction endpoints, plus feature validation.
- GitHub Actions workflow for linting, testing, and publishing images to GHCR.
- Project hygiene docs: contributing guide, security policy, code of conduct, roadmap.
- Pre-commit hooks configuration (Black, Ruff, whitespace hygiene).
