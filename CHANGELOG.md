# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup and documentation
- Comprehensive README with usage examples
- Development guide and contribution guidelines
- Testing framework with pytest
- Code quality tools (black, isort, flake8, mypy)
- Pre-commit hooks for automated quality checks
- Makefile for common development tasks
- Project structure documentation
- Environment configuration examples

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [1.0.0] - 2024-01-XX

### Added
- Core LLM Position Bias Analysis Framework
- Support for multiple datasets (MovieLens, Books, Music, News, Beauty, Steam)
- OpenAI and Anthropic API integration
- Position bias detection algorithms
- Propensity scoring for debiasing
- Comprehensive evaluation metrics (NDCG, Accuracy)
- Rate limiting and API tier management
- Checkpoint system for long-running experiments
- Batch processing capabilities
- Multi-user analysis support

### Technical Features
- Async/await support for API calls
- Configurable rate limiting tiers
- Memory-efficient data processing
- Extensible dataset support
- Comprehensive error handling
- Logging and monitoring capabilities

## [0.2.0] - 2024-XX-XX

### Added
- Additional LLM provider support
- Advanced debiasing algorithms
- Real-time bias monitoring
- Web interface for analysis
- Integration with popular ML frameworks

### Changed
- Improved performance and memory efficiency
- Enhanced error handling and recovery
- Better documentation and examples

## [0.1.0] - 2024-XX-XX

### Added
- Basic position bias detection
- Simple evaluation metrics
- Core framework structure

---

## Version History

- **1.0.0**: Production-ready framework with comprehensive features
- **0.2.0**: Enhanced functionality and performance improvements
- **0.1.0**: Initial prototype and basic implementation

## Contributing

To add entries to this changelog:

1. Add your changes under the `[Unreleased]` section
2. Use the appropriate category (Added, Changed, Deprecated, Removed, Fixed, Security)
3. Provide a clear, concise description of the change
4. Include issue numbers or PR references when applicable

## Release Process

1. **Development**: Changes accumulate under `[Unreleased]`
2. **Release**: Move `[Unreleased]` content to new version section
3. **Tag**: Create git tag for the release version
4. **Update**: Update version numbers in setup.py and pyproject.toml
