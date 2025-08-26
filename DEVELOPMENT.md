# Development Guide

This guide provides detailed information for developers who want to contribute to or work with the LLM Position Bias Analysis Framework.

## 🚀 Quick Start for Developers

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd debiased_ranking

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make install-dev

# Set up environment configuration
make setup-env
```

### 2. Verify Installation

```bash
# Run tests to verify everything works
make test

# Run the basic example
make run-example

# Check code quality
make check
```

## 🔧 Development Workflow

### Daily Development

```bash
# Start development session
make dev

# Make your changes...

# Before committing, run quality checks
make check

# If all checks pass, commit your changes
git add .
git commit -m "Description of changes"
```

### Code Quality Standards

The project uses several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting and style checking
- **mypy**: Type checking
- **pytest**: Testing framework

### Running Quality Checks

```bash
# Format code
make format

# Check linting
make lint

# Run tests
make test

# Run all checks
make check
```

## 🧪 Testing

### Test Structure

```
tests/
├── __init__.py
├── test_basic_functionality.py    # Core functionality tests
├── test_data_processing.py        # Data handling tests
├── test_bias_analysis.py          # Bias analysis tests
└── test_integration.py            # End-to-end tests
```

### Writing Tests

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Structure**: Follow the Arrange-Act-Assert pattern
3. **Fixtures**: Use pytest fixtures for common setup
4. **Mocking**: Mock external dependencies (API calls, file I/O)

Example test:

```python
def test_user_filtering_with_sufficient_data():
    """Test that users with sufficient interaction history are selected"""
    # Arrange
    data = create_sample_data_with_varying_interactions()
    
    # Act
    analyzer = LLMPositionBiasAnalyzer(data=data, ...)
    
    # Assert
    assert len(analyzer.bias_users) > 0
    assert all(user in analyzer.data['UserID'].values for user in analyzer.bias_users)
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
python -m pytest tests/test_basic_functionality.py -v

# Run tests matching pattern
python -m pytest -k "test_user" -v
```

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type Hints**: Include type hints for function parameters and return values
- **Comments**: Add inline comments for complex logic

Example:

```python
def create_candidate_list(
    self, 
    user_id: str, 
    num_candidates: int = 100
) -> Tuple[List[str], List[str], str, int]:
    """
    Create a candidate list for a given user.
    
    Args:
        user_id: The user identifier
        num_candidates: Number of candidates to generate
        
    Returns:
        Tuple containing:
        - candidate_list: List of candidate items
        - user_items: User's interaction history
        - last_item: Most recent interaction
        - actual_size: Actual number of candidates generated
    """
```

### API Documentation

- **README.md**: Main project documentation
- **PROJECT_STRUCTURE.md**: Project organization guide
- **DEVELOPMENT.md**: This development guide
- **Examples**: Jupyter notebooks demonstrating usage

## 🔄 Contributing

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/debiased_ranking.git
cd debiased_ranking

# Add upstream remote
git remote add upstream https://github.com/original/debiased_ranking.git
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow the coding standards
- Write tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 4. Test Your Changes

```bash
# Run all quality checks
make check

# Run specific tests for your changes
python -m pytest tests/test_your_feature.py -v
```

### 5. Submit Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Create PR on GitHub with:
# - Clear description of changes
# - Link to related issues
# - Screenshots if UI changes
# - Test results
```

## 🐛 Debugging

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **API Errors**: Check API keys and rate limits
3. **Memory Issues**: Reduce dataset size or use chunking
4. **Test Failures**: Check test data and mock configurations

### Debug Tools

```bash
# Enable verbose logging
export VERBOSE=true

# Run with debug output
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

# Use Python debugger
import pdb; pdb.set_trace()
```

### Logging

The framework uses structured logging. Configure logging levels:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 📊 Performance Optimization

### API Rate Limiting

- Use appropriate API tier for your usage
- Monitor API usage in provider dashboard
- Implement exponential backoff for errors

### Memory Management

- Process large datasets in chunks
- Use generators for large data streams
- Monitor memory usage during processing

### Parallel Processing

- Use the built-in worker pool for API calls
- Adjust `max_workers` based on your API limits
- Consider batch processing for efficiency

## 🔒 Security

### API Key Management

- Never commit API keys to version control
- Use environment variables for configuration
- Rotate keys regularly
- Monitor API usage for anomalies

### Data Privacy

- Ensure datasets don't contain sensitive information
- Use anonymized data for testing
- Follow data protection regulations

## 🚀 Deployment

### Local Development

```bash
# Install in development mode
pip install -e .

# Run from anywhere
llm-bias-analyzer --help
```

### Production Deployment

```bash
# Build distribution
make dist

# Install from wheel
pip install dist/llm_position_bias-*.whl
```

### Docker (Future)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["llm-bias-analyzer", "--help"]
```

## 📈 Monitoring and Logging

### Performance Metrics

- API response times
- Memory usage
- Processing throughput
- Error rates

### Log Analysis

```bash
# Search logs for errors
grep "ERROR" logs/app.log

# Monitor API usage
grep "API call" logs/app.log | wc -l
```

## 🤝 Community

### Getting Help

- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Check README and project structure docs
- **Examples**: Review Jupyter notebooks

### Contributing Guidelines

1. **Be respectful**: Follow the code of conduct
2. **Help others**: Answer questions and review PRs
3. **Share knowledge**: Document your learnings
4. **Follow standards**: Adhere to coding guidelines

## 🔮 Future Development

### Planned Features

- [ ] Support for more LLM providers
- [ ] Advanced debiasing algorithms
- [ ] Real-time bias monitoring
- [ ] Web interface for analysis
- [ ] Integration with ML frameworks

### How to Contribute

1. **Pick an issue**: Choose from the roadmap or create your own
2. **Discuss approach**: Open an issue to discuss implementation
3. **Implement**: Write code following the guidelines
4. **Test thoroughly**: Ensure all tests pass
5. **Document**: Update relevant documentation
6. **Submit PR**: Follow the contribution workflow

---

**Happy coding! 🎉**

If you have questions or need help, don't hesitate to reach out to the community.
