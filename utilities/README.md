# Utilities Directory

This directory contains utility scripts, fixes, and patches for the LLM Position Bias Analysis Framework.

## 📁 Contents

### Data Processing Fixes
- **`fix_candidate_list_method.py`**: Fixes for candidate list generation method
- **`fix_news_data_processing.py`**: Fixes for news dataset processing
- **`improved_news_processing.py`**: Enhanced news data processing

### Method Patches
- **`patch_create_candidate_list.py`**: Patches for candidate list creation
- **`test_patched_method.py`**: Tests for patched methods

### Testing Utilities
- **`test_beauty_fix.py`**: Beauty dataset specific tests
- **`test_fixed_news_dataset.py`**: News dataset fix tests
- **`test_patched_method.py`**: Method patch tests

### Debugging Tools
- **`debug_json_parsing.py`**: JSON parsing debugging utilities

## 🎯 Usage

### Applying Fixes
```python
# Import and apply fixes
from utilities.fix_candidate_list_method import fix_candidate_list_method
from utilities.patch_create_candidate_list import patch_create_candidate_list

# Apply fixes to analyzer
fix_candidate_list_method(analyzer)
patch_create_candidate_list(analyzer)
```

### Testing Fixes
```bash
# Run specific fix tests
python utilities/test_beauty_fix.py
python utilities/test_fixed_news_dataset.py
python utilities/test_patched_method.py
```

## 🔧 Fix Categories

### Candidate List Issues
- **Problem**: Insufficient candidates for bias users
- **Solution**: Enhanced candidate generation logic
- **Files**: `fix_candidate_list_method.py`, `patch_create_candidate_list.py`

### News Dataset Issues
- **Problem**: MIND dataset format compatibility
- **Solution**: Improved data processing pipeline
- **Files**: `fix_news_data_processing.py`, `improved_news_processing.py`

### Method Compatibility
- **Problem**: API changes and method updates
- **Solution**: Patched method implementations
- **Files**: `test_patched_method.py`

## 🧪 Testing

### Running Tests
```bash
# Run all utility tests
python -m pytest utilities/test_*.py -v

# Run specific test categories
python -m pytest utilities/test_*fix*.py -v
python -m pytest utilities/test_*patch*.py -v
```

### Test Coverage
- **Data Processing**: Validate fix effectiveness
- **Method Patches**: Ensure compatibility
- **Error Handling**: Test edge cases
- **Performance**: Verify no regression

## 📊 Debugging

### JSON Parsing Issues
```python
from utilities.debug_json_parsing import debug_json_response

# Debug LLM responses
debug_json_response(llm_response)
debug_json_response(problematic_json)
```

### Common Problems
- **Malformed JSON**: LLM response parsing errors
- **Missing Fields**: Incomplete response data
- **Type Mismatches**: Expected vs actual data types
- **Encoding Issues**: Character encoding problems

## 🔄 Maintenance

### Adding New Fixes
1. **Identify Problem**: Document the issue clearly
2. **Create Fix**: Implement solution in new file
3. **Add Tests**: Create comprehensive test coverage
4. **Update Documentation**: Document fix and usage
5. **Version Control**: Commit with descriptive message

### Fix Naming Convention
- **`fix_*_*.py`**: Specific fixes for named components
- **`patch_*_*.py`**: Method patches and updates
- **`improved_*_*.py`**: Enhanced implementations
- **`test_*_*.py`**: Test files for fixes

## 🚨 Best Practices

### Fix Development
- **Isolate Changes**: Make minimal, focused fixes
- **Test Thoroughly**: Verify fix doesn't break existing functionality
- **Document Changes**: Clear explanation of what was fixed
- **Version Control**: Track all changes and fixes

### Testing Strategy
- **Unit Tests**: Test individual fix components
- **Integration Tests**: Test fix with full system
- **Regression Tests**: Ensure no new issues introduced
- **Performance Tests**: Verify no performance degradation

### Documentation
- **Problem Description**: What was broken
- **Solution Details**: How it was fixed
- **Usage Examples**: How to apply the fix
- **Known Limitations**: Any remaining issues

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Check Python path and dependencies
2. **Test Failures**: Verify test data and environment
3. **Fix Conflicts**: Resolve multiple fix interactions
4. **Performance Issues**: Monitor for regression

### Debug Steps
1. **Check Logs**: Review error messages and warnings
2. **Verify Environment**: Confirm dependencies and versions
3. **Test Isolation**: Run fixes independently
4. **Compare Results**: Before/after analysis

## 📈 Future Improvements

### Planned Enhancements
- **Automated Testing**: CI/CD integration for fixes
- **Fix Validation**: Automated verification of fix effectiveness
- **Performance Monitoring**: Track fix impact on performance
- **Documentation Generation**: Auto-generate fix documentation

### Contributing
- **Report Issues**: Create detailed bug reports
- **Propose Fixes**: Suggest solutions and improvements
- **Submit PRs**: Contribute tested fixes
- **Review Code**: Help maintain code quality
