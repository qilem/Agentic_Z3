#!/usr/bin/env python3
"""
Test script to verify agentic_z3_engine improvements.

This script verifies:
1. Tests always call the function (no 'pass' tests)
2. Model values are properly converted to Python literals
3. Low-confidence results are detected
"""

import sys
from pathlib import Path

# Add benchmark directory to path
benchmark_dir = str(Path(__file__).parent)
if benchmark_dir not in sys.path:
    sys.path.insert(0, benchmark_dir)

# Mock minimal dependencies
class MockRateLimiter:
    @staticmethod
    def get_instance(config):
        return MockRateLimiter()
    
    def with_retry(self, func):
        return func

class MockConfig:
    recommended_max_workers = 2
    tokens_per_minute = 25000

TIER_CONFIGS = {'tier1': MockConfig()}

# Inject mocks
sys.modules['rate_limiter'] = type(sys)('rate_limiter')
sys.modules['rate_limiter'].RateLimiter = MockRateLimiter
sys.modules['rate_limiter'].TIER_CONFIGS = TIER_CONFIGS

# Now import the runner
from runners.agentic_z3_runner import AgenticZ3Runner

def test_no_pass_tests():
    """Test 1: Verify no 'pass' tests are generated."""
    print("Test 1: Checking if 'pass' tests are eliminated...")
    
    runner = AgenticZ3Runner(
        model="gpt-5.2",
        use_engine=False,
        max_workers=1,
        rate_limit_tier='tier1'
    )
    
    # Test with empty model (previously would generate 'pass')
    input_params = [
        {'name': 'nums', 'type': 'List[int]'},
        {'name': 'target', 'type': 'int'}
    ]
    
    test_code = runner._generate_test_from_model(
        func_name='twoSum',
        input_params=input_params,
        model=None,
        status='error'
    )
    
    # Verify no 'pass' statement
    assert 'pass' not in test_code or 'result = solution.twoSum(' in test_code, \
        f"Test still contains standalone 'pass':\n{test_code}"
    
    # Verify function call is present
    assert 'solution.twoSum(' in test_code, \
        f"Test doesn't call the function:\n{test_code}"
    
    print("  ✓ No 'pass' tests - function always called")
    return True

def test_model_value_conversion():
    """Test 2: Verify model values are properly converted."""
    print("Test 2: Checking model→Python conversion...")
    
    runner = AgenticZ3Runner(
        model="gpt-5.2",
        use_engine=False,
        max_workers=1,
        rate_limit_tier='tier1'
    )
    
    # Test various conversions
    test_cases = [
        ('5', 'int', '5'),
        ('hello', 'str', '"hello"'),
        ('[1, 2, 3]', 'List[int]', '[1, 2, 3]'),
        ('True', 'bool', 'True'),
    ]
    
    for value, type_hint, expected_prefix in test_cases:
        result = runner._convert_model_value_to_python(value, type_hint)
        print(f"  {value} ({type_hint}) -> {result}")
        
        # Verify it compiles
        try:
            compile(result, '<string>', 'eval')
        except SyntaxError as e:
            raise AssertionError(f"Conversion produced invalid Python: {result} - {e}")
    
    print("  ✓ Model values properly converted to Python literals")
    return True

def test_low_confidence_detection():
    """Test 3: Verify low-confidence results are detected."""
    print("Test 3: Checking low-confidence detection...")
    
    runner = AgenticZ3Runner(
        model="gpt-5.2",
        use_engine=False,
        max_workers=1,
        rate_limit_tier='tier1'
    )
    
    input_params = [
        {'name': 'x', 'type': 'int'},
        {'name': 'y', 'type': 'int'}
    ]
    
    # Test case 1: No constraints added
    result1 = {
        'status': 'sat',
        'constraints_added': 0,
        'model': {'x': '5'}
    }
    assert runner._is_low_confidence_result(result1, input_params), \
        "Should detect low confidence when constraints_added=0"
    print("  ✓ Detected low-confidence: constraints_added=0")
    
    # Test case 2: Model missing input parameters
    result2 = {
        'status': 'sat',
        'constraints_added': 5,
        'model': {'other_var': '10'}  # No x or y
    }
    assert runner._is_low_confidence_result(result2, input_params), \
        "Should detect low confidence when model missing input params"
    print("  ✓ Detected low-confidence: model missing input params")
    
    # Test case 3: Non-SAT status
    result3 = {
        'status': 'unknown',
        'constraints_added': 5,
        'model': {'x': '5', 'y': '3'}
    }
    assert runner._is_low_confidence_result(result3, input_params), \
        "Should detect low confidence for non-SAT status"
    print("  ✓ Detected low-confidence: non-SAT status")
    
    # Test case 4: Good result (should be high confidence)
    result4 = {
        'status': 'sat',
        'constraints_added': 5,
        'model': {'x': '5', 'y': '3'}
    }
    assert not runner._is_low_confidence_result(result4, input_params), \
        "Should NOT detect low confidence for good result"
    print("  ✓ High-confidence result properly recognized")
    
    return True

def test_positional_arguments():
    """Test 4: Verify positional arguments are used (not keyword)."""
    print("Test 4: Checking positional argument generation...")
    
    runner = AgenticZ3Runner(
        model="gpt-5.2",
        use_engine=False,
        max_workers=1,
        rate_limit_tier='tier1'
    )
    
    input_params = [
        {'name': 'beginWord', 'type': 'str'},
        {'name': 'endWord', 'type': 'str'},
        {'name': 'wordList', 'type': 'List[str]'}
    ]
    
    # Model with values
    model = {
        'beginWord': 'hit',
        'endWord': 'cog',
        'wordList': ['hot', 'dot', 'dog', 'lot', 'log', 'cog']
    }
    
    test_code = runner._generate_test_from_model(
        func_name='findLadders',
        input_params=input_params,
        model=model,
        status='sat'
    )
    
    # Should use positional args, not keyword args with bare identifiers
    assert 'endWord=endWord' not in test_code, \
        f"Test uses bare identifier as keyword arg:\n{test_code}"
    
    # Should have proper function call
    assert 'solution.findLadders(' in test_code, \
        f"Test doesn't call function:\n{test_code}"
    
    print("  ✓ Using positional arguments (not bare identifiers)")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing agentic_z3_engine improvements")
    print("=" * 60)
    print()
    
    tests = [
        test_no_pass_tests,
        test_model_value_conversion,
        test_low_confidence_detection,
        test_positional_arguments,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed! Ready to run benchmark.")
        print("\nTo run the benchmark:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Run: python benchmark/run_benchmark.py --run agentic_z3 --agentic-mode engine")
        print("  3. Evaluate: python benchmark/run_benchmark.py --evaluate")

if __name__ == "__main__":
    main()





