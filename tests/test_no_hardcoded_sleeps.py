"""Test that ensures no hardcoded sleep values exist in critical test files.

This test addresses issue #620 by verifying that hardcoded sleep
statements are not present in Streamlit smoke tests, which could cause
flaky tests.
"""

import re
from pathlib import Path


def test_no_hardcoded_sleeps_in_smoke_tests():
    """Verify that smoke tests don't contain hardcoded sleep statements."""
    smoke_test_files = [
        Path(__file__).parent / "smoke" / "test_app_launch.py",
    ]

    hardcoded_sleep_pattern = re.compile(
        r"time\.sleep\(\s*[0-9]+\s*\)",  # Matches time.sleep(5), time.sleep( 3 ), etc.
        re.IGNORECASE,
    )

    for test_file in smoke_test_files:
        if test_file.exists():
            content = test_file.read_text()

            # Look for hardcoded sleep patterns
            hardcoded_sleeps = hardcoded_sleep_pattern.findall(content)

            assert not hardcoded_sleeps, (
                f"Found hardcoded sleep statements in {test_file}: {hardcoded_sleeps}. "
                f"Use configurable polling mechanisms instead to avoid flaky tests."
            )


def test_no_magic_number_timeouts():
    """Verify that timeouts are configurable rather than magic numbers."""
    smoke_test_file = Path(__file__).parent / "smoke" / "test_app_launch.py"

    if smoke_test_file.exists():
        content = smoke_test_file.read_text()

        # Ensure environment variable configuration is present
        assert (
            "STREAMLIT_STARTUP_TIMEOUT" in content
        ), "Missing configurable STREAMLIT_STARTUP_TIMEOUT environment variable"
        assert (
            "STREAMLIT_POLL_INTERVAL" in content
        ), "Missing configurable STREAMLIT_POLL_INTERVAL environment variable"
        assert (
            "STREAMLIT_READY_TIMEOUT" in content
        ), "Missing configurable STREAMLIT_READY_TIMEOUT environment variable"

        # Ensure sophisticated readiness check function exists
        assert (
            "wait_for_streamlit_ready" in content
        ), "Missing wait_for_streamlit_ready function for sophisticated polling"


def test_sophisticated_readiness_check_features():
    """Verify that the readiness check has sophisticated features."""
    smoke_test_file = Path(__file__).parent / "smoke" / "test_app_launch.py"

    if smoke_test_file.exists():
        content = smoke_test_file.read_text()

        # Check for polling mechanism features
        features_required = [
            "poll_interval",  # Configurable polling interval
            "response.status_code == 200",  # HTTP status check
            "streamlit",  # Content validation
            "timeout",  # Timeout handling
            "requests.get",  # HTTP requests
            "time.time()",  # Time tracking for timeout
        ]

        missing_features = [feature for feature in features_required if feature not in content]

        assert (
            not missing_features
        ), f"Missing sophisticated readiness check features: {missing_features}"


if __name__ == "__main__":
    # Run tests directly
    test_no_hardcoded_sleeps_in_smoke_tests()
    test_no_magic_number_timeouts()
    test_sophisticated_readiness_check_features()
    print("✅ All hardcoded sleep tests passed!")
