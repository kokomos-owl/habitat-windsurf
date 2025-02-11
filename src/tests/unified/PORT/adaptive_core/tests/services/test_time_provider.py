"""Tests for TimeProvider service."""

import pytest
from datetime import datetime, UTC, timedelta
from ...services.time_provider import TimeProvider

def test_now_returns_utc():
    """Test that now() returns UTC time."""
    now = TimeProvider.now()
    assert now.tzinfo == UTC

def test_parse_valid_timestamp():
    """Test parsing valid ISO timestamp."""
    timestamp = "2025-02-11T08:52:38.123456+00:00"
    dt = TimeProvider.parse(timestamp)
    assert isinstance(dt, datetime)
    assert dt.year == 2025
    assert dt.month == 2
    assert dt.day == 11
    assert dt.hour == 8
    assert dt.minute == 52
    assert dt.second == 38
    assert dt.microsecond == 123456

def test_parse_invalid_timestamp():
    """Test parsing invalid timestamp raises ValueError."""
    with pytest.raises(ValueError):
        TimeProvider.parse("invalid")

def test_format_datetime():
    """Test formatting datetime to ISO string."""
    dt = datetime(2025, 2, 11, 8, 52, 38, 123456, tzinfo=UTC)
    formatted = TimeProvider.format(dt)
    assert formatted == "2025-02-11T08:52:38.123456+00:00"

def test_compare_datetimes():
    """Test datetime comparison with timezone handling."""
    dt1 = datetime(2025, 2, 11, 8, 0, 0, tzinfo=UTC)
    dt2 = datetime(2025, 2, 11, 9, 0, 0, tzinfo=UTC)
    
    assert TimeProvider.compare(dt1, dt2) == -1  # dt1 < dt2
    assert TimeProvider.compare(dt2, dt1) == 1   # dt2 > dt1
    assert TimeProvider.compare(dt1, dt1) == 0   # equal

def test_compare_handles_naive_datetimes():
    """Test comparison with naive (no timezone) datetimes."""
    naive = datetime(2025, 2, 11, 8, 0, 0)  # no timezone
    utc = datetime(2025, 2, 11, 8, 0, 0, tzinfo=UTC)
    
    assert TimeProvider.compare(naive, utc) == 0  # equal after UTC conversion
    assert TimeProvider.compare(utc, naive) == 0

def test_time_consistency():
    """Test time operations maintain consistency."""
    now = TimeProvider.now()
    formatted = TimeProvider.format(now)
    parsed = TimeProvider.parse(formatted)
    
    assert TimeProvider.compare(now, parsed) == 0

def test_monotonic_now():
    """Test that now() is monotonically increasing."""
    t1 = TimeProvider.now()
    t2 = TimeProvider.now()
    assert TimeProvider.compare(t1, t2) <= 0  # t1 <= t2
