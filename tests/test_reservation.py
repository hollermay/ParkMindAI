"""
Tests for the reservation handler (field extraction and validation logic).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reservation.handler import (
    extract_field_value,
    get_next_field,
    is_complete,
    validate_date_range,
    RESERVATION_FIELD_ORDER,
)


# ─── Field extraction ─────────────────────────────────────────────────────────

class TestNameExtraction:
    @pytest.mark.parametrize("raw, expected", [
        ("Alice", "Alice"),
        ("alice smith", "Alice Smith"),
        ("JOHN  DOE", "John  Doe"),
        ("O'Brien", "O'Brien"),
        ("Anne-Marie", "Anne-Marie"),
    ])
    def test_valid_names(self, raw, expected):
        for field in ("first_name", "last_name"):
            value, error = extract_field_value(field, raw)
            assert error is None, f"Unexpected error for '{raw}': {error}"
            assert value == expected

    @pytest.mark.parametrize("raw", ["12345", "!@#$", ""])
    def test_invalid_names(self, raw):
        value, error = extract_field_value("first_name", raw)
        assert value is None or error is not None


class TestCarNumberExtraction:
    @pytest.mark.parametrize("raw, expected_upper", [
        ("ABC-1234", "ABC-1234"),
        ("abc 1234", "ABC1234"),
        ("XY12ABC", "XY12ABC"),
        ("KA01AB1234", "KA01AB1234"),
    ])
    def test_valid_car_numbers(self, raw, expected_upper):
        value, error = extract_field_value("car_number", raw)
        assert error is None, f"Unexpected error for '{raw}': {error}"
        assert value == value.upper()

    @pytest.mark.parametrize("raw", ["A", "!@#$%^", ""])
    def test_invalid_car_numbers(self, raw):
        value, error = extract_field_value("car_number", raw)
        assert error is not None or value is None


class TestZoneExtraction:
    @pytest.mark.parametrize("raw, expected", [
        ("A", "A"), ("b", "B"), ("Zone C", "C"), ("ZONE D", "D"), (" e ", "E"),
    ])
    def test_valid_zones(self, raw, expected):
        value, error = extract_field_value("zone", raw)
        assert error is None and value == expected, f"Failed for raw={raw!r}"

    @pytest.mark.parametrize("raw", ["F", "Zone X", "1", ""])
    def test_invalid_zones(self, raw):
        value, error = extract_field_value("zone", raw)
        assert error is not None


class TestDateExtraction:
    @pytest.mark.parametrize("raw", [
        "2027-06-01",
        "01/06/2027",
        "01-06-2027",
        "01.06.2027",
    ])
    def test_valid_future_dates(self, raw):
        value, error = extract_field_value("start_date", raw)
        assert error is None and value == "2027-06-01", f"Failed for raw={raw!r}: value={value}, error={error}"

    def test_past_date_rejected(self):
        value, error = extract_field_value("start_date", "2020-01-01")
        assert error is not None and "past" in error.lower()

    def test_invalid_date_format_rejected(self):
        value, error = extract_field_value("start_date", "not-a-date")
        assert error is not None


# ─── Field ordering ───────────────────────────────────────────────────────────

class TestFieldOrdering:
    def test_next_field_empty_data(self):
        assert get_next_field({}) == "first_name"

    def test_next_field_partial(self):
        rd = {"first_name": "Alice", "last_name": "Smith"}
        assert get_next_field(rd) == "car_number"

    def test_next_field_all_filled(self):
        rd = {
            "first_name": "Alice",
            "last_name": "Smith",
            "car_number": "ABC-1234",
            "zone": "B",
            "start_date": "2027-07-01",
            "end_date": "2027-07-05",
        }
        assert get_next_field(rd) is None

    def test_is_complete_false_partial(self):
        rd = {"first_name": "Alice"}
        assert not is_complete(rd)

    def test_is_complete_true_full(self):
        rd = {
            "first_name": "Alice",
            "last_name": "Smith",
            "car_number": "ABC-1234",
            "zone": "A",
            "start_date": "2027-07-01",
            "end_date": "2027-07-03",
        }
        assert is_complete(rd)


# ─── Date range validation ────────────────────────────────────────────────────

class TestDateRangeValidation:
    def test_valid_range(self):
        rd = {"start_date": "2027-07-01", "end_date": "2027-07-10"}
        valid, msg = validate_date_range(rd)
        assert valid and not msg

    def test_end_before_start_rejected(self):
        rd = {"start_date": "2027-07-10", "end_date": "2027-07-01"}
        valid, msg = validate_date_range(rd)
        assert not valid
        assert "after" in msg.lower()

    def test_same_day_rejected(self):
        rd = {"start_date": "2027-07-01", "end_date": "2027-07-01"}
        valid, msg = validate_date_range(rd)
        assert not valid

    def test_exceeds_30_days_rejected(self):
        rd = {"start_date": "2027-07-01", "end_date": "2027-09-15"}
        valid, msg = validate_date_range(rd)
        assert not valid
        assert "30" in msg

    def test_exact_30_days_valid(self):
        rd = {"start_date": "2027-07-01", "end_date": "2027-07-31"}
        valid, msg = validate_date_range(rd)
        assert valid
