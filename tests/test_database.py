"""
Tests for the database models and CRUD operations.

Covers: DB initialisation, seeding, availability queries,
pricing retrieval, reservation creation/rejection, and
working hours.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_db(tmp_path):
    """Provide a fresh, seeded SQLite database per test function."""
    db_path = str(tmp_path / "test.db")
    from src.database.models import init_db
    init_db(db_path)
    return db_path


# ─── Initialisation and seeding ───────────────────────────────────────────────

class TestDBInitialisation:
    def test_init_creates_tables(self, fresh_db):
        from sqlalchemy import inspect as sa_inspect

        from src.database.models import get_engine
        engine = get_engine(fresh_db)
        inspector = sa_inspect(engine)
        tables = inspector.get_table_names()
        assert "parking_spaces" in tables
        assert "pricing" in tables
        assert "working_hours" in tables
        assert "reservations" in tables

    def test_seeds_parking_spaces(self, fresh_db):
        from sqlalchemy.orm import Session

        from src.database.models import ParkingSpace, get_engine
        with Session(get_engine(fresh_db)) as session:
            count = session.query(ParkingSpace).count()
        assert count == 500  # 80+200+150+50+20

    def test_seeds_pricing_rows(self, fresh_db):
        from sqlalchemy.orm import Session

        from src.database.models import Pricing, get_engine
        with Session(get_engine(fresh_db)) as session:
            count = session.query(Pricing).count()
        assert count == 5  # one per zone

    def test_seeds_working_hours(self, fresh_db):
        from sqlalchemy.orm import Session

        from src.database.models import WorkingHours, get_engine
        with Session(get_engine(fresh_db)) as session:
            count = session.query(WorkingHours).count()
        assert count == 7  # one per day

    def test_init_idempotent(self, fresh_db):
        """Calling init_db twice should not duplicate seed data."""
        from sqlalchemy.orm import Session

        from src.database.models import ParkingSpace, get_engine, init_db
        init_db(fresh_db)
        with Session(get_engine(fresh_db)) as session:
            count = session.query(ParkingSpace).count()
        assert count == 500


# ─── Availability ─────────────────────────────────────────────────────────────

class TestAvailability:
    def test_get_availability_summary(self, fresh_db):
        from src.database.operations import get_availability_summary
        summary = get_availability_summary(fresh_db)
        assert "A" in summary and "B" in summary
        assert summary["A"]["total"] == 80
        assert summary["A"]["available"] == 80  # all available initially

    def test_find_available_space_exists(self, fresh_db):
        from src.database.operations import find_available_space
        space = find_available_space(fresh_db, "A")
        assert space is not None
        assert space.zone == "A"
        assert space.is_available is True

    def test_find_available_space_invalid_zone(self, fresh_db):
        from src.database.operations import find_available_space
        space = find_available_space(fresh_db, "Z")
        assert space is None

    def test_mark_space_unavailable(self, fresh_db):
        from src.database.operations import (
            find_available_space,
            get_availability_summary,
            mark_space_unavailable,
        )
        space = find_available_space(fresh_db, "A")
        assert space is not None
        mark_space_unavailable(fresh_db, space.id)
        summary = get_availability_summary(fresh_db)
        assert summary["A"]["available"] == 79  # one less

    def test_mark_space_available_again(self, fresh_db):
        from src.database.operations import (
            find_available_space,
            get_availability_summary,
            mark_space_available,
            mark_space_unavailable,
        )
        space = find_available_space(fresh_db, "B")
        mark_space_unavailable(fresh_db, space.id)
        mark_space_available(fresh_db, space.id)
        summary = get_availability_summary(fresh_db)
        assert summary["B"]["available"] == 200  # restored


# ─── Pricing ──────────────────────────────────────────────────────────────────

class TestPricing:
    def test_get_all_pricing(self, fresh_db):
        from src.database.operations import get_all_pricing
        pricing = get_all_pricing(fresh_db)
        assert len(pricing) == 5
        zones = {p.zone for p in pricing}
        assert zones == {"A", "B", "C", "D", "E"}

    def test_get_pricing_for_zone_a(self, fresh_db):
        from src.database.operations import get_pricing_for_zone
        p = get_pricing_for_zone(fresh_db, "A")
        assert p is not None
        assert p.hourly_rate == 6.00
        assert p.daily_max == 35.00

    def test_get_pricing_for_zone_e_free(self, fresh_db):
        from src.database.operations import get_pricing_for_zone
        p = get_pricing_for_zone(fresh_db, "E")
        assert p is not None
        assert p.hourly_rate == 0.00

    def test_format_pricing_context(self, fresh_db):
        from src.database.operations import format_pricing_context
        ctx = format_pricing_context(fresh_db)
        assert "Zone A" in ctx
        assert "$6.00" in ctx
        assert "FREE" in ctx  # Zone E

    def test_format_hours_context(self, fresh_db):
        from src.database.operations import format_hours_context
        ctx = format_hours_context(fresh_db)
        assert "07:00" in ctx  # Monday open time
        assert "24/7" in ctx


# ─── Reservations ─────────────────────────────────────────────────────────────

class TestReservations:
    def test_create_reservation_returns_code(self, fresh_db):
        from src.database.operations import create_reservation
        r = create_reservation(
            db_path=fresh_db,
            full_name="Alice Smith",
            car_number="ABC-1234",
            zone="B",
            start_datetime=datetime(2026, 7, 1, 10, 0),
            end_datetime=datetime(2026, 7, 5, 10, 0),
        )
        assert r.reservation_code.startswith("SP-")
        assert r.status == "approved"
        assert r.full_name == "Alice Smith"

    def test_create_reservation_marks_space_unavailable(self, fresh_db):
        from src.database.operations import create_reservation, get_availability_summary
        before = get_availability_summary(fresh_db)["C"]["available"]
        create_reservation(
            db_path=fresh_db,
            full_name="Bob Jones",
            car_number="XY-5678",
            zone="C",
            start_datetime=datetime(2026, 8, 1),
            end_datetime=datetime(2026, 8, 3),
        )
        after = get_availability_summary(fresh_db)["C"]["available"]
        assert after == before - 1

    def test_get_reservation_by_code(self, fresh_db):
        from src.database.operations import create_reservation, get_reservation_by_code
        r = create_reservation(
            db_path=fresh_db,
            full_name="Carol White",
            car_number="DEF-9999",
            zone="A",
            start_datetime=datetime(2026, 9, 1),
            end_datetime=datetime(2026, 9, 2),
        )
        fetched = get_reservation_by_code(fresh_db, r.reservation_code)
        assert fetched is not None
        assert fetched.full_name == "Carol White"

    def test_get_reservation_nonexistent_code(self, fresh_db):
        from src.database.operations import get_reservation_by_code
        r = get_reservation_by_code(fresh_db, "SP-XXXXXX")
        assert r is None

    def test_reject_reservation_record(self, fresh_db):
        from sqlalchemy.orm import Session

        from src.database.models import Reservation, get_engine
        from src.database.operations import reject_reservation_record
        reject_reservation_record(
            db_path=fresh_db,
            reservation_data={
                "full_name": "Dave Brown",
                "car_number": "GHI-0000",
                "zone": "B",
                "start_datetime": datetime(2026, 10, 1),
                "end_datetime": datetime(2026, 10, 2),
            },
            admin_notes="Zone B full.",
        )
        with Session(get_engine(fresh_db)) as session:
            r = session.query(Reservation).filter_by(status="rejected").first()
        assert r is not None
        assert r.full_name == "Dave Brown"
