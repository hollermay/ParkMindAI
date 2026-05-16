"""
SQLAlchemy ORM models for dynamic parking data.

Dynamic data is intentionally stored in a SQL database (not the vector store)
so that real-time state (availability, prices, hours) can be updated
without re-indexing the knowledge base.

Supports both PostgreSQL (production) and SQLite (development).
Set DATABASE_URL in .env to connect to PostgreSQL:
    DATABASE_URL=postgresql://user:password@localhost:5432/smartpark
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class ParkingSpace(Base):
    """Individual parking bay record."""

    __tablename__ = "parking_spaces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone = Column(String(10), nullable=False, index=True)        # A, B, C, D, E
    level = Column(String(20), nullable=False)                   # L1, L2 …
    bay_number = Column(String(20), nullable=False)              # A-001, B-042 …
    space_type = Column(String(20), nullable=False)              # standard, premium, ev, disabled
    is_available = Column(Boolean, default=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)    # false = under maintenance


class Pricing(Base):
    """Current pricing per zone and space type."""

    __tablename__ = "pricing"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone = Column(String(10), nullable=False, index=True)
    space_type = Column(String(20), nullable=False)
    hourly_rate = Column(Float, nullable=False)
    daily_max = Column(Float, nullable=False)
    weekly_rate = Column(Float, nullable=False)
    monthly_rate = Column(Float, nullable=False)
    currency = Column(String(5), default="USD")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class WorkingHours(Base):
    """Staffed booth hours (the facility itself is 24/7)."""

    __tablename__ = "working_hours"

    id = Column(Integer, primary_key=True, autoincrement=True)
    day_name = Column(String(15), nullable=False)    # Monday, Tuesday … or Weekday / Weekend
    open_time = Column(String(10), nullable=False)   # HH:MM
    close_time = Column(String(10), nullable=False)  # HH:MM
    is_staffed = Column(Boolean, default=True)
    note = Column(Text, nullable=True)


class Reservation(Base):
    """Guest parking reservation."""

    __tablename__ = "reservations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reservation_code = Column(String(20), unique=True, nullable=False, index=True)
    # Guest details
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=True)                # contact email
    car_number = Column(String(30), nullable=False)           # vehicle registration plate
    card_number = Column(String(50), nullable=True)           # masked card, e.g. **** **** **** 1234
    # Booking details
    zone = Column(String(10), nullable=False)
    space_id = Column(Integer, nullable=True)          # assigned after approval
    start_datetime = Column(DateTime, nullable=False)
    end_datetime = Column(DateTime, nullable=False)
    # State machine
    status = Column(String(20), default="pending", nullable=False)
    # pending | approved | rejected | cancelled | completed
    admin_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class User(Base):
    """Registered client-portal user."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def get_engine(db_url: str):
    """Return a SQLAlchemy engine for the given database URL.

    Accepts:
    - Full SQLAlchemy URLs:  postgresql://user:pass@host/db  or  sqlite:///path
    - Legacy bare file paths (e.g. /data/parking.db) — treated as SQLite.
      (Used only by tests; production must set DATABASE_URL to a PostgreSQL URL.)
    """
    if "://" in db_url:
        if db_url.startswith("postgresql"):
            # PostgreSQL: enable connection health checks, pool recycling, and sizing.
            return create_engine(
                db_url,
                echo=False,
                pool_pre_ping=True,   # validate connections before use
                pool_recycle=300,     # recycle connections after 5 minutes
                pool_size=5,
                max_overflow=10,
            )
        if db_url.startswith("sqlite"):
            # SQLite: allow cross-thread access (needed for tests and dev mode).
            return create_engine(
                db_url,
                echo=False,
                connect_args={"check_same_thread": False},
            )
        return create_engine(db_url, echo=False)
    # Backward-compat: bare file path → SQLite URL (tests only)
    return create_engine(
        f"sqlite:///{db_url}",
        echo=False,
        connect_args={"check_same_thread": False},
    )


def _migrate_schema(engine) -> None:
    """
    Add any missing columns to existing tables (forward-only migrations).

    Works for both PostgreSQL (uses ADD COLUMN IF NOT EXISTS, supported
    since PG 9.6) and SQLite (guards with an existence check instead).
    """
    is_pg = engine.dialect.name == "postgresql"
    inspector = inspect(engine)
    with engine.connect() as conn:
        # ── reservations table ────────────────────────────────────────────────
        if inspector.has_table("reservations"):
            existing = {col["name"] for col in inspector.get_columns("reservations")}
            pending_columns = [
                ("email",       "VARCHAR(255)"),
                ("card_number", "VARCHAR(50)"),
                ("space_id",    "INTEGER"),
                ("admin_notes", "TEXT"),
            ]
            for col_name, col_type in pending_columns:
                if col_name not in existing:
                    if is_pg:
                        conn.execute(text(
                            f"ALTER TABLE reservations "
                            f"ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                        ))
                    else:
                        conn.execute(text(
                            f"ALTER TABLE reservations ADD COLUMN {col_name} {col_type}"
                        ))
            conn.commit()


def init_db(db_url: str) -> None:
    """Create all tables, run forward migrations, and seed initial data if empty."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    _migrate_schema(engine)

    with Session(engine) as session:
        # Re-seed parking spaces if the count doesn't match expected total.
        if session.query(ParkingSpace).count() != TOTAL_SPACES:
            session.query(ParkingSpace).delete()
            _seed_parking_spaces(session)
            # Nullify any space_id references that no longer point to valid rows
            session.execute(text("UPDATE reservations SET space_id = NULL"))
        if session.query(Pricing).count() == 0:
            _seed_pricing(session)
        if session.query(WorkingHours).count() == 0:
            _seed_working_hours(session)
        session.commit()


# ─── Seed helpers ─────────────────────────────────────────────────────────────

# Zone-specific space counts — 500 total (A=80, B=200, C=150, D=50, E=20)
_ZONE_SEED_CONFIG = [
    ("A", "L1", "premium",  80),
    ("B", "L2", "standard", 200),
    ("C", "L4", "standard", 150),
    ("D", "L5", "ev",        50),
    ("E", "L1", "disabled",  20),
]
TOTAL_SPACES = sum(count for _, _, _, count in _ZONE_SEED_CONFIG)  # 500


def _seed_parking_spaces(session: Session) -> None:
    spaces = [
        ParkingSpace(
            zone=zone, level=level,
            bay_number=f"{zone}-{i:03d}",
            space_type=space_type,
        )
        for zone, level, space_type, count in _ZONE_SEED_CONFIG
        for i in range(1, count + 1)
    ]
    session.add_all(spaces)


def _seed_pricing(session: Session) -> None:
    pricing_data = [
        Pricing(zone="A", space_type="premium",  hourly_rate=6.00, daily_max=35.00, weekly_rate=160.00, monthly_rate=500.00),
        Pricing(zone="B", space_type="standard", hourly_rate=3.00, daily_max=20.00, weekly_rate=90.00,  monthly_rate=280.00),
        Pricing(zone="C", space_type="standard", hourly_rate=2.50, daily_max=16.00, weekly_rate=75.00,  monthly_rate=240.00),
        Pricing(zone="D", space_type="ev",        hourly_rate=3.00, daily_max=22.00, weekly_rate=100.00, monthly_rate=310.00),
        Pricing(zone="E", space_type="disabled",  hourly_rate=0.00, daily_max=0.00,  weekly_rate=0.00,   monthly_rate=0.00),
    ]
    session.add_all(pricing_data)


def _seed_working_hours(session: Session) -> None:
    hours = [
        WorkingHours(day_name="Monday",    open_time="07:00", close_time="22:00", is_staffed=True),
        WorkingHours(day_name="Tuesday",   open_time="07:00", close_time="22:00", is_staffed=True),
        WorkingHours(day_name="Wednesday", open_time="07:00", close_time="22:00", is_staffed=True),
        WorkingHours(day_name="Thursday",  open_time="07:00", close_time="22:00", is_staffed=True),
        WorkingHours(day_name="Friday",    open_time="07:00", close_time="22:00", is_staffed=True),
        WorkingHours(day_name="Saturday",  open_time="08:00", close_time="20:00", is_staffed=True),
        WorkingHours(day_name="Sunday",    open_time="09:00", close_time="18:00", is_staffed=True),
    ]
    session.add_all(hours)
