"""
CRUD and query operations for all dynamic data.
All writes go through this module so security checks remain centralised.
"""
import logging
import random
import string
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from src.database.models import (
    ParkingSpace,
    Pricing,
    Reservation,
    User,
    WorkingHours,
    get_engine,
)

logger = logging.getLogger(__name__)


def _get_session(db_path: str) -> Session:
    """Return a new Session bound to the shared (cached) engine for db_path."""
    engine = get_engine(db_path)
    return Session(engine)


# ─── Availability ─────────────────────────────────────────────────────────────

def get_availability_summary(db_path: str) -> dict:
    """Return count of available spaces per zone."""
    with _get_session(db_path) as session:
        zones = session.query(ParkingSpace.zone).distinct().all()
        summary = {}
        for (zone,) in zones:
            total = session.query(ParkingSpace).filter_by(zone=zone, is_active=True).count()
            available = session.query(ParkingSpace).filter_by(zone=zone, is_available=True, is_active=True).count()
            summary[zone] = {"total": total, "available": available, "occupied": total - available}
        return summary


def count_available_spaces_for_dates(db_path: str, zone: str, start_dt: datetime, end_dt: datetime) -> int:
    """
    Return how many spaces in *zone* are free for the given date range.

    A space is considered occupied if there is an approved (or
    cancellation_pending) reservation whose dates overlap:
        reservation.start < end_dt  AND  reservation.end > start_dt
    """
    with _get_session(db_path) as session:
        total = session.query(ParkingSpace).filter_by(
            zone=zone.upper(), is_active=True
        ).count()
        if total == 0:
            return 0
        overlapping = (
            session.query(Reservation)
            .filter(
                Reservation.zone == zone.upper(),
                Reservation.status.in_(["approved", "cancellation_pending"]),
                Reservation.start_datetime < end_dt,
                Reservation.end_datetime > start_dt,
            )
            .count()
        )
        return max(0, total - overlapping)


def get_all_zones_availability_for_dates(db_path: str, start_dt: datetime, end_dt: datetime) -> dict:
    """Return {zone: available_count} for all five zones for the given date range."""
    return {
        zone: count_available_spaces_for_dates(db_path, zone, start_dt, end_dt)
        for zone in ("A", "B", "C", "D", "E")
    }


def find_available_space(db_path: str, zone: str) -> Optional[ParkingSpace]:
    """Return the first available space in the given zone, or None."""
    with _get_session(db_path) as session:
        space = (
            session.query(ParkingSpace)
            .filter_by(zone=zone.upper(), is_available=True, is_active=True)
            .first()
        )
        if space:
            session.expunge(space)
        return space


def mark_space_unavailable(db_path: str, space_id: int) -> None:
    with _get_session(db_path) as session:
        space = session.get(ParkingSpace, space_id)
        if space:
            space.is_available = False
            session.commit()


def mark_space_available(db_path: str, space_id: int) -> None:
    with _get_session(db_path) as session:
        space = session.get(ParkingSpace, space_id)
        if space:
            space.is_available = True
            session.commit()


# ─── Pricing ──────────────────────────────────────────────────────────────────

def get_all_pricing(db_path: str) -> list[Pricing]:
    """Return all current pricing records."""
    with _get_session(db_path) as session:
        pricing = session.query(Pricing).all()
        session.expunge_all()
        return pricing


def get_pricing_for_zone(db_path: str, zone: str) -> Optional[Pricing]:
    with _get_session(db_path) as session:
        p = session.query(Pricing).filter_by(zone=zone.upper()).first()
        if p:
            session.expunge(p)
        return p


def format_pricing_context(db_path: str) -> str:
    """Return a human-readable pricing summary for the LLM context."""
    pricing = get_all_pricing(db_path)
    if not pricing:
        return "Pricing information is currently unavailable."

    lines = ["Current Parking Prices at SmartPark City Center:\n"]
    for p in pricing:
        if p.hourly_rate == 0:
            lines.append(
                f"  Zone {p.zone} ({p.space_type}): FREE for valid disability badge holders."
            )
        else:
            lines.append(
                f"  Zone {p.zone} ({p.space_type}): "
                f"${p.hourly_rate:.2f}/hour | ${p.daily_max:.2f} daily maximum | "
                f"${p.weekly_rate:.2f}/week | ${p.monthly_rate:.2f}/month"
            )
    return "\n".join(lines)


# ─── Working Hours ─────────────────────────────────────────────────────────────

def get_working_hours(db_path: str) -> list[WorkingHours]:
    with _get_session(db_path) as session:
        hours = session.query(WorkingHours).all()
        session.expunge_all()
        return hours


def format_hours_context(db_path: str) -> str:
    hours = get_working_hours(db_path)
    lines = [
        "Working Hours for SmartPark City Center:\n",
        "  The parking facility is OPEN 24/7 (automated access always available).\n",
        "  Staffed customer service booth hours:",
    ]
    for h in hours:
        lines.append(f"    {h.day_name}: {h.open_time} – {h.close_time}")
    return "\n".join(lines)


# ─── Reservations ─────────────────────────────────────────────────────────────

def _generate_code() -> str:
    """Generate a unique reservation code like SP-A3K9."""
    chars = random.choices(string.ascii_uppercase + string.digits, k=6)
    return "SP-" + "".join(chars)


def create_reservation(
    db_path: str,
    full_name: str,
    car_number: str,
    zone: str,
    start_datetime: datetime,
    end_datetime: datetime,
    email: str = "",
    card_number: str = "",
    admin_notes: str = "",
) -> Reservation:
    """Create an approved reservation and mark a space as unavailable."""
    with _get_session(db_path) as session:
        # Assign a space
        space = (
            session.query(ParkingSpace)
            .filter_by(zone=zone.upper(), is_available=True, is_active=True)
            .first()
        )
        space_id = space.id if space else None
        if space:
            space.is_available = False

        code = _generate_code()
        reservation = Reservation(
            reservation_code=code,
            full_name=full_name,
            email=email or None,
            car_number=car_number.upper(),
            card_number=card_number or None,
            zone=zone.upper(),
            space_id=space_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            status="approved",
            admin_notes=admin_notes,
        )
        session.add(reservation)
        session.commit()
        session.refresh(reservation)
        session.expunge(reservation)
        return reservation


def reject_reservation_record(
    db_path: str,
    reservation_data: dict,
    admin_notes: str = "Rejected by administrator.",
) -> None:
    """Persist a rejected reservation for audit purposes."""
    with _get_session(db_path) as session:
        code = _generate_code()
        reservation = Reservation(
            reservation_code=code,
            full_name=reservation_data.get("full_name", ""),
            car_number=reservation_data.get("car_number", "").upper(),
            zone=reservation_data.get("zone", ""),
            start_datetime=reservation_data.get("start_datetime"),
            end_datetime=reservation_data.get("end_datetime"),
            status="rejected",
            admin_notes=admin_notes,
        )
        session.add(reservation)
        session.commit()


def get_reservation_by_code(db_path: str, code: str) -> Optional[Reservation]:
    with _get_session(db_path) as session:
        r = session.query(Reservation).filter_by(reservation_code=code.upper()).first()
        if r:
            session.expunge(r)
        return r


# ─── Expired reservation cleanup ───────────────────────────────────────────────────

def cleanup_expired_reservations(db_path: str) -> int:
    """
    Delete approved reservations whose end_datetime is in the past, and
    free up the parking spaces they were holding.

    Returns the number of reservations deleted.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC
    deleted_count = 0

    with _get_session(db_path) as session:
        expired = (
            session.query(Reservation)
            .filter(
                Reservation.status == "approved",
                Reservation.end_datetime < now,
            )
            .all()
        )

        for res in expired:
            # Free up the assigned parking space
            if res.space_id:
                space = session.get(ParkingSpace, res.space_id)
                if space:
                    space.is_available = True

            # Mark reservation as completed (audit trail) then delete
            logger.info(
                "Expiring reservation %s (%s, zone %s, ended %s)",
                res.reservation_code, res.full_name,
                res.zone, res.end_datetime,
            )
            session.delete(res)
            deleted_count += 1

        session.commit()

    if deleted_count:
        logger.info("Cleaned up %d expired reservation(s).", deleted_count)
    return deleted_count


def get_all_reservations(db_path: str) -> list:
    """Return all current (non-expired) reservations — for admin use only."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC
    with _get_session(db_path) as session:
        reservations = (
            session.query(Reservation)
            .filter(Reservation.end_datetime >= now)
            .order_by(Reservation.start_datetime)
            .all()
        )
        session.expunge_all()
        return reservations


def get_all_reservations_full(db_path: str) -> list:
    """Return ALL reservations (approved, rejected, pending, expired) ordered by creation date."""
    with _get_session(db_path) as session:
        reservations = (
            session.query(Reservation)
            .order_by(Reservation.created_at.desc())
            .all()
        )
        session.expunge_all()
        return reservations


# ─── Cancellation ──────────────────────────────────────────────────────────────

FREE_CANCEL_HOURS = 24  # hours after booking creation for free cancellation


def get_cancellation_policy(reservation: Reservation) -> dict:
    """
    Return the cancellation policy for a reservation.

    Rules:
      - Within FREE_CANCEL_HOURS of created_at → free cancellation (100% refund)
      - After FREE_CANCEL_HOURS              → 50% refund
      - Reservation must be in 'approved' status to be cancellable.

    Returns a dict with keys: eligible, refund_pct, reason, free
    """
    if reservation.status not in ("approved", "cancellation_pending"):
        return {
            "eligible": False,
            "refund_pct": 0,
            "free": False,
            "reason": f"Reservation is '{reservation.status}' — only approved reservations can be cancelled.",
        }

    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC
    created = reservation.created_at
    # Normalise to naive UTC — handles timezone-aware datetimes from in-memory
    # test fixtures or timezone=True columns.
    if hasattr(created, "tzinfo") and created.tzinfo is not None:
        created = created.astimezone(timezone.utc).replace(tzinfo=None)

    hours_since_booking = (now - created).total_seconds() / 3600

    if hours_since_booking <= FREE_CANCEL_HOURS:
        return {
            "eligible": True,
            "refund_pct": 100,
            "free": True,
            "reason": (
                f"Booked {hours_since_booking:.1f}h ago — within the "
                f"{FREE_CANCEL_HOURS}h free cancellation window. Full refund applies."
            ),
        }
    else:
        return {
            "eligible": True,
            "refund_pct": 50,
            "free": False,
            "reason": (
                f"Booked {hours_since_booking:.1f}h ago — outside the "
                f"{FREE_CANCEL_HOURS}h free window. 50% refund applies."
            ),
        }


def cancel_reservation(db_path: str, code: str, user_notes: str = "") -> bool:
    """
    Immediately cancel an approved reservation and free the parking space.
    Returns True on success, False if not found or wrong status.
    """
    with _get_session(db_path) as session:
        r = session.query(Reservation).filter_by(reservation_code=code.upper()).first()
        if not r or r.status not in ("approved",):
            return False
        r.status = "cancelled"
        r.admin_notes = (r.admin_notes or "") + f" | Cancelled by user: {user_notes}"
        if r.space_id:
            space = session.get(ParkingSpace, r.space_id)
            if space:
                space.is_available = True
        session.commit()
        logger.info("Reservation %s cancelled directly by user.", code)
        return True


# Keep old name as alias so any existing references don't break
request_cancellation = cancel_reservation


def finalize_cancellation(db_path: str, code: str, admin_approved: bool, admin_notes: str = "") -> bool:
    """
    Admin approves or rejects the cancellation request.
      - approved → status = 'cancelled', free the parking space
      - rejected → status = 'approved' (restored)
    Returns True on success.
    """
    with _get_session(db_path) as session:
        r = session.query(Reservation).filter_by(reservation_code=code.upper()).first()
        if not r or r.status != "cancellation_pending":
            return False

        if admin_approved:
            r.status = "cancelled"
            if r.space_id:
                space = session.get(ParkingSpace, r.space_id)
                if space:
                    space.is_available = True
            r.admin_notes = (r.admin_notes or "") + f" | Cancellation approved: {admin_notes}"
            logger.info("Reservation %s cancelled by admin.", code)
        else:
            r.status = "approved"   # restore
            r.admin_notes = (r.admin_notes or "") + f" | Cancellation rejected: {admin_notes}"
            logger.info("Cancellation of %s rejected by admin.", code)

        session.commit()
        return True


# ─── Web-frontend pending reservation helpers ──────────────────────────────────

def create_pending_reservation(
    db_path: str,
    full_name: str,
    car_number: str,
    zone: str,
    start_datetime: datetime,
    end_datetime: datetime,
    email: str = "",
    card_number: str = "",
) -> Reservation:
    """Create a reservation with status='pending' for web-form submissions."""
    with _get_session(db_path) as session:
        code = _generate_code()
        reservation = Reservation(
            reservation_code=code,
            full_name=full_name,
            email=email or None,
            car_number=car_number.upper(),
            card_number=card_number or None,
            zone=zone.upper(),
            space_id=None,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            status="pending",
        )
        session.add(reservation)
        session.commit()
        session.refresh(reservation)
        session.expunge(reservation)
        return reservation


def approve_pending_reservation(db_path: str, code: str, admin_notes: str = "") -> bool:
    """Approve a DB-pending reservation: assign a space and update status to approved."""
    with _get_session(db_path) as session:
        r = session.query(Reservation).filter_by(
            reservation_code=code.upper(), status="pending"
        ).first()
        if not r:
            return False
        space = (
            session.query(ParkingSpace)
            .filter_by(zone=r.zone, is_available=True, is_active=True)
            .first()
        )
        if space:
            r.space_id = space.id
            space.is_available = False
        r.status = "approved"
        r.admin_notes = admin_notes or "Approved via web dashboard."
        session.commit()
        logger.info("Pending reservation %s approved.", code)
        return True


def reject_pending_reservation(db_path: str, code: str, admin_notes: str = "") -> bool:
    """Reject a DB-pending reservation."""
    with _get_session(db_path) as session:
        r = session.query(Reservation).filter_by(
            reservation_code=code.upper(), status="pending"
        ).first()
        if not r:
            return False
        r.status = "rejected"
        r.admin_notes = admin_notes or "Rejected via web dashboard."
        session.commit()
        logger.info("Pending reservation %s rejected.", code)
        return True


def get_db_pending_reservations(db_path: str) -> list:
    """Return all reservations with status='pending' ordered by creation date."""
    with _get_session(db_path) as session:
        pending = (
            session.query(Reservation)
            .filter_by(status="pending")
            .order_by(Reservation.created_at)
            .all()
        )
        session.expunge_all()
        return pending


# ─── User Authentication ──────────────────────────────────────────────────────

def register_user(
    db_path: str,
    full_name: str,
    email: str,
    password: str,
) -> tuple[bool, str]:
    """
    Create a new user account.

    Returns (True, "") on success or (False, error_message) on failure.
    Uses werkzeug's PBKDF2-HMAC-SHA256 password hashing.
    """
    from werkzeug.security import generate_password_hash

    email = email.strip().lower()
    if not email or not password or not full_name:
        return False, "All fields are required."

    with _get_session(db_path) as session:
        existing = session.query(User).filter_by(email=email).first()
        if existing:
            return False, "An account with this email already exists."

        user = User(
            full_name=full_name.strip(),
            email=email,
            password_hash=generate_password_hash(password),
        )
        session.add(user)
        session.commit()
        return True, ""


def authenticate_user(db_path: str, email: str, password: str) -> Optional[User]:
    """
    Verify credentials and return the User object on success, or None on failure.
    """
    from werkzeug.security import check_password_hash

    email = email.strip().lower()
    with _get_session(db_path) as session:
        user = session.query(User).filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session.expunge(user)
            return user
        return None


def get_user_by_id(db_path: str, user_id: int) -> Optional[User]:
    with _get_session(db_path) as session:
        user = session.get(User, user_id)
        if user:
            session.expunge(user)
        return user


def get_reservations_for_user(db_path: str, email: str) -> list:
    """Return all reservations associated with the given email, newest first."""
    email = email.strip().lower()
    with _get_session(db_path) as session:
        reservations = (
            session.query(Reservation)
            .filter(Reservation.email == email)
            .order_by(Reservation.created_at.desc())
            .all()
        )
        session.expunge_all()
        return reservations
