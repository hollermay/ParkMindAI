"""
Email notification sender for reservation approval requests.

Uses Python's stdlib smtplib — no extra pip install required.
Sends an HTML email to the administrator with:
  - Reservation details table
  - Clickable Approve / Reject buttons (linked to the REST API)

Falls back to a console log if SMTP settings are not configured.
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import src.config as cfg

logger = logging.getLogger(__name__)

_EMAIL_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body  {{ font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; color: #333; }}
  h2    {{ color: #2c5282; margin-bottom: 4px; }}
  p     {{ margin: 4px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  td    {{ padding: 10px 14px; border: 1px solid #ddd; }}
  td:first-child {{ font-weight: bold; background: #f7fafc; width: 160px; }}
  .actions {{ margin-top: 24px; padding: 20px; background: #ebf8ff;
              border-radius: 8px; text-align: center; }}
  .btn  {{ display: inline-block; padding: 14px 36px; border-radius: 6px;
           color: white; text-decoration: none; font-size: 16px; font-weight: bold;
           margin: 8px; }}
  .approve {{ background: #38a169; }}
  .reject  {{ background: #e53e3e; }}
  .footer  {{ margin-top: 24px; font-size: 12px; color: #888; }}
</style>
</head>
<body>
  <h2>🅿 SmartPark City Center</h2>
  <p>A visitor has submitted a parking reservation request.</p>
  <p>Please review the details and approve or reject:</p>

  <table>
    <tr><td>Request Code</td><td><strong>{code}</strong></td></tr>
    <tr><td>Name</td><td>{first_name} {last_name}</td></tr>
    <tr><td>Email</td><td>{email}</td></tr>
    <tr><td>Vehicle Plate</td><td>{car_number}</td></tr>
    <tr><td>Zone</td><td>Zone {zone}</td></tr>
    <tr><td>Start Date</td><td>{start_date}</td></tr>
    <tr><td>End Date</td><td>{end_date}</td></tr>
    <tr><td>Card</td><td>{card_masked}</td></tr>
  </table>

  <div class="actions">
    <p><strong>Quick Decision:</strong></p>
    <a class="btn approve"
       href="{api_url}/admin/decide?code={code}&amp;decision=approve">
      ✅ Approve
    </a>
    <a class="btn reject"
       href="{api_url}/admin/decide?code={code}&amp;decision=reject">
      ❌ Reject
    </a>
    <p style="margin-top:16px; font-size:13px;">
      Or open the full dashboard:
      <a href="{api_url}/admin">{api_url}/admin</a>
    </p>
  </div>

  <p class="footer">
    SmartPark City Center — 123 Downtown Avenue, City Center
  </p>
</body>
</html>
"""


def send_notification(code: str, reservation_data: dict, api_url: str) -> bool:
    """
    Send an HTML email notification to the administrator.

    Returns True if the email was sent successfully.
    Returns False if SMTP is not configured or sending failed (non-fatal).
    """
    if not cfg.ADMIN_EMAIL or not cfg.SMTP_HOST:
        logger.info(
            "[AdminAgent] Email skipped — SMTP_HOST or ADMIN_EMAIL not set in .env."
        )
        return False

    rd = reservation_data
    subject = (
        f"🅿️ SmartPark Reservation Request [{code}] — "
        f"{rd.get('first_name', '')} {rd.get('last_name', '')} | Zone {rd.get('zone', '')}"
    )
    html_body = _EMAIL_HTML.format(
        code=code,
        first_name=rd.get("first_name", ""),
        last_name=rd.get("last_name", ""),
        email=rd.get("email", "—"),
        car_number=rd.get("car_number", ""),
        zone=rd.get("zone", ""),
        start_date=rd.get("start_date", ""),
        end_date=rd.get("end_date", ""),
        card_masked=rd.get("card_masked", "—"),
        api_url=api_url,
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg.SMTP_FROM
    msg["To"] = cfg.ADMIN_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(cfg.SMTP_HOST, cfg.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            if cfg.SMTP_USE_TLS:
                server.starttls()
                server.ehlo()
            if cfg.SMTP_USER and cfg.SMTP_PASSWORD:
                server.login(cfg.SMTP_USER, cfg.SMTP_PASSWORD)
            server.send_message(msg)
        logger.info(
            "[AdminAgent] Email notification sent to %s (request %s)",
            cfg.ADMIN_EMAIL, code,
        )
        return True
    except Exception as exc:
        logger.error("[AdminAgent] Email send failed: %s", exc)
        return False
