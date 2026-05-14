"""
Admin Notification Agent — second LangChain agent in the SmartPark system.

Responsibilities:
  - Receive reservation approval requests from the first (user-facing) chatbot agent
  - Notify the human administrator via email and/or REST API
  - Wait for the administrator's approve / reject decision
  - Return the decision to the first agent so it can finalise the reservation

Communication channels:
  1. Email (SMTP) — sends an HTML email with clickable Approve / Reject buttons
  2. REST API   — Flask server at http://localhost:5001/admin where the admin
                  can view pending requests and submit decisions via browser or curl
"""
