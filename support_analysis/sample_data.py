"""Makes up fake support tickets for the demo. Six kinds of issues, purposely messy
(HTML, signatures, phone numbers, bad dates, duplicates...) so the cleaning has work
to do, plus an app-crash spike near the end for the insights to catch."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

THEMES = {
    "billing": {
        "product": "billing portal", "res_hours": 10, "csat": 3.4, "priority": [0.2, 0.4, 0.3, 0.1],
        "subjects": ["Charged twice this month", "Double charge on my card", "Wrong amount on invoice", "Billing error on latest invoice"],
        "bodies": [
            "I was charged twice for my {plan} subscription this month. The invoice shows ${amount} two times. Please refund the duplicate charge.",
            "My credit card statement shows two identical payments of ${amount} for the same invoice. Why was I billed twice?",
            "The invoice amount is ${amount} but my plan is {plan}. I think the billing is wrong, can you correct the invoice?",
            "We got charged after we downgraded to {plan}. The payment of ${amount} should not have gone through.",
        ],
    },
    "login": {
        "product": "web app", "res_hours": 3, "csat": 4.2, "priority": [0.1, 0.3, 0.5, 0.1],
        "subjects": ["Can't log in", "Password reset email never arrives", "Locked out of my account", "Two-factor code not working"],
        "bodies": [
            "I can't log in to my account. The password reset email never arrives, I checked spam too.",
            "My account is locked after too many login attempts. How do I unlock it?",
            "The two-factor authentication code is always rejected as invalid when I try to sign in.",
            "I reset my password but the new password doesn't work and the login page keeps saying invalid credentials.",
        ],
    },
    "shipping": {
        "product": "orders", "res_hours": 20, "csat": 3.1, "priority": [0.15, 0.35, 0.4, 0.1],
        "subjects": ["Where is my order?", "Package hasn't arrived", "Tracking not updating", "Delivery is late"],
        "bodies": [
            "My order #{order} was supposed to arrive last week but the tracking number hasn't updated in 6 days.",
            "The package for order #{order} still shows in transit. The delivery date has passed. Where is my shipment?",
            "Tracking says delivered but I never received the package for order #{order}.",
            "Shipping is very delayed for order #{order}. Can you check with the carrier?",
        ],
    },
    "app_crash": {
        "product": "mobile app", "res_hours": 40, "csat": 2.6, "priority": [0.05, 0.2, 0.45, 0.3],
        "subjects": ["App crashes on startup", "Mobile app keeps crashing", "App freezes after update", "Crash when uploading photo"],
        "bodies": [
            "Since the latest update the mobile app crashes on startup on my {device}. I reinstalled it and it still crashes.",
            "The app freezes and then closes every time I open the dashboard. Using {device}, latest app version.",
            "App crashes when I try to upload a photo. This started after the update. Device: {device}.",
            "After updating the app it shows a blank screen and crashes. {device}, version 5.2.",
        ],
    },
    "refund": {
        "product": "orders", "res_hours": 15, "csat": 3.6, "priority": [0.2, 0.45, 0.3, 0.05],
        "subjects": ["Refund request", "Return and refund status", "Still waiting for my refund", "Cancel order and refund"],
        "bodies": [
            "I returned the item from order #{order} two weeks ago and still haven't received my refund of ${amount}.",
            "Please cancel order #{order} and refund ${amount} to my original payment method.",
            "The return was accepted but the refund status has been pending for 10 days. When will I get my money back?",
            "I want to return a damaged item from order #{order} and get a full refund.",
        ],
    },
    "feature_request": {
        "product": "web app", "res_hours": 30, "csat": 4.0, "priority": [0.6, 0.3, 0.1, 0.0],
        "subjects": ["Feature request: export to Excel", "Suggestion: dark mode", "Would love a calendar integration", "Request: bulk edit"],
        "bodies": [
            "It would be great if we could export reports to Excel or CSV directly from the dashboard.",
            "Any plans to add dark mode? It would really help when working late.",
            "Feature request: integrate with Google Calendar so tasks show up as events.",
            "We need a bulk edit option to update many records at once. Is this on the roadmap?",
        ],
    },
}

PRIORITY_LABELS = [["low", "Low", "P4"], ["medium", "Medium", "Normal", "P3"], ["high", "High", "P2"], ["urgent", "Urgent", "P1", "critical"]]
CHANNELS = ["email", "chat", "web_form", "phone"]
DEVICES = ["iPhone 15", "Pixel 8", "Galaxy S24", "iPad Air"]
PLANS = ["Basic", "Pro", "Team", "Enterprise"]
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Jamie"]


def make_sample(n: int = 2000, random_state: int = 42, start: str = "2026-08-01", weeks: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    theme_names = list(THEMES)
    base_p = np.array([0.20, 0.18, 0.18, 0.10, 0.16, 0.18])
    start_dt = datetime.fromisoformat(start)
    total_minutes = weeks * 7 * 24 * 60

    rows = []
    for i in range(n):
        created = start_dt + timedelta(minutes=int(rng.integers(0, total_minutes)))
        week = (created - start_dt).days // 7
        p = base_p.copy()
        if week >= weeks - 3:  # simulated bad mobile release
            p[theme_names.index("app_crash")] *= 3
        theme_key = rng.choice(theme_names, p=p / p.sum())
        t = THEMES[theme_key]

        fill = {
            "order": f"A{rng.integers(10000, 99999)}", "amount": f"{rng.integers(9, 400)}.{rng.integers(0, 99):02d}",
            "plan": rng.choice(PLANS), "device": rng.choice(DEVICES),
        }
        subject = rng.choice(t["subjects"])
        body = rng.choice(t["bodies"]).format(**fill)
        name = rng.choice(NAMES)
        body = f"Hi,\n{body}"

        if rng.random() < 0.25:  # PII
            body += f"\nYou can reach me at {name.lower()}{rng.integers(1, 99)}@example.com or {rng.integers(200, 999)}-555-{rng.integers(1000, 9999)}."
        if theme_key == "billing" and rng.random() < 0.15:
            body += "\nThe card used was 4111 1111 1111 1111."
        if rng.random() < 0.45:  # signature
            body += f"\n\nThanks,\n{name}" + ("\nSent from my iPhone" if rng.random() < 0.3 else "")
        if rng.random() < 0.10:  # quoted previous reply
            body += "\n\nOn Mon, Aug 3, 2026 at 9:14 AM Support <help@example.com> wrote:\n> Thanks for reaching out, we are looking into it.\n> Ticket reference attached."
        if rng.random() < 0.15:  # HTML export
            body = "<p>" + body.replace("&", "&amp;").replace("\n", "<br>") + "</p>"

        prio_idx = rng.choice(4, p=t["priority"])
        status = rng.choice(["resolved", "open", "escalated"], p=[0.82, 0.08, 0.10] if theme_key != "app_crash" else [0.68, 0.08, 0.24])
        res_h = rng.gamma(2.0, t["res_hours"] / 2.0)
        resolved = created + timedelta(hours=float(res_h)) if status != "open" else None
        csat = float(np.clip(round(rng.normal(t["csat"], 0.8)), 1, 5)) if (status == "resolved" and rng.random() < 0.6) else np.nan

        rows.append({
            "Ticket ID": f"T-{100000 + i}",
            "Created At": created.strftime("%Y-%m-%d %H:%M:%S") if rng.random() > 0.05 else created.strftime("%m/%d/%Y %I:%M %p"),
            "Resolved At": resolved.strftime("%Y-%m-%d %H:%M:%S") if resolved else "",
            "Customer ID": f"C-{rng.integers(1000, 1600)}",
            "Channel": rng.choice(CHANNELS),
            "Priority": " " + rng.choice(PRIORITY_LABELS[prio_idx]) + " ",
            "Product": t["product"].title(),
            "Status": status.title(),
            "Subject": subject,
            "Body": body,
            "CSAT": csat,
        })

    df = pd.DataFrame(rows)

    # --- Inject data-quality problems ---
    m = len(df)
    bad = rng.choice(m, size=max(1, m // 100), replace=False)
    df.loc[bad, "Created At"] = "not a date"
    neg = rng.choice(m, size=max(1, m // 100), replace=False)
    df.loc[neg, "Resolved At"] = "2026-01-01 00:00:00"  # resolved before created
    out_of_range = rng.choice(m, size=max(1, m // 200), replace=False)
    df.loc[out_of_range, "CSAT"] = rng.choice([0, 10], size=len(out_of_range))
    empty = rng.choice(m, size=max(1, m // 100), replace=False)
    df.loc[empty, "Subject"] = ""
    df.loc[empty, "Body"] = "  "
    exact_dups = df.sample(n=max(1, m // 50), random_state=random_state)                 # same ticket exported twice
    content_dups = df.sample(n=max(1, m // 100), random_state=random_state + 1).copy()   # same customer re-submitted
    content_dups["Ticket ID"] = [f"T-9{j:05d}" for j in range(len(content_dups))]
    df = pd.concat([df, exact_dups, content_dups], ignore_index=True)
    df["Empty Column"] = np.nan
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
