"""Generate a realistic synthetic dataset for the signal detector.

Six customer archetypes, so the demo can show that the detector catches the
*quiet* decliners (usage + billing drift, no complaints yet) and not just the
customers who are already shouting.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

random.seed(7)
TODAY = date(2026, 8, 27)
OUT = Path(__file__).parent / "data"

FIRST = "Priya Marcus Aisha Daniel Lena Tomas Ravi Sofia Omar Chen Nadia Ellis Kwame Ingrid Hugo Mei Jonas Farida Leo Anika".split()
LAST = "Shah Delgado Okafor Lindqvist Moreau Nakamura Bianchi Adeyemi Novak Halvorsen Reyes Kaur Fitzgerald Osei Petrov".split()
COMPANY = "Northwind Vertex Harbor Cobalt Ironwood Lumen Pacific Kestrel Arbor Solstice Fairmont Quill Basalt Meridian Halcyon".split()
SUFFIX = ["Group", "Labs", "Systems", "Health", "Retail", "Logistics", "Media"]
SEGMENTS = ["Enterprise", "Mid-Market", "SMB"]

SCRIPTS = {
    "healthy": [
        ("chat", "Quick one - is there a way to export the weekly report as CSV? Not urgent."),
        ("email", "Thanks for the fast turnaround on the SSO config last week, the team is happy."),
        ("chat", "We are onboarding 12 more people next month, anything we should prep?"),
        ("survey", "Works well for us. Support has been responsive. Would recommend."),
    ],
    "quiet_decline": [
        ("chat", "Hi, we have not been using the dashboards much lately - is there a way to pause seats?"),
        ("email", "Our team restructured and the analytics owner left. Nobody has picked it back up yet."),
        ("survey", "It is fine. Honestly we are not getting to it much these days."),
        ("chat", "Can you send over what our current seat count and utilisation looks like?"),
    ],
    "billing_friction": [
        ("email", "We were charged twice on the July invoice again. This is the second month running."),
        ("chat", "The card on file keeps failing and nobody told us until the account was suspended."),
        ("email", "Finance is asking why the renewal quote went up 22% with no notice. Please explain the uplift."),
        ("chat", "Still waiting on the credit note from the duplicate charge. Any update?"),
    ],
    "support_strain": [
        ("chat", "The sync has failed four times this week. Same error as ticket 44812 which was marked resolved."),
        ("email", "Reopening this. The fix you shipped did not hold - we are back to the same failure."),
        ("chat", "This is now blocking our month-end close. Can someone senior look at it today?"),
        ("email", "Third time raising this. I do not feel like anyone owns the problem on your side."),
        ("survey", "Frustrating. Issues get closed before they are actually fixed."),
    ],
    "churn_intent": [
        ("email", "We are evaluating alternatives ahead of the December renewal. Please send our data export options."),
        ("chat", "Leadership has asked what it would take to cancel. What is the notice period in our contract?"),
        ("email", "Being straight with you - a competitor demoed something that solves this out of the box."),
        ("survey", "Unlikely to renew unless the reliability issues are fixed. We have lost confidence."),
    ],
    "onboarding_stall": [
        ("chat", "We signed 6 weeks ago and still have not got the data import working. Who owns this?"),
        ("email", "The implementation call keeps getting rescheduled. We have not gone live with a single team."),
        ("chat", "Is there documentation for the API auth? The guide we were sent is out of date."),
        ("survey", "Rocky start. Not what we expected after the sales process."),
    ],
}

# archetype -> usage ratio, ticket ratio, reopens, failed pay, disputes, overdue days,
#              csat now, csat prev, open p1, P(downgrade)
PROFILE = {
    "healthy":          ((0.95, 1.25), (0.6, 1.1), (0, 0), (0, 0), (0, 0), (0, 0),   (4.2, 5.0), (4.0, 5.0), (0, 0), 0.0),
    "quiet_decline":    ((0.18, 0.45), (0.3, 0.8), (0, 1), (0, 1), (0, 0), (0, 12),  (3.2, 4.0), (4.1, 4.8), (0, 0), 0.35),
    "billing_friction": ((0.70, 1.00), (1.1, 1.8), (0, 1), (2, 4), (1, 3), (14, 62), (2.8, 3.6), (3.9, 4.6), (0, 1), 0.15),
    "support_strain":   ((0.60, 0.95), (2.2, 4.0), (2, 5), (0, 1), (0, 1), (0, 8),   (2.0, 3.0), (3.8, 4.5), (1, 3), 0.10),
    "churn_intent":     ((0.25, 0.60), (1.5, 3.0), (1, 4), (0, 2), (0, 2), (0, 30),  (1.4, 2.4), (3.5, 4.4), (0, 2), 0.30),
    "onboarding_stall": ((0.10, 0.35), (1.4, 2.6), (1, 3), (0, 1), (0, 1), (0, 10),  (2.4, 3.4), (3.6, 4.3), (0, 1), 0.05),
}

MIX = (["healthy"] * 14 + ["quiet_decline"] * 6 + ["billing_friction"] * 5
       + ["support_strain"] * 5 + ["churn_intent"] * 4 + ["onboarding_stall"] * 4)


def rr(lo, hi, nd=2):
    return round(random.uniform(lo, hi), nd)


def main(out: Path = OUT):
    random.seed(7)
    random.shuffle(MIX)
    customers, interactions = [], []
    names = [f"{c} {s}" for c in COMPANY for s in SUFFIX]
    random.shuffle(names)   # account names must be unique - ops identifies by name

    for i, arch in enumerate(MIX, start=1):
        uz, tz, rz, fz, dz, oz, csat, prev, p1z, down = PROFILE[arch]
        cid = f"C{1000 + i}"
        seg = random.choices(SEGMENTS, weights=[3, 4, 5])[0]
        base_logins = {"Enterprise": (400, 1400), "Mid-Market": (150, 500), "SMB": (30, 160)}[seg]
        prev_logins = random.randint(*base_logins)
        prev_tickets = max(1, int(random.gauss({"Enterprise": 9, "Mid-Market": 5, "SMB": 2}[seg], 2)))
        seats = random.randint(10, 600)
        near_renewal = arch in ("churn_intent", "quiet_decline")

        customers.append({
            "customer_id": cid,
            "account_name": names.pop(),
            "primary_contact": f"{random.choice(FIRST)} {random.choice(LAST)}",
            "segment": seg,
            "plan": random.choice(["Starter", "Growth", "Business", "Enterprise"]),
            "arr_usd": {
                "Enterprise": random.randrange(120_000, 900_000, 5_000),
                "Mid-Market": random.randrange(24_000, 120_000, 1_000),
                "SMB": random.randrange(3_000, 24_000, 500),
            }[seg],
            "tenure_months": random.randint(2, 8) if arch == "onboarding_stall" else random.randint(9, 74),
            "renewal_date": (TODAY + timedelta(
                days=random.randint(20, 70) if near_renewal else random.randint(45, 330))).isoformat(),
            "seats_licensed": seats,
            "seats_active": min(seats, max(1, int(seats * rr(*uz)))),
            "logins_prev_30d": prev_logins,
            "logins_last_30d": max(0, int(prev_logins * rr(*uz))),
            "feature_adoption_pct": round(min(0.95, max(0.05, rr(*uz) * 0.7)), 2),
            "tickets_prev_30d": prev_tickets,
            "tickets_last_30d": max(0, int(prev_tickets * rr(*tz))),
            "tickets_reopened_30d": random.randint(*rz),
            "open_p1_tickets": random.randint(*p1z),
            "avg_first_response_hrs": rr(1.0, 6.0) if arch == "healthy" else rr(4.0, 30.0),
            "csat_current": rr(*csat, nd=1),
            "csat_prev_quarter": rr(*prev, nd=1),
            "nps_last": random.randint(20, 80) if arch == "healthy" else random.randint(-60, 10),
            "failed_payments_90d": random.randint(*fz),
            "invoice_disputes_90d": random.randint(*dz),
            "days_payment_overdue": random.randint(*oz),
            "downgraded_last_90d": int(random.random() < down),
            "_archetype": arch,  # ground truth, held out of the model input - eval only
        })

        msgs = SCRIPTS[arch][:]
        random.shuffle(msgs)
        for n, (ch, txt) in enumerate(msgs[:random.randint(3, len(msgs))]):
            interactions.append({
                "customer_id": cid,
                "date": (TODAY - timedelta(days=random.randint(1, 40) + n * 3)).isoformat(),
                "channel": ch,
                "text": txt,
            })

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(customers).to_csv(out / "customers.csv", index=False)
    pd.DataFrame(interactions).sort_values(["customer_id", "date"]).to_csv(
        out / "interactions.csv", index=False)
    print(f"{len(customers)} customers, {len(interactions)} interactions -> {out}")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
