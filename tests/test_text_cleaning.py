"""Checks that the text cleaning actually strips HTML, signatures, replies and personal info."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_analysis.text_cleaning import (  # noqa: E402
    CARD_TOKEN,
    EMAIL_TOKEN,
    PHONE_TOKEN,
    clean_ticket_text,
    redact_pii,
    remove_quoted_reply,
    remove_signature,
    strip_html,
)


def test_strip_html_keeps_line_breaks_and_unescapes():
    text, had = strip_html("<p>Hi,<br>I was charged &amp; billed twice</p>")
    assert had
    assert "<" not in text
    assert "charged & billed" in text
    assert "\n" in text


def test_strip_html_ignores_email_in_angle_brackets():
    text, had = strip_html("Support <help@example.com> wrote:")
    assert not had
    assert "help@example.com" in text


def test_remove_quoted_reply():
    text, removed = remove_quoted_reply(
        "My app crashes.\n\nOn Mon, Aug 3, 2026 at 9:14 AM Support <a@b.com> wrote:\n> We are looking into it."
    )
    assert removed
    assert text.strip() == "My app crashes."


def test_remove_signature_only_after_first_line():
    text, removed = remove_signature("Login broken\nPassword reset never arrives.\n\nThanks,\nSam\nSent from my iPhone")
    assert removed
    assert "Sam" not in text and "iPhone" not in text
    assert "Password reset" in text


def test_redact_pii():
    text, n = redact_pii("Mail sam12@example.com, call 415-555-1234, card 4111 1111 1111 1111.")
    assert n == 3
    assert EMAIL_TOKEN in text and PHONE_TOKEN in text and CARD_TOKEN in text
    assert "4111" not in text and "example.com" not in text


def test_order_numbers_are_not_redacted():
    text, n = redact_pii("Order #A48213 has not arrived")
    assert n == 0 and "A48213" in text


def test_clean_ticket_text_full_chain():
    raw = (
        "<p>Where is my order?<br>Hi,<br>Order #A12345 tracking hasn't updated. Reach me at jo@example.com"
        "<br><br>Thanks,<br>Jo</p>\n\nOn Tue, Aug 4, 2026 at 1:00 PM Support <s@example.com> wrote:\n> old reply"
    )
    r = clean_ticket_text(raw)
    assert r.had_html and r.removed_signature and r.pii_redactions == 1
    assert "old reply" not in r.text and "Jo" not in r.text.split()
    assert "tracking hasn't updated" in r.text
    assert "  " not in r.text and "\n" not in r.text


def test_clean_ticket_text_handles_missing():
    assert clean_ticket_text(None).text == ""
    assert clean_ticket_text(float("nan")).text == ""
