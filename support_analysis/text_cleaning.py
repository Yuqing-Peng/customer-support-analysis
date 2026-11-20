"""Strips HTML, chops off signatures and old quoted
replies, and blanks out emails, phone numbers and card numbers. """
from __future__ import annotations

import html
import re
from dataclasses import dataclass

# Placeholders used for redacted PII. They are single word-tokens so the
# TF-IDF step can drop them as stop words (see features.REDACTION_TOKENS).
EMAIL_TOKEN = "[EMAIL_REDACTED]"
PHONE_TOKEN = "[PHONE_REDACTED]"
CARD_TOKEN = "[CARD_REDACTED]"

_BLOCK_TAG_RE = re.compile(r"<\s*(br|/p|/div|/li|/tr|/h\d)\s*/?\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
_REPLY_HEADER_RE = re.compile(
    r"^(on\s.{3,200}\swrote:|-{2,}\s*original message\s*-{2,}|_{5,})$", re.IGNORECASE
)
_SIGNOFF_RE = re.compile(
    r"^(--\s*|sent from my .+|(?:(?:best|kind|warm)\s+)?regards,?|thanks,|thank you,|thx,|cheers,|best,)$",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,15}\d\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_WS_RE = re.compile(r"\s+")


@dataclass
class TextCleanResult:
    text: str
    had_html: bool = False
    removed_quoted_reply: bool = False
    removed_signature: bool = False
    pii_redactions: int = 0


def strip_html(text: str) -> tuple[str, bool]:
    """Turn block tags into newlines, drop the remaining tags, unescape entities."""
    had_html = bool(_TAG_RE.search(text))
    if had_html:
        text = _BLOCK_TAG_RE.sub("\n", text)
        text = _TAG_RE.sub(" ", text)
    unescaped = html.unescape(text)
    return unescaped, had_html or unescaped != text


def remove_quoted_reply(text: str) -> tuple[str, bool]:
    """Cut everything from an 'On ... wrote:' / 'Original Message' header and drop '>' lines."""
    kept, removed = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if _REPLY_HEADER_RE.match(stripped):
            removed = True
            break
        if stripped.startswith(">"):
            removed = True
            continue
        kept.append(line)
    return "\n".join(kept), removed


def remove_signature(text: str) -> tuple[str, bool]:
    """Cut at the first sign-off line ('Thanks,', 'Best regards', 'Sent from my iPhone', '--')."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if i > 0 and _SIGNOFF_RE.match(line.strip()):
            return "\n".join(lines[:i]), True
    return text, False


def redact_pii(text: str) -> tuple[str, int]:
    """Replace emails, card numbers and phone numbers. Cards before phones (longer digit runs)."""
    text, n_email = _EMAIL_RE.subn(EMAIL_TOKEN, text)
    text, n_card = _CARD_RE.subn(CARD_TOKEN, text)
    text, n_phone = _PHONE_RE.subn(PHONE_TOKEN, text)
    return text, n_email + n_card + n_phone


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_ticket_text(text: object) -> TextCleanResult:
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return TextCleanResult(text="")
    text = str(text)
    text, had_html = strip_html(text)
    text, removed_reply = remove_quoted_reply(text)
    text, removed_sig = remove_signature(text)
    text, n_pii = redact_pii(text)
    return TextCleanResult(
        text=normalize_whitespace(text),
        had_html=had_html,
        removed_quoted_reply=removed_reply,
        removed_signature=removed_sig,
        pii_redactions=n_pii,
    )
