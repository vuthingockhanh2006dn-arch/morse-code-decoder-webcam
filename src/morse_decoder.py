"""Morse code dictionary lookup and text buffer.

This part needs no machine learning: a confirmed sequence of dots and dashes is
looked up in the International Morse Code table.
"""
from __future__ import annotations

from typing import Dict, Optional

MORSE_TO_CHAR: Dict[str, str] = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
}
CHAR_TO_MORSE: Dict[str, str] = {v: k for k, v in MORSE_TO_CHAR.items()}

MAX_SYMBOLS = 5      # longest code in the table above
UNKNOWN_CHAR = "?"   # written when a confirmed code is not in the table


def decode_symbol(code: str) -> Optional[str]:
    """Return the character for one Morse code (e.g. ``"..."`` -> ``"S"``)."""
    return MORSE_TO_CHAR.get(code)


def decode_message(message: str) -> str:
    """Decode a full message: letters separated by spaces, words by ``" / "``."""
    words = []
    for word in message.strip().split(" / "):
        words.append("".join(MORSE_TO_CHAR.get(code, UNKNOWN_CHAR) for code in word.split()))
    return " ".join(words)


def encode_message(text: str) -> str:
    """Inverse of :func:`decode_message`; handy for tests and the guide panel."""
    words = []
    for word in text.upper().split():
        words.append(" ".join(CHAR_TO_MORSE[c] for c in word if c in CHAR_TO_MORSE))
    return " / ".join(words)


class MorseBuffer:
    """Holds the Morse symbols being typed and the confirmed text.

    Event semantics:
      * ``DOT`` / ``DASH`` append ``.`` / ``-`` to the current buffer
        (ignored once the buffer already holds ``MAX_SYMBOLS`` symbols).
      * ``CONFIRM`` decodes the buffer into one character (``?`` if unknown).
      * ``SPACE`` decodes any pending buffer first, then adds a single space.
      * ``DELETE`` removes the last symbol; if the buffer is empty it removes
        the last confirmed character instead.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.text = ""
        self.last_decoded: Optional[str] = None

    # ------------------------------------------------------------------ #
    def apply(self, event: str) -> None:
        handler = {
            "DOT": lambda: self._append("."),
            "DASH": lambda: self._append("-"),
            "CONFIRM": self.confirm,
            "SPACE": self.space,
            "DELETE": self.delete,
        }.get(event)
        if handler is None:
            raise ValueError(f"Unknown event: {event}")
        handler()

    def _append(self, symbol: str) -> None:
        if len(self.buffer) < MAX_SYMBOLS:
            self.buffer += symbol

    def confirm(self) -> None:
        if not self.buffer:
            return
        char = decode_symbol(self.buffer) or UNKNOWN_CHAR
        self.text += char
        self.last_decoded = char
        self.buffer = ""

    def space(self) -> None:
        self.confirm()
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def delete(self) -> None:
        if self.buffer:
            self.buffer = self.buffer[:-1]
        elif self.text:
            self.text = self.text[:-1]

    def clear(self) -> None:
        self.buffer = ""
        self.text = ""
        self.last_decoded = None

    # ------------------------------------------------------------------ #
    @property
    def preview(self) -> Optional[str]:
        """Character the current buffer would decode to, if any."""
        return decode_symbol(self.buffer) if self.buffer else None
