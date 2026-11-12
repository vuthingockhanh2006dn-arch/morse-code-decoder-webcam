"""Tra bảng Morse và bộ đệm văn bản. Không dùng học máy."""
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

MAX_SYMBOLS = 5      # mã dài nhất trong bảng
UNKNOWN_CHAR = "?"   # ghi ra khi chuỗi đã xác nhận không có trong bảng


def decode_symbol(code: str) -> Optional[str]:
    """``"..."`` -> ``"S"``; không có trong bảng thì trả về None."""
    return MORSE_TO_CHAR.get(code)


def decode_message(message: str) -> str:
    """Chữ cách nhau bằng khoảng trắng, từ cách nhau bằng ``" / "``."""
    return " ".join(
        "".join(MORSE_TO_CHAR.get(code, UNKNOWN_CHAR) for code in word.split())
        for word in message.strip().split(" / ")
    )


def encode_message(text: str) -> str:
    """Ngược lại của ``decode_message``, dùng trong kiểm thử."""
    return " / ".join(
        " ".join(CHAR_TO_MORSE[c] for c in word if c in CHAR_TO_MORSE)
        for word in text.upper().split()
    )


class MorseBuffer:
    """Giữ chuỗi chấm/gạch đang gõ và văn bản đã xác nhận.

    DOT / DASH   thêm "." / "-" vào bộ đệm (bỏ qua nếu đã đủ MAX_SYMBOLS).
    CONFIRM      giải mã bộ đệm thành một ký tự ("?" nếu không có trong bảng).
    SPACE        giải mã phần đang chờ (nếu có) rồi thêm một dấu cách.
    DELETE       xóa ký hiệu cuối; bộ đệm rỗng thì xóa ký tự cuối của văn bản.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.text = ""

    def apply(self, event: str) -> None:
        if event == "DOT":
            self._append(".")
        elif event == "DASH":
            self._append("-")
        elif event == "CONFIRM":
            self.confirm()
        elif event == "SPACE":
            self.space()
        elif event == "DELETE":
            self.delete()
        else:
            raise ValueError(f"Sự kiện không hợp lệ: {event}")

    def _append(self, symbol: str) -> None:
        if len(self.buffer) < MAX_SYMBOLS:
            self.buffer += symbol

    def confirm(self) -> None:
        if self.buffer:
            self.text += decode_symbol(self.buffer) or UNKNOWN_CHAR
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
