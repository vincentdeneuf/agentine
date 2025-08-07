import re
import textwrap
from typing import Dict, Any, Literal, Optional, List, Union
import tkinter as tk
from tkinter import filedialog

class Utility:
    ANSI_COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "orange": "\033[38;5;208m",
        "purple": "\033[38;5;93m",
        "pink": "\033[38;5;205m",
        "brown": "\033[38;5;94m",
        "gray": "\033[90m",
        "reset": "\033[0m",
    }

    @staticmethod
    def format(
        string: str,
        data: Dict[str, Any],
        fallback: str = "(Not Available)"
    ) -> str:
        assert isinstance(string, str), f"string must be a string. Value: {string!r}"
        assert isinstance(data, dict), f"data must be a dictionary. Value: {data!r}"
        assert isinstance(fallback, str), f"fallback must be a string. Value: {fallback!r}"

        placeholders = re.findall(r'<<(.*?)>>', string)

        for key in set(placeholders):
            value = data.get(key, fallback)
            if not isinstance(value, str):
                value = str(value)

            string = string.replace(f"<<{key}>>", value.strip())

        return string

    @staticmethod
    def print2(
        text: str,
        width: int = 120,
        color: Literal[
            "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
            "orange", "purple", "pink", "brown", "gray"
        ] = "orange",
        **kwargs
    ) -> None:
        wrapped_lines = []
        for line in text.splitlines():
            wrapped = textwrap.wrap(line, width=width, replace_whitespace=False)
            if not wrapped:
                wrapped_lines.append("")
            else:
                wrapped_lines.extend(wrapped)

        ansi_color = Utility.ANSI_COLORS.get(color, Utility.ANSI_COLORS["orange"])
        reset = Utility.ANSI_COLORS["reset"]

        for line in wrapped_lines:
            print(f"{ansi_color}{line}{reset}", **kwargs)

    @staticmethod
    def get_file_path_via_terminal() -> Optional[str]:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        if file_path == "":
            return None
        return file_path

class ObjectService:
    @staticmethod
    def validate_keys(data: Optional[Dict[str, Any]], keys: List[str]) -> bool:
        if data is None:
            data = {}

        assert isinstance(data, dict), f"data must be a dictionary. Value: {data}"
        assert isinstance(keys, list), f"keys must be a list. Value: {keys}"
        assert all(isinstance(key, str) for key in keys), f"all keys must be strings. Value: {keys}"

        return not keys or all(key in data for key in keys)

    @staticmethod
    def wrap(data: Any, key: Optional[str]) -> Union[Dict[str, Any], Any]:
        if not key:
            return data

        if isinstance(data, dict) and len(data) == 1 and key in data:
            return data

        return {key: data}

    @staticmethod
    def keys(data: Dict[Any, Any]) -> List[Any]:
        assert isinstance(data, dict), "Input object must be a dictionary"
        return list(data.keys())
