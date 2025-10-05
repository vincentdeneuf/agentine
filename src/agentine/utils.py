import re
import tkinter as tk
from tkinter import filedialog
from print9 import print9


class Utility:
    
    @staticmethod
    def format(
        string: str,
        data: dict[str, object],
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
    def get_file_path_via_terminal() -> str | None:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        if file_path == "":
            return None
        return file_path


class ObjectService:
    @staticmethod
    def validate_keys(data: dict[str, object] | None, keys: list[str]) -> bool:
        if data is None:
            data = {}

        assert isinstance(data, dict), f"data must be a dictionary. Value: {data}"
        assert isinstance(keys, list), f"keys must be a list. Value: {keys}"
        assert all(isinstance(key, str) for key in keys), f"all keys must be strings. Value: {keys}"

        return not keys or all(key in data for key in keys)

    @staticmethod
    def wrap(data: object, key: str | None) -> dict[str, object] | object:
        if not key:
            return data

        if isinstance(data, dict) and len(data) == 1 and key in data:
            return data

        return {key: data}

    @staticmethod
    def keys(data: dict[object, object]) -> list[object]:
        assert isinstance(data, dict), "Input object must be a dictionary"
        return list(data.keys())