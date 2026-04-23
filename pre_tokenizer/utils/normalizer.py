import unicodedata
import re

def normalizer(text:str):
    normalized = unicodedata.normalize("NFD", text)

    ascii_text = "".join(c for c in normalized if not unicodedata.combining(c))

    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"\s+", " ", ascii_text)
    return ascii_text.strip()