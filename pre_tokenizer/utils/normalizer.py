import unicodedata

def normalizer(text:str):
    normalized = unicodedata.normalize("NFD", text)

    ascii_text = "".join(c for c in normalized if not unicodedata.combining(c))
    return ascii_text