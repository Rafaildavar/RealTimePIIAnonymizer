from typing import Dict

def unmask_pii(masked_text: str, mapping: Dict[str, str]) -> str:
    """Восстанавливает исходный текст из masked_text и mapping."""
    restored = masked_text

    for placeholder in sorted(mapping, key=len, reverse=True):
        restored = restored.replace(placeholder, mapping[placeholder])

    return restored

