"""Run parent's independent saved-only reviewer with child output root."""
from pathlib import Path
source=Path(__file__).resolve().parent.parent/'review_saved.py'
exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),{'__name__':'__main__','__file__':__file__})
