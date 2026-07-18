from pathlib import Path

file = "dashboard.py"

text = Path(file).read_text(encoding="utf-8")
text = text.replace("\u00A0", " ")
Path(file).write_text(text, encoding="utf-8")

print("Done! Removed all non-breaking spaces.")