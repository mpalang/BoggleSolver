from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "resources"          # bundled, fixed
TEMPLATES = RESOURCES / "templates"
TRAININGSIMAGES = RESOURCES / "trainings_images"
DICTIONARIES = RESOURCES / "dictionaries"
