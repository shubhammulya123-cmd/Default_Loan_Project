from pathlib import Path

folders = [
    "data/raw",
    "data/processed",
    "src"
]

files = [
    "requirements.txt",
    "README.md",
    "src/data_ingestion.py",
    "src/data_validation.py"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch(exist_ok=True)

print("Project skeleton created successfully.")
