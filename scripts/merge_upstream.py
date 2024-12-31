#!/usr/bin/env python
"""
Helper script to manage upstream merges and handle path/import changes.
Run this after git merge upstream/main to fix common conflicts.
"""
from pathlib import Path


def fix_imports(file_path: Path) -> None:
    """Fix imports from aisuite to aisuiteplus"""
    if not file_path.is_file():
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace imports
    content = content.replace("from aisuite", "from aisuiteplus")
    content = content.replace("import aisuite", "import aisuiteplus")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    # Get repository root
    repo_root = Path(__file__).parent.parent

    # Directories to process
    dirs_to_process = [
        repo_root / "aisuiteplus",
        repo_root / "tests",
        repo_root / "examples",
    ]

    # Process Python files
    for directory in dirs_to_process:
        if not directory.exists():
            continue
        for file_path in directory.rglob("*.py"):
            print(f"Processing {file_path}")
            fix_imports(file_path)

    print("\nDon't forget to:")
    print("1. Review the changes")
    print("2. Run tests to ensure everything works")
    print("3. Update version numbers if needed")


if __name__ == "__main__":
    main()
