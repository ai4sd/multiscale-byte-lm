import datetime
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_changed_files(repo_path: Path) -> List[Path]:
    """
    Get the list of changed Python files (staged in Git).

    Args:
        repo_path: Path to the main repository folder.

    Returns:
        List of changed Python files.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--name-only", "--cached", "--diff-filter=ACM"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        changed_files = [
            Path(file.strip())
            for file in result.stdout.splitlines()
            if file.strip().endswith(".py")
        ]
        # Filter files to ensure they are under repo_path
        return [file for file in changed_files if file.is_relative_to(repo_path)]
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e.stderr}")
        sys.exit(1)


def check_file(file_path: Path) -> bool:
    """
    Check if the `__copyright__` attribute is present in the file.

    Args:
        file_path: Path to the Python file to check.

    Returns:
        True if `__copyright__` is present, False otherwise.
    """
    lines = file_path.read_text().splitlines()
    return any("__copyright__" in line for line in lines)


def process_file(file_path: Path, license_text: str) -> None:
    """
    Add a `__copyright__` attribute to the file if it does not already exist.

    Args:
        file_path: Path to the Python file to process.
        license_text: License text to insert as the value of `__copyright__`.
    """
    lines = file_path.read_text().splitlines()

    # Skip if __copyright__ already exists
    if any("__copyright__" in line for line in lines):
        return

    # Add __copyright__ at the top of the file
    with file_path.open("w") as file:
        file.write(f'__copyright__ = """{license_text}"""\n\n')
        file.write("\n".join(lines) + "\n")

    print(f"Updated {file_path}")


def main(
    repo_path: Path,
    license_file: Path,
    year: Optional[int] = None,
    changed_only: bool = False,
    check: bool = False,
) -> None:
    """
    Main function to add or check `__copyright__` attributes in Python files.

    Args:
        repo_path (Path): Path to the main repository folder.
        license_file: Path to the LICENSE template file.
        year: Year to use in the license (defaults to the current year).
        changed_only: Whether to process only changed files (staged in Git).
        check: Whether to check for the presence of the `__copyright__` attribute.
    """
    year = year or datetime.datetime.now().year

    # Check if LICENSE file exists
    if not license_file.exists():
        print(f"Error: LICENSE.tmpl not found at {license_file}")
        sys.exit(1)

    # Read and replace year in the license text
    license_text = license_file.read_text().replace("${year}", str(year))

    # Determine which files to process
    if changed_only:
        files_to_process = get_changed_files(repo_path)
    else:
        files_to_process = [path for path in repo_path.rglob("*.py") if path.is_file()]

    if check:
        # Check mode: verify that all files contain the `__copyright__` attribute
        missing_license = [
            str(file_path) for file_path in files_to_process if not check_file(file_path)
        ]
        if missing_license:
            print("The following files are missing the license header:")
            print("\n".join(missing_license))
            sys.exit(1)  # Return non-zero exit code to indicate failure
        print("All files have the license header.")
        sys.exit(0)  # Return zero exit code to indicate success

    # Process each file
    for file_path in files_to_process:
        process_file(file_path, license_text)


if __name__ == "__main__":
    import argparse

    # Determine the default location of the LICENSE.tmpl file
    script_path = Path(__file__).resolve()
    default_license_file = script_path.parent.parent / "mit.tmpl"

    parser = argparse.ArgumentParser(description="Add __copyright__ to Python files.")
    parser.add_argument("repo_path", type=Path, help="Path to the main repository folder")
    parser.add_argument(
        "--license-file",
        type=Path,
        default=default_license_file,
        help=f"Path to the LICENSE template file (default: {default_license_file})",
    )
    parser.add_argument(
        "--year", type=int, help="Year to use in the license (defaults to current year)"
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="Process only changed files (staged in Git)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if license headers are present without modifying files",
    )

    args = parser.parse_args()

    main(args.repo_path, args.license_file, args.year, args.changed_only, args.check)
