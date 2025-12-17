import os
import random
import numpy as np
import torch as T
from config import DATA_DIR
from pathlib import Path
import yaml
import re

def slugify(text: str) -> str:
    """
    Turn a title or arbitrary string into a URL/ID-safe slug.
    - Lowercases
    - Replaces non-alphanumeric runs with hyphens
    - Strips leading/trailing hyphens
    """
    text = text.lower()
    # Replace any sequence of non a-z0-9 with a single hyphen
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")

def _clean_markdown_body(body: str) -> str:
    """
    Clean the markdown body to remove navigation/promo boilerplate.
    You can refine these rules over time as you see more files.
    """
    lines = body.splitlines()
    cleaned = []
    skip_nav_block = False

    for line in lines:
        stripped = line.strip()

        # Skip "Navigate here" navigation section
        if stripped.startswith("Navigate here"):
            skip_nav_block = True
            continue

        # Skip the bullet links under "Navigate here"
        if skip_nav_block:
            if stripped.startswith("* ["):
                continue
            if stripped == "":
                # blank line ends the nav block
                skip_nav_block = False
                continue

        # Skip obvious promo / coupon / group links
        if stripped.startswith("Discover more"):
            continue
        if "WhatsApp group" in stripped or "Telegram Channel" in stripped:
            continue
        if "Saudi Coupon Codes" in stripped:
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def load_publication(
    publication_external_id="7-types-of-saudi-premium-residency-3d92712a",
):
    """Loads the publication markdown file.

    Returns:
        Content of the publication as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    publication_fpath = Path(os.path.join(DATA_DIR, f"{publication_external_id}.md"))

    # Check if file exists
    if not publication_fpath.exists():
        raise FileNotFoundError(f"Publication file not found: {publication_fpath}")

    # Read and parse the file
    try:
        raw_text = publication_fpath.read_text(encoding="utf-8")
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e

    frontmatter = {}
    body = raw_text

    # Parse YAML frontmatter if present
    if raw_text.startswith("---"):
        # split into: leading '---', frontmatter block, rest of content
        try:
            _, fm_text, body = raw_text.split("---", 2)
            frontmatter = yaml.safe_load(fm_text) or {}
        except ValueError:
            # no proper triple-split, treat as no frontmatter
            frontmatter = {}
            body = raw_text

    title = frontmatter.get("title") or publication_external_id
    source_url = frontmatter.get("source_url")
    scraped_at = frontmatter.get("scraped_at")

    cleaned_body = _clean_markdown_body(body)

    return {
        "title": title,
        "source_url": source_url,
        "scraped_at": scraped_at,
        "path": str(publication_fpath),
        "content": cleaned_body,
    }


def load_all_publications(publication_dir: str = DATA_DIR) -> list[dict]:
    """
    Loads all publication markdown files in the given directory and returns
    a list of structured publication dicts.

    Each dict has at least:
        - title
        - source_url
        - scraped_at
        - path
        - content
    """
    publications: list[dict] = []
    for filename in os.listdir(publication_dir):
        if not filename.endswith(".md"):
            continue

        external_id = filename[:-3]  # strip .md
        pub = load_publication(external_id)
        publications.append(pub)
    return publications


def set_seeds(seed_value: int) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int): The seed value to use for random
            number generation. Must be an integer.

    Returns:
        None
    """
    if isinstance(seed_value, int):
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        T.manual_seed(seed_value)
        T.cuda.manual_seed(seed_value)
        T.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
    else:
        raise ValueError(f"Invalid seed value: {seed_value}. Cannot set seeds.")
