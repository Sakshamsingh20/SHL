"""
Catalog loading and normalization.

The raw scrape has a few rough edges we clean up here:
  - Literal control characters in some description fields (multi-line scraped text)
  - A fixed taxonomy of 8 "keys" categories that map to single-letter test_type codes,
    matching SHL's own convention (confirmed against the labeled sample conversations).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

# Confirmed against SHL's own product pages / the labeled sample conversation traces.
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


@dataclass
class Assessment:
    entity_id: str
    name: str
    url: str
    description: str
    keys: list = field(default_factory=list)
    test_type: str = ""  # comma-joined codes, e.g. "K,S"
    job_levels: list = field(default_factory=list)
    languages: list = field(default_factory=list)
    duration: str = ""
    remote: str = ""
    adaptive: str = ""

    def to_recommendation(self) -> dict:
        """Shape required by the /chat response schema."""
        return {"name": self.name, "url": self.url, "test_type": self.test_type}

    def searchable_text(self) -> str:
        """Text blob used for embedding / BM25 indexing."""
        parts = [
            self.name,
            self.description,
            "Categories: " + ", ".join(self.keys),
            "Job levels: " + ", ".join(self.job_levels) if self.job_levels else "",
            f"Duration: {self.duration}" if self.duration else "",
        ]
        return " | ".join(p for p in parts if p)


def _derive_test_type(keys: list) -> str:
    codes = []
    for k in keys:
        code = KEY_TO_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes)


def load_catalog(path: str) -> list:
    """Load and normalize the SHL catalog JSON. Tolerant of stray control chars."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f, strict=False)

    assessments = []
    seen_urls = set()
    for item in raw:
        url = (item.get("link") or "").strip()
        name = (item.get("name") or "").strip()
        if not url or not name:
            continue
        if url in seen_urls:
            continue  # de-dup, the scrape has a few repeats
        seen_urls.add(url)

        keys = item.get("keys") or []
        assessments.append(
            Assessment(
                entity_id=str(item.get("entity_id", "")),
                name=name,
                url=url,
                description=(item.get("description") or "").strip(),
                keys=keys,
                test_type=_derive_test_type(keys),
                job_levels=item.get("job_levels") or [],
                languages=item.get("languages") or [],
                duration=(item.get("duration") or "").strip(),
                remote=item.get("remote", ""),
                adaptive=item.get("adaptive", ""),
            )
        )
    return assessments


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    cat = load_catalog(os.path.join(here, "..", "data", "shl_product_catalog.json"))
    print(f"Loaded {len(cat)} assessments")
    from collections import Counter

    code_counts = Counter()
    for a in cat:
        for c in a.test_type.split(","):
            if c:
                code_counts[c] += 1
    print("Test type distribution:", dict(code_counts))
    print("\nSample:")
    for a in cat[:3]:
        print(f"  {a.name} | {a.test_type} | {a.url}")
