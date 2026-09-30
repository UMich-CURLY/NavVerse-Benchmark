#!/usr/bin/env python3
"""Build storefront intent instructions and an HTML review page."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from html import escape
from pathlib import Path

from dotenv import dotenv_values


DEFAULT_OUT_DIR = Path("navverse_tools/storefront")

INTENT_INSTRUCTIONS = {
    "arts_centre": [
        "a place to see local art exhibitions",
        "a place to attend a gallery opening",
        "a place to take an art class or workshop",
        "a place to browse paintings and creative displays",
        "a place to spend an afternoon looking at artwork",
    ],
    "atm": [
        "a place to withdraw cash quickly",
        "a place to check a bank balance",
        "a place to get money before shopping",
        "a place to use a cash machine",
        "a place to take out cash from an account",
    ],
    "bank": [
        "a place to open a bank account",
        "a place to deposit checks or cash",
        "a place to speak with a banker about finances",
        "a place to handle loans and savings",
        "a place to manage personal banking",
    ],
    "bar": [
        "a place to get drinks with friends",
        "a place to meet someone for an evening cocktail",
        "a place to relax at the counter after work",
        "a place to listen to music over a beer",
        "a place to have a late-night drink",
    ],
    "bicycle_rental": [
        "a place to rent a bike for the day",
        "a place to pick up a bicycle for sightseeing",
        "a place to get wheels for a short city ride",
        "a place to borrow a bike near downtown",
        "a place to start a cycling trip",
    ],
    "bicycle_repair_station": [
        "a place to fix a flat bike tire",
        "a place to repair bicycle brakes",
        "a place to service a bike before riding",
        "a place to get help with a broken chain",
        "a place to tune up a bicycle",
    ],
    "bureau_de_change": [
        "a place to exchange foreign currency",
        "a place to get local cash while traveling",
        "a place to convert dollars into another currency",
        "a place to handle travel money",
        "a place to check exchange rates",
    ],
    "cafe": [
        "a place to get coffee and a pastry",
        "a place to sit down for tea",
        "a place to meet a friend for breakfast",
        "a place to work with a latte",
        "a place to take a quiet coffee break",
    ],
    "car_rental": [
        "a place to rent a car for a trip",
        "a place to pick up a vehicle reservation",
        "a place to get a car for the weekend",
        "a place to arrange transportation out of town",
        "a place to return a rental car",
    ],
    "cinema": [
        "a place to watch a movie",
        "a place to buy tickets for a film",
        "a place to see a new release with friends",
        "a place to get popcorn before a show",
        "a place to spend an evening at the movies",
    ],
    "clinic": [
        "a place to get medical care",
        "a place to visit a health clinic",
        "a place to see a nurse for a quick checkup",
        "a place to get treatment for a minor illness",
        "a place to ask about health concerns",
    ],
    "community_centre": [
        "a place to join a neighborhood event",
        "a place to attend a local meeting",
        "a place to find community services",
        "a place to take part in group activities",
        "a place to meet people from the neighborhood",
    ],
    "convenience_store": [
        "a place to buy snacks and drinks",
        "a place to pick up everyday essentials",
        "a place to grab something quick on the way home",
        "a place to buy bottled water and small groceries",
        "a place to get last-minute supplies",
    ],
    "coworking_space": [
        "a place to work at a shared desk",
        "a place to take a meeting outside the office",
        "a place to use a quiet workspace for the day",
        "a place to work with other freelancers",
        "a place to book a desk with reliable wifi",
    ],
    "dentist": [
        "a place to get a dental checkup",
        "a place to fix a toothache",
        "a place to have teeth cleaned",
        "a place to see a dentist about a cavity",
        "a place to get help with dental care",
    ],
    "doctors": [
        "a place to see a doctor",
        "a place to go for a medical appointment",
        "a place to discuss symptoms with a physician",
        "a place to get a general health check",
        "a place to visit for primary care",
    ],
    "fast_food": [
        "a place to get burgers and fries",
        "a place to buy a quick chicken sandwich",
        "a place to grab fast food for lunch",
        "a place to order a combo meal",
        "a place to get a quick inexpensive meal",
    ],
    "food_court": [
        "a place where everyone can choose different food",
        "a place to find several quick meal options",
        "a place to eat with a group at shared tables",
        "a place to browse multiple food stalls",
        "a place to grab lunch in a mall",
    ],
    "kindergarten": [
        "a place to drop off young children for preschool",
        "a place for early childhood learning",
        "a place where small kids attend class",
        "a place to enroll a child in kindergarten",
        "a place for children to play and learn",
    ],
    "lab": [
        "a place to get lab tests done",
        "a place to submit samples for analysis",
        "a place to visit for diagnostic testing",
        "a place to pick up laboratory results",
        "a place to work on science experiments",
    ],
    "library": [
        "a place to borrow books",
        "a place to study quietly",
        "a place to read or research",
        "a place to use public computers",
        "a place to find books and learning resources",
    ],
    "parking_entrance": [
        "a place to enter a parking garage",
        "a place to park a car downtown",
        "a place to drive in for paid parking",
        "a place to leave a vehicle while shopping",
        "a place to access public parking",
    ],
    "pharmacy": [
        "a place to pick up medicine",
        "a place to buy health supplies",
        "a place to fill a prescription",
        "a place to ask a pharmacist about medication",
        "a place to get cold medicine and vitamins",
    ],
    "police": [
        "a place to report an incident",
        "a place to ask police for help",
        "a place to find public safety services",
        "a place to speak with an officer",
        "a place to get assistance in an emergency",
    ],
    "post_office": [
        "a place to mail a package",
        "a place to buy stamps",
        "a place to send letters",
        "a place to pick up postal deliveries",
        "a place to ship something across town",
    ],
    "restaurant": [
        "a place to have dinner with friends",
        "a place to sit down for a full meal",
        "a place to enjoy ramen or noodles",
        "a place to celebrate over food",
        "a place to eat at a table-service restaurant",
    ],
    "school": [
        "a place to attend classes",
        "a place for students to learn",
        "a place to go for education",
        "a place to meet a teacher after class",
        "a place to enroll in a learning program",
    ],
    "theatre": [
        "a place to watch a live performance",
        "a place to see a play on stage",
        "a place to attend a theatre show",
        "a place to enjoy acting and drama",
        "a place to buy tickets for a performance",
    ],
    "toilets": [
        "a place to use the restroom",
        "a place to find public toilets",
        "a place to wash hands during a trip",
        "a place to take a bathroom break",
        "a place to find accessible restrooms",
    ],
    "university": [
        "a place to visit a university building",
        "a place to find campus books and supplies",
        "a place to attend higher education classes",
        "a place to meet someone on campus",
        "a place to look for academic resources",
    ],
}


REPO_ROOT = Path(__file__).resolve().parents[2]


def default_storefront_dir() -> Path:
    env_path = os.environ.get("NAVVERSE_STOREFRONT_DIR")
    if env_path:
        return Path(env_path).expanduser()
    scene_folder = os.environ.get("SCENE_FOLDER")
    if not scene_folder:
        scene_folder = dotenv_values(REPO_ROOT / ".env").get(
            "SCENE_FOLDER", "navverse_data"
        )
    data_root = Path(scene_folder).expanduser()
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    return data_root / "storefront"


def portable_path(path: Path, data_root: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return str(Path("navverse_data") / path.relative_to(data_root))
        except ValueError:
            return str(path)


def category_from_asset_id(asset_id: str) -> str:
    return re.sub(r"_\d+$", "", asset_id)


def find_png(asset_dir: Path) -> str | None:
    textures = asset_dir / "textures"
    if not textures.is_dir():
        return None
    pngs = sorted(path for path in textures.glob("*.png") if path.name != "color_121212.png")
    return str(pngs[0]) if pngs else None


def image_suffix_for_browser(image_path: Path) -> str:
    signature = image_path.read_bytes()[:8]
    if signature.startswith(b"\xff\xd8"):
        return ".jpg"
    if signature.startswith(b"\x89PNG"):
        return ".png"
    return image_path.suffix


def materialize_review_image(image_path: str | None, review_image_dir: Path, asset_id: str) -> str | None:
    if not image_path:
        return None
    source = Path(image_path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    suffix = image_suffix_for_browser(source)
    review_image_dir.mkdir(parents=True, exist_ok=True)
    target = review_image_dir / f"{asset_id}{suffix}"
    if target.exists() or target.is_symlink():
        target.unlink()
    relative_source = os.path.relpath(source, target.parent)
    try:
        target.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, target)
    return f"review_images/{target.name}"


def build_catalog(storefront_dir: Path) -> dict:
    storefront_dir = storefront_dir.expanduser()
    data_root = storefront_dir.parent
    assets = []
    categories = {}
    for asset_dir in sorted(path for path in storefront_dir.iterdir() if path.is_dir()):
        asset_id = asset_dir.name
        category = category_from_asset_id(asset_id)
        instructions = INTENT_INSTRUCTIONS.get(category)
        if instructions is None:
            raise KeyError(f"Missing intent instructions for category {category!r}")
        image = find_png(asset_dir)
        assets.append({
            "id": asset_id,
            "category": category,
            "image": portable_path(Path(image), data_root) if image else None,
            "usd": portable_path(asset_dir / f"{asset_id}.usd", data_root),
        })
        categories.setdefault(category, {
            "category": category,
            "instructions": instructions,
            "assets": [],
        })["assets"].append(asset_id)

    return {
        "storefront_dir": portable_path(storefront_dir, data_root),
        "instruction_count_per_category": 5,
        "categories": dict(sorted(categories.items())),
        "assets": assets,
    }


def render_html(catalog: dict, out_dir: Path) -> str:
    cards = []
    category_lookup = catalog["categories"]
    review_image_dir = out_dir / "review_images"
    for index, asset in enumerate(catalog["assets"]):
        image = materialize_review_image(asset.get("image"), review_image_dir, asset["id"])
        if image:
            img_html = f'<img src="{escape(image)}" alt="{escape(asset["id"])}">'
        else:
            img_html = '<div class="missing">missing image</div>'
        instructions = category_lookup[asset["category"]]["instructions"]
        instruction_items = "\n".join(
            f"<li><code>{i}</code>: {escape(text)}</li>"
            for i, text in enumerate(instructions)
        )
        cards.append(f"""
        <article class="card" id="{escape(asset['id'])}">
          <div class="media">{img_html}</div>
          <div class="body">
            <div class="meta">
              <span class="index">#{index:03d}</span>
              <span class="asset">{escape(asset['id'])}</span>
              <span class="category">{escape(asset['category'])}</span>
            </div>
            <ol>{instruction_items}</ol>
          </div>
        </article>
        """)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Storefront Intent Instructions</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f5f5;
      color: #1d1d1f;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 16px 24px;
      background: #ffffff;
      border-bottom: 1px solid #d8d8d8;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      line-height: 1.25;
    }}
    .summary {{
      color: #555;
      font-size: 14px;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 14px;
      padding: 18px;
    }}
    .card {{
      display: grid;
      grid-template-columns: 160px minmax(0, 1fr);
      background: #ffffff;
      border: 1px solid #d6d6d6;
      border-radius: 8px;
      overflow: hidden;
    }}
    .media {{
      display: grid;
      place-items: center;
      min-height: 120px;
      background: #e9e9e9;
    }}
    img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .missing {{
      padding: 12px;
      color: #777;
      font-size: 13px;
      text-align: center;
    }}
    .body {{
      padding: 10px 12px;
      min-width: 0;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .index, .category {{
      padding: 2px 6px;
      border-radius: 999px;
      background: #ededed;
      white-space: nowrap;
    }}
    .asset {{
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    ol {{
      margin: 0;
      padding-left: 22px;
      font-size: 13px;
      line-height: 1.35;
    }}
    li + li {{
      margin-top: 4px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Storefront Intent Instructions</h1>
    <div class="summary">{len(catalog['assets'])} storefront assets, {len(catalog['categories'])} categories, 5 intent instructions per category.</div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storefront-dir", type=Path, default=default_storefront_dir())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    catalog = build_catalog(args.storefront_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.out_dir / "storefront_intent_catalog.json"
    html_path = args.out_dir / "storefront_intent_review.html"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_html(catalog, args.out_dir), encoding="utf-8")
    print(f"Wrote {catalog_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
