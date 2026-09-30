import argparse
import json
import math
import os
import random
import xml.etree.ElementTree as ET

DATA_DIR = "D:/Desktop/ViCo"
N_POINTS = 10000

CITIES = [
    "AMSTERDAM", "AUSTIN", "BALTIMORE", "BARCELONA", "BELGRADE", "BERLIN", "BOISE", "BOSTON",
    "BRATISLAVA", "BRUSSELS", "BUDAPEST", "CALGARY", "CHARLOTTE", "CHICAGO", "CHRISTCHURCH",
    "COLUMBUS", "DENVER", "DETROIT", "EL_PASO", "FLORENCE", "FORT_WORTH", "FRANKFURT", "HAMBURG",
    "HARVARD", "KANSAS_CITY", "LASVEGAS", "LONDON", "LONGISLAND", "MADISON", "MADRID", "MADRID2",
    "MILAN", "MINNEAPOLIS", "MIT", "MONTREAL", "NY", "ORLANDO", "PARIS", "PHILADELPHIA", "PORTLAND",
    "ROME", "SANFRANCISCO", "SANFRANCISCO2", "SILICONVALLEY", "STANFORD", "SYDNEY", "TORONTO",
    "UCLA", "UMASS", "WHITEHOUSE", "YALE", "ZURICH",
]
# CITIES = [
#     "AMSTERDAM"
# ]


def parse_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_width(value):
    if not value:
        return None
    text = str(value).strip().lower()
    try:
        if text.endswith("m"):
            return float(text[:-1].strip())
        if "ft" in text:
            return float(text.replace("ft", "").strip()) * 0.3048
        if "'" in text:
            return float(text.replace("'", "").strip()) * 0.3048
        return float(text)
    except (ValueError, TypeError):
        return None


def is_sidewalk_way(tags):
    """判断是否为sidewalk way"""
    highway = tags.get("highway")
    footway = tags.get("footway")
    sidewalk = tags.get("sidewalk")

    if highway == "footway" and footway != "crossing":
        return True
    if footway == "sidewalk":
        return True
    # if sidewalk in {"yes", "both", "left", "right"}:
    #     return True
    # Removed: if tags.get("sidewalk:both") == "separate": return True
    return False


def get_sidewalk_width(tags):
    width = None
    left = parse_width(tags.get("sidewalk:left:width"))
    right = parse_width(tags.get("sidewalk:right:width"))
    if left is not None and right is not None:
        width = (left + right) / 2.0
    elif left is not None:
        width = left
    elif right is not None:
        width = right

    for key in [
        # "sidewalk:width",
        # "sidewalk:both:width",
        "footway:width",
        "width:sidewalk",
        "width",
    ]:
        if width is not None:
            break
        width = parse_width(tags.get(key))

    if width is None or width <= 0:
        width = 1.5
    return width


def load_osm_with_height(osm_file_path):
    tree = ET.parse(osm_file_path)
    root = tree.getroot()

    nodes = {}
    ways = []

    for element in root:
        if element.tag != "node":
            continue
        node_data = {
            "id": element.get("id"),
            "lat": parse_float(element.get("lat")) or 0.0,
            "lon": parse_float(element.get("lon")) or 0.0,
            "height": 0.0,
            "utm_x": 0.0,
            "utm_y": 0.0,
        }

        for tag in element.findall("tag"):
            k = tag.get("k")
            v = tag.get("v")
            if k == "height":
                node_data["height"] = parse_float(v) or 0.0
            elif k == "utm_x":
                node_data["utm_x"] = parse_float(v) or 0.0
            elif k == "utm_y":
                node_data["utm_y"] = parse_float(v) or 0.0

        nodes[node_data["id"]] = node_data

    for element in root:
        if element.tag != "way":
            continue
        way_data = {
            "id": element.get("id"),
            "nodes": [nd.get("ref") for nd in element.findall("nd")],
            "tags": {},
        }
        for tag in element.findall("tag"):
            way_data["tags"][tag.get("k")] = tag.get("v")
        ways.append(way_data)

    return nodes, ways


def build_sidewalk_segments(nodes, ways):
    segments = []
    for way in ways:
        tags = way.get("tags", {})
        if not is_sidewalk_way(tags):
            continue
        width = get_sidewalk_width(tags)
        node_ids = way.get("nodes", [])
        for i in range(len(node_ids) - 1):
            start = nodes.get(node_ids[i])
            end = nodes.get(node_ids[i + 1])
            if not start or not end:
                continue
            dx = end["utm_x"] - start["utm_x"]
            dy = end["utm_y"] - start["utm_y"]
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            heading = math.atan2(dy, dx)
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "length": length,
                    "width": width,
                    "heading": heading,
                    "way_id": way.get("id"),
                }
            )
    return segments


def sample_points_on_segments(segments, count, rng):
    if not segments or count <= 0:
        return []

    total_length = sum(seg["length"] for seg in segments)
    cumulative = []
    running = 0.0
    for seg in segments:
        running += seg["length"]
        cumulative.append(running)

    points = []
    for _ in range(count):
        target = rng.uniform(0.0, total_length)
        seg_idx = 0
        while seg_idx < len(cumulative) and cumulative[seg_idx] < target:
            seg_idx += 1
        seg = segments[min(seg_idx, len(segments) - 1)]

        start = seg["start"]
        end = seg["end"]
        t = rng.random()
        dx = end["utm_x"] - start["utm_x"]
        dy = end["utm_y"] - start["utm_y"]
        length = seg["length"]
        if length <= 0:
            continue

        unit_x = dx / length
        unit_y = dy / length
        perp_x = -unit_y
        perp_y = unit_x

        width = seg["width"]
        offset = rng.uniform(-width / 2.0, width / 2.0)

        x = start["utm_x"] + t * dx + perp_x * offset
        y = start["utm_y"] + t * dy + perp_y * offset
        height = start["height"] + t * (end["height"] - start["height"])
        lat = start["lat"] + t * (end["lat"] - start["lat"])
        lon = start["lon"] + t * (end["lon"] - start["lon"])

        points.append(
            {
                "utm_x": x,
                "utm_y": y,
                "height": height,
                "lat": lat,
                "lon": lon,
                "width": width,
                "way_id": seg["way_id"],
                "heading": seg["heading"],
            }
        )
    return points


def process_city(city, n_points, seed):
    osm_file = os.path.join(DATA_DIR, city, "road_data", "road_data_with_height.osm")
    if not os.path.exists(osm_file):
        print(f"Skip {city}: missing {osm_file}")
        return None

    nodes, ways = load_osm_with_height(osm_file)
    segments = build_sidewalk_segments(nodes, ways)
    if not segments:
        print(f"Skip {city}: no sidewalk segments found")
        return None

    rng = random.Random(seed)
    points = sample_points_on_segments(segments, n_points, rng)
    total_length = sum(seg["length"] for seg in segments)

    return {
        "city": city,
        "input": osm_file,
        "sidewalk_segments": len(segments),
        "sidewalk_length_m": total_length,
        "points": points,
    }


def get_city_list(selected_city, selected_cities):
    if selected_city:
        return [selected_city]
    if selected_cities:
        return selected_cities
    return CITIES


def main():
    parser = argparse.ArgumentParser(description="Generate random points on sidewalks.")
    parser.add_argument("--city", type=str, default=None, help="Single city name")
    parser.add_argument("--cities", nargs="+", default=None, help="Multiple city names")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(DATA_DIR, "generated"),
        help="Base output directory (will write into {city}/{city}_split_terrain_all)",
    )
    args = parser.parse_args()

    city_list = get_city_list(args.city, args.cities)

    for city in city_list:
        result = process_city(city, N_POINTS, args.seed)
        if not result:
            continue
        out_dir = os.path.join(args.output_dir, city, f"{city}_split_terrain_all")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "sidewalk_points.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(result['points'])} points to {out_path}")


if __name__ == "__main__":
    main()
