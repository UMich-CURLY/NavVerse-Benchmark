"""Shared Objaverse asset placement rules for VC+ generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


OBJAVERSE_DISABLED_ASSETS = {
    "bleach_bottle.usd",
    "green_exit_sign.usd",
    "puddle.usd",
    "purple_jacket.usd",
}

OBJAVERSE_SCALE_OVERRIDES = {
    "electric_car.usd": 0.75,
    "park_bench.usd": 0.45,
    "race_car.usd": 0.65,
}

OBJAVERSE_VEHICLE_ASSETS = {
    "ambulance.usd",
    "ambulance_1.usd",
    "bus.usd",
    "electric_car.usd",
    "food_truck.usd",
    "garbage_truck.usd",
    "offroad_truck.usd",
    "police_van.usd",
    "race_car.usd",
    "taxi.usd",
    "truck.usd",
    "van.usd",
}

OBJAVERSE_OUTDOOR_GROUPS = {
    "baby_care",
    "beach_trip_prep",
    "bike_parking",
    "bird_watching",
    "bus_stop_wait",
    "camping_rough",
    "construction_repair",
    "dining_alfresco",
    "dog_walk",
    "feeding_cat",
    "gardening",
    "mail_delivery",
    "moving_day",
    "outdoor_reading",
    "park_relaxing",
    "picnic",
    "rainy_day",
    "recycling",
    "scooter_ride",
    "sports_gear",
    "street_cleaning",
    "waiting_area",
}

OBJAVERSE_OUTDOOR_UNGROUPED_ASSETS = {
    "apartment_mailbox.usd",
    "bicycle.usd",
    "bowling_ball.usd",
    "garden_fountain.usd",
    "lilac_bike.usd",
    "parking_meter.usd",
    "squirrel.usd",
    "trash_can.usd",
    "water_fountain.usd",
    "white_scooter.usd",
}


def objaverse_asset_name(asset: Any) -> str:
    return Path(str(asset)).name


def objaverse_asset_enabled(asset: Any) -> bool:
    return objaverse_asset_name(asset) not in OBJAVERSE_DISABLED_ASSETS


def objaverse_asset_scale(asset: Any) -> float:
    return float(OBJAVERSE_SCALE_OVERRIDES.get(objaverse_asset_name(asset), 1.0))


def objaverse_outdoor_group_enabled(name: Any) -> bool:
    return str(name) in OBJAVERSE_OUTDOOR_GROUPS


def objaverse_outdoor_ungrouped_enabled(asset: Any) -> bool:
    return objaverse_asset_name(asset) in OBJAVERSE_OUTDOOR_UNGROUPED_ASSETS
