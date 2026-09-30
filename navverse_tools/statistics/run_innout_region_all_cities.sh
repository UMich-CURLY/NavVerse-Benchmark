#!/usr/bin/env bash
# 对 vc_plus 下所有全大写 city 分别执行 view_innout_region.py：
# 读取 {city}_innout.usd 得到室内位置，以该位置为中心取 100m 正方形，
# 显示范围内的 object/storefront，红五角星标室内位置。
# 依赖：需先有 usd_positions_{city}.json（可先运行 run_bev_export_all_cities.sh 生成）。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VC_PLUS_DIR="${VC_PLUS_DIR:-${REPO_ROOT}/navverse_data/vc_apr14}"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"
JSON_DIR="$DATA_DIR/usd_positions_json"
IMAGE_DIR="$DATA_DIR/bev_innout_images"

mkdir -p "$IMAGE_DIR"

PLOT_SCRIPT="$SCRIPT_DIR/view_innout_region.py"
if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "错误: 未找到 $PLOT_SCRIPT"
  exit 1
fi

for city_dir in "$VC_PLUS_DIR"/*/; do
  city="$(basename "$city_dir")"
  if [[ ! "$city" =~ ^[A-Z][A-Z0-9]*$ ]]; then
    continue
  fi
  scene_name="vc_${city,,}_store"
  json_path="$JSON_DIR/usd_positions_${city}.json"
  image_path="$IMAGE_DIR/view_innout_region_${scene_name}.png"

  if [[ ! -f "$json_path" ]]; then
    echo "跳过 $city: 未找到 $json_path（可先运行 run_bev_export_all_cities.sh）"
    continue
  fi

  echo "========== $city (100m 正方形 + 红五角星) =========="
  (cd "$SCRIPT_DIR" && python "$PLOT_SCRIPT" --city "$city" --positions-json "$json_path" --output "$image_path" --vc-plus-dir "$VC_PLUS_DIR") || {
    echo "警告: $city 绘图失败"
  }
done

echo "完成. 图片: $IMAGE_DIR"
