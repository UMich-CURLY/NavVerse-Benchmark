#!/usr/bin/env bash
# 对 vc_plus 下所有全大写 city 文件夹分别执行 Blender 导出 + view_out_innout 绘图，
# 中间 JSON 与输出图片保存到 ../data 下两个新建文件夹。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VC_PLUS_DIR="${VC_PLUS_DIR:-${REPO_ROOT}/navverse_data/vc_apr14}"
# ../data 相对脚本所在目录（即 robot_env/tools -> robot_env/data）
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"
JSON_DIR="$DATA_DIR/usd_positions_json"
IMAGE_DIR="$DATA_DIR/bev_outdoor_images"

mkdir -p "$JSON_DIR"
mkdir -p "$IMAGE_DIR"

BLENDER_SCRIPT="$SCRIPT_DIR/view_out_innout_blender_export.py"
PLOT_SCRIPT="$SCRIPT_DIR/view_out_innout.py"

if [[ ! -f "$BLENDER_SCRIPT" ]]; then
  echo "错误: 未找到 $BLENDER_SCRIPT"
  exit 1
fi
if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "错误: 未找到 $PLOT_SCRIPT"
  exit 1
fi

# 仅处理全大写目录名（city）
for city_dir in "$VC_PLUS_DIR"/*/; do
  city="$(basename "$city_dir")"
  # 排除非全大写（如 NY、road_objects）
  if [[ ! "$city" =~ ^[A-Z][A-Z0-9]*$ ]]; then
    continue
  fi
  echo "========== $city =========="
  json_path="$JSON_DIR/usd_positions_${city}.json"
  scene_name="vc_${city,,}_store"
  image_path="$IMAGE_DIR/view_out_innout_${scene_name}.png"

  echo "[1/2] Blender 导出 JSON -> $json_path"
  blender --background --python "$BLENDER_SCRIPT" -- --city "$city" --vc-plus-dir "$VC_PLUS_DIR" --output "$json_path" || {
    echo "警告: Blender 导出 $city 失败，跳过绘图"
    continue
  }

  echo "[2/2] 绘图（含室内五角星）-> $image_path"
  (cd "$SCRIPT_DIR" && python "$PLOT_SCRIPT" --city "$city" --positions-json "$json_path" --output "$image_path" --vc-plus-dir "$VC_PLUS_DIR" --draw-indoor-star) || {
    echo "警告: 绘图 $city 失败"
  }
done

echo "完成. JSON: $JSON_DIR  图片: $IMAGE_DIR"
