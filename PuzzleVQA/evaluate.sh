#!/bin/bash
set -e

MODEL_NAME=$1

# List of datasets
DATASETS=(
  "circle_size_number"
  "color_grid"
  "color_hexagon"
  "color_number_hexagon"
  "color_overlap_squares"
  "color_size_circle"
  "grid_number_color"
  "grid_number"
  "polygon_sides_color"
  "polygon_sides_number"
  "rectangle_height_color"
  "rectangle_height_number"
  "shape_morph"
  "shape_reflect"
  "shape_size_grid"
  "shape_size_hexagon"
  "size_cycle"
  "size_grid"
  "triangle"
  "venn"
)

# Loop through each dataset and run the evaluation
for DATA in "${DATASETS[@]}"; do
  echo "Evaluating dataset: $DATA with model: $MODEL_NAME"
  python main.py evaluate_multi_choice data/${DATA}.json \
  --model_name ${MODEL_NAME} \
  --prompt_name cot_multi_extract
done