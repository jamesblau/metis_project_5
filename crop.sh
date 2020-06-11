#!/bin/bash -e

image_dir="data/tbh7_purp_fox_vgbc"
output_dir="data/tbh7_purp_fox_vgbc_larger_cropped"
region_file="labels/regions/tbh7_purp_fox_vgbc_larger_cropped_labels.csv"

for image_filename in $(ls ${image_dir} | shuf); do
  if grep -q "^${image_filename}," ${region_file}; then
    echo $image_filename Already done
  else
    echo $image_filename to crop
    python3 manually_crop.py --image ${image_dir}/${image_filename} --output_dir ${output_dir} --region_file ${region_file}
  fi
  echo
done
