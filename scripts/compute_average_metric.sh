#!/bin/bash

base_dirs=(
outputs/base_config-1.zju_mocap.logl2/zju_mocap/315/230306_084634
outputs/base_config-1.zju_mocap.logl2/zju_mocap/392/230306_084729
outputs/base_config-1.zju_mocap.logl2/zju_mocap/313/230306_084623
outputs/base_config-1.zju_mocap.logl2/zju_mocap/394/230306_084719
outputs/base_config-1.zju_mocap.logl2/zju_mocap/387/230306_084707
outputs/base_config-1.zju_mocap.logl2/zju_mocap/386/230306_084656
)
# outputs/base_config-1.zju_mocap.logl2/zju_mocap/390/230306_084718
# outputs/base_config-1.zju_mocap.logl2/zju_mocap/393/230306_084741
# outputs/base_config-1.zju_mocap.logl2/zju_mocap/377/230306_084645


output_file_path=tmp/240416-metric.log
if [ -n $FORCE_RERUN ]; then
    echo "Force rerun"
    rm $output_file_path
fi

if [ -f $output_file_path ]; then
    echo "Skip running inference. Output file already exists at $output_file_path"
else
for base_dir in ${base_dirs[@]}; do
echo $base_dir
python -W ignore scripts/metric_psnr_ssmi_lpips.py \
-t $base_dir/validate_otf/better/validate_novel_view/opt \
-g $base_dir/validate_otf/better/validate_novel_view/ref \
--exp_name metric --log_file $output_file_path
done
fi



# Initialize variables to store total values
total_psnr=0
total_ssim=0
total_lpips=0
count=0

# Read the log file line by line
while IFS= read -r line
do
  # Extract psnr, ssim, and lpips values using awk and add them to total values
  psnr=$(echo $line | awk '{print $3}')
  ssim=$(echo $line | awk '{print $5}')
  lpips=$(echo $line | awk '{print $7}')

  echo $psnr $ssim $lpips

  total_psnr=$(echo "$total_psnr + $psnr" | bc -l)
  total_ssim=$(echo "$total_ssim + $ssim" | bc -l)
  total_lpips=$(echo "$total_lpips + $lpips" | bc -l)

  # Increment the count
  ((count++))
done < "$output_file_path"

# Calculate averages
average_psnr=$(echo "$total_psnr / $count" | bc -l)
average_ssim=$(echo "$total_ssim / $count" | bc -l)
average_lpips=$(echo "$total_lpips / $count" | bc -l)

# Print averages
echo "Average PSNR: $average_psnr"
echo "Average SSIM: $average_ssim"
echo "Average LPIPS: $average_lpips"