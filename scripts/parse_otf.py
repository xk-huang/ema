#%%
import subprocess
import os
import pandas as pd

os.chdir("..")

#%%
metrics_file_paths = subprocess.check_output("find outputs/xk-tet-full-dyn-mlp-5  -path '*/validate_otf/*' -name metrics.txt", shell=True).decode('utf-8').split('\n')

#%%
metric_dict_ls = []
for metrics_file_path in metrics_file_paths:
    # metrics_file_path = metrics_file_paths[0]
    if not os.path.exists(metrics_file_path):
        continue
    metric_file_path_sample = metrics_file_path.split("/")
    dir_path = "/".join(metric_file_path_sample[:-4])

    subject_id, iteration, split = metric_file_path_sample[2], metric_file_path_sample[6], metric_file_path_sample[7]



    with open(metrics_file_path, 'r') as f:
        psnr = f.readlines()[-1]
        psnr = psnr.split(',')[1]
        psnr = float(psnr)
    metric_dict = {"subject_id": subject_id, "iteration": iteration, "split": split, 'psnr': psnr, "dir_path": dir_path}
    metric_dict_ls.append(metric_dict)
# %%
df = pd.DataFrame(metric_dict_ls)
df.to_csv("outputs/otf_metrics.csv", index=False)
# %%
