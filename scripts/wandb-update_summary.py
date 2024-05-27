import pandas as pd 
import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("123/large_batch-2")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    summary = {}
    for row in run.scan_history():
        summary.update(row)
    run.summary.update(summary)
    print("Updated summary to: ", summary)
