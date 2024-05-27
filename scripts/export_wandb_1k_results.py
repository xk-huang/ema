import pandas as pd 
import wandb

history_keys = [
    "metrics_otf_better/psnr/validate_novel_view",
    "metrics_otf_better/psnr/validate_novel_pose",
    "metrics_otf_better/ssim/validate_novel_view",
    "metrics_otf_better/ssim/validate_novel_pose",
]
history_step = 10
config_keys = [
    "subject_id",
    "learning_rate_schedule_type",
    "learning_rate_final_mult",
    "subdivide_tetmesh_iters",
    "out_dir",
]

api = wandb.Api()
entity, project = "123", "base_config-1.zju_mocap.logl2"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    # summary_list.append(run.summary._json_dict)
    try:
        summary_list.append({
        _history_key: run.history()[_history_key][10] for _history_key in history_keys
        })
    except KeyError as e:
        print(f"missing key: {e} in {run.name}")
        continue

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    # config_list.append(
    #     {k: v for k,v in run.config.items()
        #  if not k.startswith('_')})
    config_list.append(
        {k:run.config[k] for k in config_keys}
    )

    # .name is the human-readable name of the run.
    name_list.append(run.name)


pd_list = [{**summary_list[i] , **config_list[i]} for i in range(len(summary_list))]
df = pd.DataFrame(pd_list)

df.to_csv("tmp/230306.1k.csv")


summary_seris = []
out_dir_seris = []
df_idxmax = df.groupby("subject_id").idxmax()
for _history_key in history_keys:
    print(_history_key)
    # print(df.iloc[df_idxmax[_history_key]])
    _summary = df.iloc[df_idxmax[_history_key]].mean()
    _summary = _summary.append(pd.Series([_history_key], index=["based_on_which_metric_key"]))
    summary_seris.append(_summary)

    out_dirs = pd.concat([pd.Series([_history_key  for _ in range(len(df_idxmax))]), df.iloc[df_idxmax[_history_key]]["out_dir"].reset_index(drop=True)], axis=1)
    out_dir_seris.append(out_dirs)

summary_df = pd.DataFrame(summary_seris)
summary_df.to_csv("tmp/230306.1k.summary.csv")

out_dir_df = pd.concat(out_dir_seris)
out_dir_df.to_csv("tmp/230306.1k.out_dir.csv")

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("tmp/230306.project.csv")
