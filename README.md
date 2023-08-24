# RL next generalization (building on wouterkool/attention-learn-to-route)
This project extends on former work by Kool et al. and was forked from this [repo](https://github.com/wouterkool/attention-learn-to-route) with the main goal of comparing additional reinforcement learning algorithms on the TSP and OP and analyzing there generalization capabilities with regard to unlearned problem sizes.

## dependency management
Dependencies are managed using [pip-tools](https://github.com/jazzband/pip-tools).
Add new dependencies to `requirements.in` and run `pip-compile` to update `requirements.txt`.

## tmux command examples
```
tmux new -s Kenneth
tmux attach

Ctrl+B [ # scroll mode, q to quit
Ctrl+B D # detach
Ctrl+B W # window overview
Ctrl+B C # create window
Ctrl+D   # delete window
```

## copy data between remote server and local repo quickly using rsync
```
rsync -avP . <USER>@<SERVER>:<FULL_PATH>/attention-learn-to-route --delete --exclude-from rsync_excludes.txt
rsync -avP <USER>@<SERVER>:<FULL_PATH>/attention-learn-to-route/log_dir 
```

## make sure all necessary folders exist on remote (like `policy_dir`)

## set up new experiments
All experiment configurations are set up in `experiment_configurations.numbers`.
The tables can be exported to csv and saved in the `run_configs/` directory.
After copying this directory to the remote server, experiments can be run.

## run command examples
```
bash scheduler.sh run_configs/<file>.csv <x_first_experiments>
python3 run.py --args_from_csv run_configs/<file>.csv --csv_row <row_id> --gpu_id 0
```
The best policies are saved to the `policy_dir` directory.

## eval command examples
```
python3 run.py --saved_policy_path policy_dir/run_127__20230823T094935.pth --gpu_id 0
```


## preview log data using tensorboard
```
tensorboard --logdir log_dir/ --reload_multifile TRUE
```

## preparing log data for visualizations/plots
```
git clone https://github.com/Kenneth-Schroeder/tensorboard-aggregator
python3 aggregator.py --path ../attention-learn-to-route/log_dir/trainings
```

## visualize/plot log data using code in plotting.ipynb
Plots are saved in `figures/` and corresponding meta data to each plot is saved in `figure_metas/`.
