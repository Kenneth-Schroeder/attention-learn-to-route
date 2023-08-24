# next gen(eralization) RL (building on wouterkool/attention-learn-to-route)
This project extends on former work by Kool et al. and was forked from this [repo](https://github.com/wouterkool/attention-learn-to-route) with the main goal of comparing additional reinforcement learning algorithms on the TSP and OP and analyzing there generalization capabilities with regard to unlearned problem sizes.

## directory overview
- `args/` contains all configuration arguments of started experiments
- `custom_classes/` contains custom tianshou classes
- `eval_logs/` contains optionally saved logs of evaluation runs of trained policies
- `figure_metas/` contains metadata for saved figures for easy adjustments to existing figures
- `figures/` contains created figures
- `nets/` contains torch modules for the attention model and value estimators e.g.
- `problems/` contains code for the tsp and op environments
- `run_configs/` contains csvs exported from `experiment_configurations.numbers` with specific run configurations
- `utils/` contains utility code
- `log_dir/` contains training and evaluation logs and results
- `policy_dir/` contains trained tianshou policies that can be used for evaluation runs

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
rsync -avP . <USER>@<SERVER>:<FULL_PATH>/attention-next-gen-rl --delete --exclude-from rsync_excludes.txt
rsync -avP <USER>@<SERVER>:<FULL_PATH>/attention-next-gen-rl/log_dir 
```

## set up new experiments
All experiment configurations are set up in `experiment_configurations.numbers`.
The tables can be exported to csv and saved in the `run_configs/` directory.
After copying this directory to the remote server, experiments can be run.

## run command examples
make sure all necessary folders exist on remote (like `policy_dir`)
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
python3 aggregator.py --path ../attention-next-gen-rl/log_dir/trainings
```

## visualize/plot log data using code in plotting.ipynb
Plots are saved in `figures/` and corresponding meta data to each plot is saved in `figure_metas/`.
