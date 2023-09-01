# Attention, Learn to Solve Routing Problems!

This is an adjusted codebase of Kool et al.'s Attention, Learn to Solve Routing Problems!.
It was adjusted to the concepts of our main branch to be able to achieve reproducable, comparable results to our other algorithms.

## train model options
```
bash scheduler.sh run_configs/<config_file> <x_first_rows>
```
OR
```
python3 run.py --args_from_csv <path_to_csv> --csv_row <row_to_run>
```

## evaluate trained model
`python3 run.py --load_path outputs/<path_to_model> --eval_only`
