#!/bin/bash
# bash scheduler.sh run_configs/<file>.csv <number_experiments>
for ((i=0;i<$2;i++));
do
   python3 run.py --args_from_csv $1 --csv_row $i
done