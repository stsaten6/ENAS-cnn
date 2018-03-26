srun -p Segmentation --gres=gpu:1 --ntasks-per-node=1 python -u main.py  --shared_initial_step=1000 --reward_c=80 --ppl_square=True

