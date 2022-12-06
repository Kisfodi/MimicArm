# @title Other imports and helper functions

import datetime
import os


def root_dir(folder="Models"):
    # Components of current time
    current_time = datetime.datetime.now()
    short_year = current_time.strftime("%y")
    current_month = current_time.strftime("%m")
    current_day = current_time.strftime("%d")
    current_hour = current_time.strftime("%H")
    current_minute = current_time.strftime("%M")

    # Structure of path: folder\yy_mm_dd\hh_mm\files
    current_path_root = f"{short_year}_{current_month}_{current_day}"
    current_hour_min = f"{current_hour}_{current_minute}"
    model_path = f"./{folder}/{current_path_root}/{current_hour_min}"

    # it does not exist, create one and return its string
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return model_path


def model_names(it_num):

    # Create name based on the number of iterations so far
    actor_name = f"ppo_actor_{it_num:02d}.pth"
    critic_name = f"ppo_critic_{it_num:02d}.pth"

    return actor_name, critic_name
