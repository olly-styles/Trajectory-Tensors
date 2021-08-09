import pandas as pd
import os
import numpy as np
from global_config.global_config import DATA_PATH, NUM_DAYS

bounding_boxes = pd.read_csv(os.path.join(DATA_PATH, "all_bounding_boxes", "all_bounding_boxes_day_1.csv"))
bounding_boxes["day"] = 1
for day in range(2, NUM_DAYS + 1):
    day_bounding_boxes = pd.read_csv(
        os.path.join(DATA_PATH, "all_bounding_boxes", "all_bounding_boxes_day_" + str(day) + ".csv")
    )
    day_bounding_boxes["day"] = day
    bounding_boxes = bounding_boxes.append(day_bounding_boxes)

person_density = np.mean(bounding_boxes.groupby(["frame_num", "hour", "camera", "day"]).count())["track"]
print("Average person density:", person_density)
