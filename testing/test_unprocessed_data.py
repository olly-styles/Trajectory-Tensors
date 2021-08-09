# Internal
from global_config.global_config import LABELED_TRACK_PATH, FUTURE_TRAJECTORY_LENGTH

# External
import os
import pandas as pd
import pytest
import numpy as np


all_data = []
for day in range(1, 21):
    data_path = os.path.join(LABELED_TRACK_PATH, "day_" + str(day) + ".json")
    all_data.append(pd.read_json(data_path))


@pytest.mark.parametrize("data", all_data)
def test_entrance_departure_different_cameras(data):
    """Tests that the individual does no re-appear in the same camera they were observerd"""
    assert np.sum(data["camera"] == data["next_cam"]) == 0


@pytest.mark.parametrize("data", all_data)
def test_transition_time(data):
    """Tests that transition time is greater than 0 but less than the maximum transition time"""
    assert np.sum(data["transition_time"] <= 0) == 0
    assert np.sum(data["transition_time"] > FUTURE_TRAJECTORY_LENGTH) == 0
