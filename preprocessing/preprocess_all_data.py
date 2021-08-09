from preprocessing.src.input_data_preprocessors import (
    get_coordinate_trajectory_inputs,
    get_trajectory_tensor_inputs,
    get_trajectory_tensor_single_view_inputs,
)
from preprocessing.src.target_preprocessors import get_which_targets, get_when_targets, get_where_targets
from preprocessing.src.multi_target_preprocessor import get_all_multi_target_inputs
from preprocessing.src.train_set_split import split_data_train_val_test, multi_target_split_data_train_val_test
from global_config.global_config import BASE_HEATMAP_SIZE

# # Targets
# get_which_targets()
# get_when_targets()
# get_where_targets()
#
# # Input data
# get_coordinate_trajectory_inputs()
#
# for heatmap_scale in [1, 2, 3]:
#     get_trajectory_tensor_inputs((BASE_HEATMAP_SIZE[0] * heatmap_scale, BASE_HEATMAP_SIZE[1] * heatmap_scale))
#     get_all_multi_target_inputs((BASE_HEATMAP_SIZE[0] * heatmap_scale, BASE_HEATMAP_SIZE[1] * heatmap_scale))
#     get_trajectory_tensor_single_view_inputs(
#         (BASE_HEATMAP_SIZE[0] * heatmap_scale, BASE_HEATMAP_SIZE[1] * heatmap_scale)
#     )


# Split data
for heatmap_scale in [1, 2, 3]:
    for multi_view in [True, False]:
        split_data_train_val_test(heatmap_scale, multi_view)
    multi_target_split_data_train_val_test(heatmap_scale)
