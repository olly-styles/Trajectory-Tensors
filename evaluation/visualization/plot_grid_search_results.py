import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from global_config.global_config import (
    CROSS_VALIDATION_WHICH_GRID_SEARCH_PATH,
    CROSS_VALIDATION_WHEN_GRID_SEARCH_PATH,
    CROSS_VALIDATION_WHERE_GRID_SEARCH_PATH,
)

sns.set()
task = "where"

for model in ["2d1dcnn", "3dcnn", "cnn_gru"]:
    if task == "which":
        grid_search_path = CROSS_VALIDATION_WHICH_GRID_SEARCH_PATH
    elif task == "when":
        grid_search_path = CROSS_VALIDATION_WHEN_GRID_SEARCH_PATH
    else:
        grid_search_path = CROSS_VALIDATION_WHERE_GRID_SEARCH_PATH

    grid_search_results_path = os.path.join(grid_search_path, model)

    if task == "which":
        grid_search_results = pd.read_csv(os.path.join(grid_search_results_path, "results.csv"))
    else:
        grid_search_results_1 = pd.read_csv(os.path.join(grid_search_results_path, "results_scale_1.csv"))
        grid_search_results_2 = pd.read_csv(os.path.join(grid_search_results_path, "results_scale_2.csv"))
        grid_search_results_3 = pd.read_csv(os.path.join(grid_search_results_path, "results_scale_3.csv"))
        grid_search_results = grid_search_results_1.append(grid_search_results_2, ignore_index=True).append(
            grid_search_results_3, ignore_index=True
        )

    grid_search_results[["heatmap_scale", "heatmap_smoothing_sigma"]] = grid_search_results[
        ["heatmap_scale", "heatmap_smoothing_sigma"]
    ].astype(int)
    grid_search_results["test_ap"] = grid_search_results["test_ap"] * 100
    grid_search_results = grid_search_results.groupby(["heatmap_scale", "heatmap_smoothing_sigma"]).mean().reset_index()
    grid_search_results = grid_search_results.pivot("heatmap_scale", "heatmap_smoothing_sigma", "test_ap")
    print(model, grid_search_results.max().max(), grid_search_results.min().min())
    print(grid_search_results)
    ax = sns.heatmap(grid_search_results, annot=False, cmap="RdBu_r", vmin=21, vmax=38)
    ax.invert_yaxis()
    plt.savefig("./figures/grid_search_results/" + task + "_" + model + ".png")
    plt.clf()
