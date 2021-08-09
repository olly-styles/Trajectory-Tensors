import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
from global_config.global_config import CROSS_VALIDATION_WHICH_TARGETS_PATH, CROSS_VALIDATION_WHICH_PREDICTIONS_PATH

sns.set_style("whitegrid")

GROUND_TRUTH_PATH = CROSS_VALIDATION_WHICH_TARGETS_PATH
PREDICTED_PATH = CROSS_VALIDATION_WHICH_PREDICTIONS_PATH
NUM_FOLDS = 5
APPROXIMATE = True

MODELS = [
    "baselines/training_set_mean/",
    "baselines/most_similar_trajectory/",
    "baselines/hand_crafted_features/",
    "baselines/shortest_realworld_distance/",
    "coordinate_trajectories/gru/",
    "coordinate_trajectories/lstm/",
    "coordinate_trajectories/1dcnn/",
    "trajectory_tensors/2d1dcnn/",
    "trajectory_tensors/3dcnn/",
    "trajectory_tensors/cnn_gru/",
]

all_precisions = dict()
all_recalls = dict()
all_maps = dict()

# Figure asthetics
plt.figure()
sns.set(font_scale=1.7, style="ticks")
sns.set_palette(sns.color_palette("muted"))
fig, ax = plt.subplots()
fig.tight_layout(pad=1.8)
fig.set_size_inches(8, 5)
ax.set(xlabel="Recall", ylabel="Precision")


for model in MODELS:
    model_print_name = model.split("/")[1]
    precision = dict()
    recall = dict()
    maps = []
    for fold_num in range(1, NUM_FOLDS + 1):
        fold = str(fold_num)
        ground_truth_labels = np.load(GROUND_TRUTH_PATH + "/test_fold" + fold + ".npy")
        predictions = np.load(PREDICTED_PATH + "/" + model + "/test_fold" + fold + ".npy")
        precision[fold_num - 1], recall[fold_num - 1], _ = metrics.precision_recall_curve(
            ground_truth_labels.ravel(), predictions.ravel()
        )
        maps.append(metrics.average_precision_score(ground_truth_labels.flatten(), predictions.flatten()))
    print(model, maps)
    all_precision = np.unique(np.concatenate([precision[i] for i in range(NUM_FOLDS)]))

    # Then interpolate all ROC curves at this points
    mean_recall = np.zeros_like(all_precision)
    for i in range(NUM_FOLDS):
        mean_recall += interp(all_precision, precision[i], recall[i])

    # Finally average it and compute AUC
    mean_recall /= NUM_FOLDS

    all_precisions[model] = all_precision * 100
    all_recalls[model] = mean_recall * 100
    all_maps[model] = np.mean(maps) * 100

    if APPROXIMATE:
        print(all_recalls[model])
        print(len(all_recalls[model]))
        if len(all_precisions[model]) > 1000:
            last_precision = all_precisions[model][-1]
            last_recall = all_recalls[model][-1]
            all_precisions[model] = all_precisions[model][0::50]
            all_recalls[model] = all_recalls[model][0::50]
            all_precisions[model] = np.append(all_precisions[model], last_precision)
            all_recalls[model] = np.append(all_recalls[model], last_recall)
        print(all_recalls[model])
        print(len(all_recalls[model]))
    sns.lineplot(all_precisions[model], all_recalls[model], label=str(model_print_name), linewidth=3)

plt.xlim([0.0, 100])
plt.ylim([0.0, 105])
plt.title("WHICH camera: Precision-recall plot")
plt.legend(loc="lower left")
sns.despine()
plt.savefig("./figures/roc_legend.png")

plt.xlim([0.0, 100])
plt.ylim([0.0, 105])
plt.title("WHICH camera: Precision-recall plot")
plt.legend(loc="lower left")
ax.get_legend().remove()
sns.despine()
plt.savefig("./figures/roc_which.png")

plt.xlim([70, 100])
plt.ylim([60, 100])
plt.title("WHICH camera: Precision-recall plot")
plt.legend(loc="lower left")
ax.get_legend().remove()
sns.despine()
plt.savefig("./figures/roc_which_zoomed.png")
