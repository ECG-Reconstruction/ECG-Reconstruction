from typing import Optional

import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np

from tensorflow.python.summary.summary_iterator import summary_iterator

import torch
from torch import nn

from ecg.util.lead import get_lead_names
LEAD_NAMES = get_lead_names()

def visualize_model(
    ax: mpl.axes.Axes,
    model: nn.Module,
    dataset,
    dataset_index: int,
    channel: int,
) -> None:
    items = dataset[dataset_index]
    data = items['input']
    data_tensor = torch.from_numpy(data[None, ...]).to(next(model.parameters()).device)
    with torch.no_grad():
        output_tensor = model(data_tensor)
    output = np.squeeze(output_tensor.cpu().numpy())
    time_all = np.linspace(0, 10, num=data.shape[-1], endpoint=False)

    channeled_target = items['filtered_signal'][channel]
    if channel in dataset.out_leads:
        output_map = {k: v for k, v in zip(list(dataset.out_leads), np.arange(len(output)))}
        channeled_output = output[output_map[channel]]
    elif (1 in dataset.in_leads) and (2 in dataset.in_leads):
        lead_II = items['filtered_signal'][1]
        lead_III = items['filtered_signal'][2]
        lead_I = lead_II - lead_III
        if channel == 0:  # lead_I
            channeled_output = lead_I
        elif channel == 3:  # AVR
            channeled_output = -0.5 * (lead_I + lead_II)
        elif channel == 4:  # AVL
            channeled_output = 0.5 * (lead_I - lead_III)
        elif channel == 5:  # AVF
            channeled_output = 0.5 * (lead_II + lead_III)
        else:
            channeled_output = None
    else:
        channeled_output = None

    time_output = time_all

    plt.title(f'Reconstruct for lead-{LEAD_NAMES[int(channel)]}')    
    plt.plot(time_all, channeled_target, "b", linewidth=1)
    if channeled_output is not None:
        plt.plot(time_output, channeled_output, "r", linewidth=1)


def visualize_model_all(
    experiments,
    ax: mpl.axes.Axes,
    model_dict: nn.Module,
    dataset,
    dataset_index: int,
    channel: int,
) -> None:
    # fig = plt.gcf()
    plt.suptitle(f'Reconstruct for lead-{LEAD_NAMES[int(channel)]}')
    for exp_idx, experiment in enumerate(experiments):
        model = model_dict[experiment]['model']
        items = dataset[dataset_index]
        data = items['input'] 
        data_tensor = torch.from_numpy(data[None, ...]).to(next(model.parameters()).device)
        with torch.no_grad():
            output_tensor = model(data_tensor)
        output = np.squeeze(output_tensor.cpu().numpy())
        time_all = np.linspace(0, 10, num=data.shape[-1], endpoint=False)
        # receptive_field = data.shape[-1] - output.shape[-1] + 1
        # i_start = receptive_field // 2
        # i_end = -(receptive_field - 1 - i_start)
        # time_output = time_all[i_start:i_end]

        channeled_target = items['filtered_signal'][channel]
        if channel in dataset.out_leads:
            output_map = {k: v for k, v in zip(list(dataset.out_leads), np.arange(len(output)))}
            channeled_output = output[output_map[channel]]
        elif (1 in dataset.in_leads) and (2 in dataset.in_leads):
            lead_II = items['filtered_signal'][1]
            lead_III = items['filtered_signal'][2]
            lead_I = lead_II - lead_III
            if channel == 0:  # lead_I
                channeled_output = lead_I
            elif channel == 3:  # AVR
                channeled_output = -0.5 * (lead_I + lead_II)
            elif channel == 4:  # AVL
                channeled_output = 0.5 * (lead_I - lead_III)
            elif channel == 5:  # AVF
                channeled_output = 0.5 * (lead_II + lead_III)
            else:
                channeled_output = None
        else:
            channeled_output = None
        plt.subplot(len(experiments), 1, exp_idx+1)
        plt.title(experiment.split('_')[0])
        plt.plot(time_all, channeled_target, "b", linewidth=1)
        if channeled_output is not None:
            plt.plot(time_all, channeled_output, "r", linewidth=1)






