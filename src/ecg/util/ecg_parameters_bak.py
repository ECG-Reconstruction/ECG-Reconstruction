import neurokit2 as nk
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

LEADS = np.array(["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])
LEAD2index = {x: i for i, x in enumerate(LEADS)}


def get_valid_locations(wave_peaks, rpeaks):
    rpeaks = np.array(rpeaks)
    # get rid of some locations where no peaks is found
    cond = []
    for k in wave_peaks:
        cur_peaks = np.array(wave_peaks[k])
        wave_peaks[k] = cur_peaks
        cur_valid_peaks = np.isnan(cur_peaks)
        cond.append(cur_valid_peaks)
    # get rid of situations where the peak sequence for each type is not of equal length
    if min([len(k) for k in cond]) != max([len(k) for k in cond]):
        # raise ValueError("The number of peaks found for each wave is not the same")
        ret = np.zeros(len(rpeaks), dtype=bool)
        return ret
    cond = np.vstack(cond)
    valid_locations = cond.T.mean(-1) == 0
    return valid_locations


def get_diff_params(params_label, params_pred, sampling_rate=500):
    wave_peaks_label = params_label["wave_peaks"]
    wave_peaks_pred = params_pred["wave_peaks"]
    rpeaks_label = params_label["rpeaks"]
    rpeaks_pred = params_pred["rpeaks"]
    valid_location_label = params_label["valid_location"]
    valid_location_pred = params_pred["valid_location"]

    rr_label = rpeaks_label[1:] - rpeaks_label[:-1]
    valid_rr_labels = rr_label[np.isnan(rr_label) == False]
    valid_rr_label = np.mean(valid_rr_labels)
    valid_hr_label = sampling_rate * 60 / valid_rr_label
    rr_pred = rpeaks_pred[1:] - rpeaks_pred[:-1]
    valid_rr_preds = rr_pred[np.isnan(rr_pred) == False]
    valid_rr_pred = np.mean(valid_rr_preds)
    valid_hr_pred = sampling_rate * 60 / valid_rr_pred

    pr_diffs, qrs_diffs, qt_diffs, qtcb_diffs, qtcr_diffs = [], [], [], [], []

    pr_labels, pr_preds = [], []
    qrs_labels, qrs_preds = [], []
    qt_labels, qt_preds = [], []
    qtcb_labels, qtcb_preds = [], []
    qtcr_labels, qtcr_preds = [], []

    for i in range(len(valid_location_pred)):
        # if both are valid, then we can compare
        if valid_location_pred[i] and valid_location_label[i]:
            # this pr is not exact: it measures from P_onset to Q_peak
            pr_label = wave_peaks_label["ECG_Q_Peaks"][i] - wave_peaks_label["ECG_P_Onsets"][i]
            pr_pred = wave_peaks_pred["ECG_Q_Peaks"][i] - wave_peaks_pred["ECG_P_Onsets"][i]
            pr_labels.append(pr_label)
            pr_preds.append(pr_pred)
            pr_diff = pr_label - pr_pred
            if pr_label != 0:
                pr_diffs.append(pr_diff / pr_label)

            # this qrs is also not exact: it measures from Q_peak to S_peak
            qrs_label = wave_peaks_label["ECG_S_Peaks"][i] - wave_peaks_label["ECG_Q_Peaks"][i]
            qrs_pred = wave_peaks_pred["ECG_S_Peaks"][i] - wave_peaks_pred["ECG_Q_Peaks"][i]
            qrs_labels.append(qrs_label)
            qrs_preds.append(qrs_pred)
            qrs_diff = qrs_label - qrs_pred
            if qrs_label != 0:
                qrs_diffs.append(qrs_diff / qrs_label)

            # this qt is also not exact?: it measures from Q_peak to T_offset
            qt_label = wave_peaks_label["ECG_T_Offsets"][i] - wave_peaks_label["ECG_Q_Peaks"][i]
            qt_pred = wave_peaks_pred["ECG_T_Offsets"][i] - wave_peaks_pred["ECG_Q_Peaks"][i]
            qt_labels.append(qt_label)
            qt_preds.append(qt_pred)
            qt_diff = qt_label - qt_pred
            if qt_label != 0:
                qt_diffs.append(qt_diff / qt_label)

            qtcb_label = qt_label / np.sqrt(valid_rr_label)
            qtcb_pred = qt_pred / np.sqrt(valid_rr_pred)
            qtcb_labels.append(qtcb_label)
            qtcb_preds.append(qtcb_pred)
            qtcb_diff = qtcb_label - qtcb_pred
            if qtcb_label != 0:
                qtcb_diffs.append(qtcb_diff / qtcb_label)

            qtcr_label = qt_label * (120 + valid_hr_label) / 180
            qtcr_pred = qt_pred * (120 + valid_hr_pred) / 180
            qtcr_labels.append(qtcr_label)
            qtcr_preds.append(qtcr_pred)
            qtcr_diff = qtcr_label - qtcr_pred
            if qtcr_label != 0:
                qtcr_diffs.append(qtcr_diff / qtcr_label)

    return {
        "pr_diff": pr_diffs,
        "qrs_diff": qrs_diffs,
        "qt_diff": qt_diffs,
        "qtcb_diff": qtcb_diffs,
        "qtcr_diff": qtcr_diffs,
        "rr_diff": [(valid_rr_label - valid_rr_pred) / valid_rr_label],
        "hr_diff": [(valid_hr_label - valid_hr_pred) / valid_hr_label],
    }, {
        "hr_label": sampling_rate * 60 / valid_rr_labels,
        "hr_pred": sampling_rate * 60 / valid_rr_preds,
        "pr_label": pr_labels,
        "pr_pred": pr_preds,
        "qrs_label": qrs_labels,
        "qrs_pred": qrs_preds,
        "qt_label": qt_labels,
        "qt_pred": qt_preds,
        "qtcb_label": qtcb_labels,
        "qtcb_pred": qtcb_preds,
        "qtcr_label": qtcr_labels,
        "qtcr_pred": qtcr_preds,
    }

def find_rpeak_between(t1, t2, ecg, previous_rpeak):
    # only work for V2 probably
    threshold = (t2 - t1) // 2
    percentiles = 5

    if t1 < 0:
        t1 = 0
    r_peak_idx = np.argmin(ecg[t1:t2]) + t1

    if previous_rpeak:
        if r_peak_idx - previous_rpeak <= threshold:
            return None

        previous_rpeak_height = np.max(ecg[previous_rpeak - threshold // 4 : previous_rpeak + threshold // 4]) - np.min(ecg[previous_rpeak - threshold // 2 : previous_rpeak + threshold // 2])
        current_rpeak_height = np.max(ecg[r_peak_idx - threshold // 4 : r_peak_idx + threshold // 4]) - ecg[r_peak_idx]

        if current_rpeak_height < 0.5 * previous_rpeak_height:
            return None

    if ecg[r_peak_idx] > np.percentile(ecg, percentiles):
        return None

    return r_peak_idx

def find_rpeaks(dataset, test_lead_label, test_lead_pred):
    _, rpeaks_dict_label = nk.ecg_peaks(test_lead_label, dataset.sampling_rate)
    _, rpeaks_dict_pred = nk.ecg_peaks(test_lead_pred, dataset.sampling_rate)

    index_label = 0
    index_pred = 0

    # DEBUG = True

    # if len(rpeaks_dict_label["ECG_R_Peaks"]) != len(rpeaks_dict_pred["ECG_R_Peaks"]):
    #     DEBUG = True
    # else:
    #     DEBUG = False

    # if DEBUG:
    #     print("\n\n")
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(test_lead_label, color="k", linewidth=0.5)
    #     plt.plot(test_lead_pred, color="r", linewidth=0.2)
    #     plt.vlines(rpeaks_dict_label["ECG_R_Peaks"], ymin=-2, ymax=-1.5, colors="r")
    #     plt.vlines(rpeaks_dict_pred["ECG_R_Peaks"], ymin=-1.5, ymax=-1, colors="g")
    #     plt.show()

    # min-matching algorithm
    TOLERANCE = 0.3
    while (index_label < len(rpeaks_dict_label["ECG_R_Peaks"])) and (index_pred < len(rpeaks_dict_pred["ECG_R_Peaks"])):
        rr_label = rpeaks_dict_label["ECG_R_Peaks"][1:] - rpeaks_dict_label["ECG_R_Peaks"][:-1]
        rr_pred = rpeaks_dict_pred["ECG_R_Peaks"][1:] - rpeaks_dict_pred["ECG_R_Peaks"][:-1]

        median_rr = np.median(np.concatenate((rr_label, rr_pred)))
        threshold = int(median_rr * TOLERANCE)

        if np.abs(rpeaks_dict_label["ECG_R_Peaks"][index_label] - rpeaks_dict_pred["ECG_R_Peaks"][index_pred]) > threshold:
            if rpeaks_dict_pred["ECG_R_Peaks"][index_pred] - rpeaks_dict_label["ECG_R_Peaks"][index_label] > threshold:
                if (index_label > 0 and median_rr * (1 - TOLERANCE) < rr_label[index_label - 1] < median_rr * (1 + TOLERANCE)) or (index_label < len(rr_label) and median_rr * (1 - TOLERANCE) < rr_label[index_label] < median_rr * (1 + TOLERANCE)):
                    rpeak_idx = find_rpeak_between(
                        rpeaks_dict_label["ECG_R_Peaks"][index_label] - threshold,
                        rpeaks_dict_label["ECG_R_Peaks"][index_label] + threshold,
                        test_lead_pred,
                        rpeaks_dict_pred["ECG_R_Peaks"][index_pred - 1] if index_pred > 0 else None,
                    )
                    if rpeak_idx:
                        rpeaks_dict_pred["ECG_R_Peaks"] = np.insert(rpeaks_dict_pred["ECG_R_Peaks"], index_pred, rpeak_idx)
                        continue

                rpeaks_dict_label["ECG_R_Peaks"] = np.delete(rpeaks_dict_label["ECG_R_Peaks"], index_label)

            else:
                if (index_pred > 0 and median_rr * (1 - TOLERANCE) < rr_pred[index_pred - 1] < median_rr * (1 + TOLERANCE)) or (index_pred < len(rr_pred) and median_rr * (1 - TOLERANCE) < rr_pred[index_pred] < median_rr * (1 + TOLERANCE)):
                    rpeak_idx = find_rpeak_between(
                        rpeaks_dict_pred["ECG_R_Peaks"][index_pred] - threshold,
                        rpeaks_dict_pred["ECG_R_Peaks"][index_pred] + threshold,
                        test_lead_label,
                        rpeaks_dict_label["ECG_R_Peaks"][index_label - 1] if index_label > 0 else None,
                    )
                    if rpeak_idx:
                        rpeaks_dict_label["ECG_R_Peaks"] = np.insert(rpeaks_dict_label["ECG_R_Peaks"], index_label, rpeak_idx)
                        continue

                rpeaks_dict_pred["ECG_R_Peaks"] = np.delete(rpeaks_dict_pred["ECG_R_Peaks"], index_pred)
        else:
            index_label += 1
            index_pred += 1

    while len(rpeaks_dict_label["ECG_R_Peaks"]) > len(rpeaks_dict_pred["ECG_R_Peaks"]):
        # code copied from above
        rr_label = rpeaks_dict_label["ECG_R_Peaks"][1:] - rpeaks_dict_label["ECG_R_Peaks"][:-1]
        rr_pred = rpeaks_dict_pred["ECG_R_Peaks"][1:] - rpeaks_dict_pred["ECG_R_Peaks"][:-1]

        median_rr = np.median(np.concatenate((rr_label, rr_pred)))
        threshold = int(median_rr * TOLERANCE)

        if index_label < len(rr_label) and median_rr * (1 - TOLERANCE) < rr_label[index_label] < median_rr * (1 + TOLERANCE):
            rpeak_idx = find_rpeak_between(
                rpeaks_dict_label["ECG_R_Peaks"][index_label] - threshold,
                rpeaks_dict_label["ECG_R_Peaks"][index_label] + threshold,
                test_lead_pred,
                rpeaks_dict_pred["ECG_R_Peaks"][index_pred - 1] if index_pred > 0 else None,
            )

            if rpeak_idx:
                rpeaks_dict_pred["ECG_R_Peaks"] = np.insert(rpeaks_dict_pred["ECG_R_Peaks"], index_pred, rpeak_idx)
                index_label += 1
                index_pred += 1
                continue
        rpeaks_dict_label["ECG_R_Peaks"] = np.delete(rpeaks_dict_label["ECG_R_Peaks"], -1)

    while len(rpeaks_dict_label["ECG_R_Peaks"]) < len(rpeaks_dict_pred["ECG_R_Peaks"]):
        # code copied from above
        rr_label = rpeaks_dict_label["ECG_R_Peaks"][1:] - rpeaks_dict_label["ECG_R_Peaks"][:-1]
        rr_pred = rpeaks_dict_pred["ECG_R_Peaks"][1:] - rpeaks_dict_pred["ECG_R_Peaks"][:-1]

        median_rr = np.median(np.concatenate((rr_label, rr_pred)))
        threshold = int(median_rr * TOLERANCE)

        if index_pred < len(rr_pred) and median_rr * (1 - TOLERANCE) < rr_pred[index_pred] < median_rr * (1 + TOLERANCE):
            rpeak_idx = find_rpeak_between(
                rpeaks_dict_pred["ECG_R_Peaks"][index_pred] - threshold,
                rpeaks_dict_pred["ECG_R_Peaks"][index_pred] + threshold,
                test_lead_label,
                rpeaks_dict_label["ECG_R_Peaks"][index_label - 1] if index_label > 0 else None,
            )

            if rpeak_idx:
                rpeaks_dict_label["ECG_R_Peaks"] = np.insert(rpeaks_dict_label["ECG_R_Peaks"], index_label, rpeak_idx)
                index_label += 1
                index_pred += 1
                continue
        rpeaks_dict_pred["ECG_R_Peaks"] = np.delete(rpeaks_dict_pred["ECG_R_Peaks"], -1)

    # if DEBUG:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(test_lead_label, color="k", linewidth=0.5)
    #     plt.plot(test_lead_pred, color="r", linewidth=0.2)
    #     plt.vlines(rpeaks_dict_label["ECG_R_Peaks"], ymin=-2, ymax=-1.5, colors="r")
    #     plt.vlines(rpeaks_dict_pred["ECG_R_Peaks"], ymin=-1.5, ymax=-1, colors="g")
    #     plt.show()

    return rpeaks_dict_label, rpeaks_dict_pred


# more details: https://neuropsychology.github.io/NeuroKit/examples/ecg_delineate/ecg_delineate.html
def compare_parameters(
    model: nn.Module,
    dataset,
    dataset_index: int,
    test_lead="V2",
) -> None:
    signals_input, signals_all = (
        dataset[dataset_index]["input"],
        dataset[dataset_index]["filtered_signal"],
    )
    signals_input_tensor = torch.from_numpy(signals_input[None, ...]).to(next(model.parameters()).device)
    with torch.no_grad():
        signals_output_tensor = model(signals_input_tensor)
    signals_output = np.squeeze(signals_output_tensor.cpu().numpy())
    signals_output_map = {k: v for k, v in zip(list(dataset.out_leads), np.arange(len(signals_output)))}

    test_lead_label = signals_all[LEAD2index[test_lead]]
    test_lead_pred = signals_output[signals_output_map[LEAD2index[test_lead]]]

    rpeaks_dict_label, rpeaks_dict_pred = find_rpeaks(dataset, test_lead_label, test_lead_pred)

    rpeaks_label = rpeaks_dict_label["ECG_R_Peaks"]
    rpeaks_pred = rpeaks_dict_pred["ECG_R_Peaks"]
    assert len(rpeaks_label) == len(rpeaks_pred)

    if len(rpeaks_pred) <= 1:
        return {}

    # METHOD='peak'
    METHOD = "dwt"

    def get_PQRST(ecg_signal, rpeaks, sampling_rate=500):
        try:
            _, wave_peaks = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, method=METHOD)
        except:
            return None
        return wave_peaks

    # using the pruned peak lists to get other classes of peaks, which will assert to be the same number
    wave_peaks_label = get_PQRST(test_lead_label, rpeaks_dict_label, dataset.sampling_rate)
    wave_peaks_pred = get_PQRST(test_lead_pred, rpeaks_dict_pred, dataset.sampling_rate)

    if wave_peaks_label is None or wave_peaks_pred is None:
        return {}

    assert len(wave_peaks_label) == len(wave_peaks_pred)

    valid_location_label = get_valid_locations(wave_peaks_label, rpeaks_label)
    valid_location_pred = get_valid_locations(wave_peaks_pred, rpeaks_pred)

    params_label = {
        "wave_peaks": wave_peaks_label,
        "rpeaks": rpeaks_label,
        "valid_location": valid_location_label,
    }
    params_pred = {
        "wave_peaks": wave_peaks_pred,
        "rpeaks": rpeaks_pred,
        "valid_location": valid_location_pred,
    }

    diff_params, params = get_diff_params(params_label, params_pred, sampling_rate=dataset.sampling_rate)

    diff_dict = {}
    for k in ["pr_diff", "qrs_diff", "qtcb_diff", "qtcr_diff", "rr_diff", "hr_diff"]:
        diff_dict[f"{k}"] = np.abs(diff_params[k])

    return diff_dict, params
