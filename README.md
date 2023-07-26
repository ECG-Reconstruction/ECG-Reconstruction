# ECG Reconstruction Using Neural Networks

## Environment Setup

### Package Installation

This project uses [conda](https://docs.conda.io) to manage the required packages. To create a new conda environment with the required packages, please refer to the `.yml` files under the `docs/` directory. E.g., `conda create -n ecg -f docs/OS_installation/DEVICE_environment.yaml`.

- For XPU, there is a [known issue](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/performance_tuning/known_issues.html): 

  OSError: `libmkl_intel_lp64.so.2`: cannot open shared object file: No such file or directory. 

  As a walkaround, `export LD_LIBRARY_PATH=$CONDA_ROOT/envs/ecg/lib/`.
- For windows with cuda, after installing from `docs/windows_installation/cuda_environment_base.yaml`, activate the newly created environment and run `docs/windows_installation/cuda_environment_nvidia_pytorch.bat` to install pytorch and its required cuda environment into the conda environment.

### Dataset Preparation

This project uses several public ECG datasets including [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) and [CODE-15%](https://zenodo.org/record/4916206) to train and test ECG reconstruction models. In order to compensate for the differences between these datasets and to load the data efficiently without being bounded by IO, all ECG data are converted to a custom format as follows.

- The training, validation, and test splits of a dataset are stored in 3 `.hdf5` files.
- In each `.hdf5` file, there should be at least a dataset named `signal` containing ECG signals. The dataset must has shape `(num_elements, num_channels, signal_length)`.
- The `signal` dataset should have at least a `sampling_rate` attribute, which stores the sampling rate of the ECG signals in Hertz.
  > The sampling rate is required to properly denoise the ECG signals, which is a necessary data preprocessing step.

Datasets with the above format can be consumed by the `Dataset` class implemented in `src/ecg/dataset.py`.

The [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset is used to exemplify the process of preparing a dataset. In `src/ecg/dataset-gen/ptb-xl`:

- The `download.sh` script downloads the original dataset from the Internet.
- The `generate.py` script takes the downloaded original dataset, splits it into training, validation, and test sets, and then stores the data in the aforementioned custom format.

The generated `.hdf5` files should be copied/symlinked to a subdirectory under `src/datasets`, say `src/datasets/ptb-xl/{train,validation,test}.hdf5`. Our code follows this convention to look for dataset files.

## Model Training

The [training notebook](./src/notebooks/demo_training_on_ptbxl.ipynb) provides an example of how models can be trained on the PTB-XL dataset. The notebook will walk through the essential steps of how configurations can be set up to train a new model. The checkpoints and training curves will be saved in the [checkpoint] and [tensorboard] folder respectively.

The [tuning notebook](./src/notebooks/demo_tune_model.ipynb) provides an example of how the hyperparameter of a model can be tuned via the Optuna Framework. During tuning, no checkpoint or model will be saved. The tuning will result in the best hyperparameter, which will then be used to train the model from bottom up with the approach described in [training notebook](./src/notebooks/demo_training_on_ptbxl.ipynb)

## Model Testing

After being trained, the model can be further analyzed in the [testing notebook](./src/notebooks/demo_testing_and_visualize.ipynb)
