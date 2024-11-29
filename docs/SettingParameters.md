# SettingParameters
The Setting class is used to configure the parameters for reading experimental data, signal preprocessing methods, and the setup of experimental tasks (including data partitioning).
## Random setting

**seed**

Set the random seed, which will affect the random partitioning during the subsequent data division process and the model training process.
## Dataset setting

**dataset**

Specify the dataset to read the data from.
For datasets that provide features, we will specify the original data and the provided features during the reading process. For example, for the seed dataset, to read the DE features provided by the seed dataset, this parameter should be set to "seed_de"; for PSD features, it should be set to "seed_psd"; and if you want to read the original dataset, it should be set to "seed_raw".

The available datasets are as follows:
```python
available_dataset = [  
    "seed_raw", "seediv_raw", "deap", "deap_raw", "hci", "dreamer", "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds"  
    , "seed_rasm", "seed_rasm_lds", "seed_asm", "seed_asm_lds", "seed_dcau", "seed_dcau_lds", "seediv_de_lds", "seediv_de_movingAve",  
    "seediv_psd_movingAve", "seediv_psd_lds", "faced_de", "faced_psd", "faced_de_lds", "faced_psd_lds"  
]
```
The datasets available for supported feature extraction are as follows:
```python
extract_dataset = {  
    "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds"  
    , "seed_rasm", "seed_rasm_lds", "seed_asm", "see_und_asm_lds", "seed_dcau", "seed_dcau_lds", "seediv_de_lds", "seediv_de_movingAve",  
    "seediv_psd_movingAve", "seediv_psd_lds", "faced_de", "faced_psd", "faced_de_lds", "faced_psd_lds"  
}
```

**dataset_path**

Specify the location where the dataset is stored.

## Preprocess setting

This section defines the operations for preprocessing the data.

**pass_band**

Data at indices 0 and 1 represent the lower and higher thresholds of bandpass filtering

**only_seg**

Indicate whether to perform only the sample segmentation operation. If set to true, only the sample segmentation will be executed without proceeding to the subsequent frequency domain feature extraction. If set to false, subsequent frequency domain feature extraction will be carried out.

**feature_type**

Specify the frequency bands for feature extraction.Currently, the supported features are de, de_lds, psd and psd_lds.

**extract_bands**

Set the frequency bands for extracting frequency features. The data format is a two-dimensional array, where each element at an index represents the range of each frequency band.

**time_window**

Set the duration in seconds for which to extract frequency features.

**overlap**

Specify the amount of overlap in seconds for the time window when extracting frequency features.

**sample_length**

When organizing the samples, specify the desired sample length. Let’s assume this parameter is set to t. When frequency features are extracted, the t extracted features will be arranged together in chronological order to form a sample with a shape of (t,channel,num_feat). When frequency features are not extracted, t EEG time slices will be organized together into a single sample with a shape of (channel,t).

**stride**

When organizing the samples, the stride for sliding is specified. If the stride is consistent with the sample length, it indicates that the samples are organized without overlap.

### label process

**bounds**

When using labels such as valence and arousal dimensions, the `bounds` parameter is used to define the thresholds for high and low. Values below `bounds[0]` are considered negative samples, while values above `bounds[1]` are considered positive samples.

**onehot**

Whether to use one-hot encoding for labels.

**label_used**

Specify which labels will be used for the upcoming task. For emotion tasks typically involving dimension-based labels, the default setting is `[valence]`, indicating that only the valence dimension is used. When set to `[arousal]`, it means the arousal dimension is used. When set to `[valence, arousal]`, both the valence and arousal dimensions are used simultaneously.

## Data split setting

**cross_trail**

Indicate whether to perform cross-trial settings; it is recommended to set this to true.

**experiment_mode**

Indicate whether the current task is performing subject-dependent operations, cross-subject operations, or cross-session operations. Different settings will affect the basic unit of data partitioning.

**split_type**

Indicate how to split the dataset into training, validation, and test sets.
Currently supported methods include *"kfold"* for k-fold cross-validation, *"leave-one-out"* for leave-one-out cross-validation, *"front-back"* for splitting data to train and val from the front and back (always used in seed and seediv), and *"train-val-test"* for training, validation, and test set partitioning.

**fold_num**  

Represents the number of folds to be used in k-fold cross-validation.

**fold_shuffle**  

Indicates whether to randomly shuffle the order during the folding process or to maintain the original order.

**front**  

Specifies the number of data samples to be allocated to the training set from the front.

**test_size**  

Indicates the amount of data to be allocated to the test set during the train-val-test split.

**val_size**  

Indicates the amount of data to be allocated to the validation set during the train-val-test split.

**sessions**

Which session data to use: when specifying [1], the data from the first session is used; when specifying [2,3], the data from second and third sessions is used.
