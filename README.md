# LibEER: A Comprehensive Benchmark and Algorithm Library for EEG-based Emotion Recognition
## Project Description
LibEER estabilshes a unified evaluation framework with standardized experimental settings, enabling unbiased evaluations of over representative deep learning-based EER models across the four most commonly used datasets.
- **Standardized Benchmark**: LibEER provides a unified benchmark for fair comparisons in EER research, addressing inconsistencies in datasets, settings, and metrics, making it easier to evaluate various models.
- **Comprehensive Algorithm Library**: The framework includes implementations of over ten deep learning models, covering a wide range of architectures (CNN, RNN, GNN, and Transformers), making it highly versatile for EEG analysis.
- **Efficient Preprocessing and Training**: LibEER offers various preprocessing techniques and customizable settings, enabling efficient model fine-tuning, lowering the entry barrier for researchers, and boosting research efficiency.
- **Extensive Dataset Support**: LibEER gives standardized access to major datasets like SEED, SEED-IV, DEAP, and MAHNOB-HCI, supporting both subject-dependent and cross-subject evaluations, with plans to add more datasets in the future.
## Requirements
To run this project, you'll need the following dependencies:
1. Python 3.x recommended
2. Dependencies: You can install the required Python packages by running:
```cmd
pip install -r requirements.txt
```
## Usage

LibEER implements three main modules: data loader, data splitting, and model training and evaluation. It also incorporates many representative algorithms in the field of EEG-based Emotion Recognition. The specific usage is detailed as follows. Additionally, to make it easier for users, we have implemented several one-step methods for common data processing and data splitting tasks. For more details, please refer to the quick start of this chapter.
![[Pasted image 20241018114425.png]]
### Quick Start
To facilitate easy use for users, we implemented the **Setting** class, allowing one-stop data usage through parameter configuration. Additionally, we have preconfigured many common experimental settings to help users quickly get started. 
Data is achieved through the **Setting** class:
```python
from config.setting import Setting  
  
setting = Setting(dataset='deap',  
                  dataset_path='DEAP/data_preprocessed_python',  
                  pass_band=[0.3, 50],  
                  extract_bands=[[0.5,4],[4,8],[8,14],[14,30],[30,50]],   
                  time_window=1,   
                  overlap=0,  
                  sample_length=1,   
                  stride=1,   
                  seed=2024,   
                  feature_type='de_lds',   
                  only_seg=False,   
                  experiment_mode="subject-dependent",  
                  split_type='train-val-test',   
                  test_size=0.2,   
                  val_size=0.2)
data, label, channels, feature_dim, num_classes = get_data(setting)
data, label = merge_to_part(data, label, setting)
for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):  
    tts = get_split_index(data_i, label_i, setting)  
    for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):  
	    train_data, train_label, val_data, val_label, test_data, test_label = \  
    index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
		model = Model['DGCNN'](channels, feature_dim, num_classes)  
        # Train one round using the train one round function defined in the model  
        dataset_train = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))  
        dataset_val = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_label))  
        dataset_test = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))  
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)  
        criterion = nn.CrossEntropyLoss()  
        loss_func = NewSparseL2Regularization(0.01).to(device)  
        round_metric = train(model=model, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test, device=device,  
                             output_dir=output_dir, metrics=args.metrics, metric_choose=args.metric_choose, optimizer=optimizer,  
                             batch_size=args.batch_size, epochs=args.epochs, criterion=criterion, loss_func=loss_func, loss_param=model)  
        best_metrics.append(round_metric)  
    result_log(args, best_metrics)
```
Currently supported predefined setting classes:
<div>
<table border="0" cellpadding="0" cellspacing="0" width="818" style="border-collapse:
 collapse;table-layout:fixed;width:612pt">
 <colgroup><col width="109" style="mso-width-source:userset;mso-width-alt:3474;width:81pt">
 <col width="122" style="mso-width-source:userset;mso-width-alt:3894;width:91pt">
 <col width="587" style="mso-width-source:userset;mso-width-alt:18779;width:440pt">
 </colgroup><tbody><tr height="20" style="height:15.0pt">
  <td rowspan="18" height="989" class="xl70" width="109" style="height:740.7pt;
  width:81pt">SEED Dataset</td>
  <td colspan="2" class="xl67" width="709" style="border-right:.5pt solid black;
  border-left:none;width:531pt;white-space:no-wrap">subject-dependent, train
  val test split<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl69" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_dependent_train_val_test_setting</td>
 </tr>
 <tr height="132" style="height:99.0pt">
  <td height="132" class="xl65" style="height:99.0pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  the 15 subjects in the dataset, perform one round of training and testing for
  each individual subject’s session data. Specifically, randomly select any 9
  trials from the 15 trials in a session for the training set, 3 trials for the
  validation set, and 3 trials for the test set.<br>
    If using data from two sessions, then for one subject’s session, randomly
  select any 9 trials from the 15 trials as the training set, 3 trials as the
  validation set, and 3 trials as the test set for one training session. In
  total, this results in 30 training sessions (num_session * num_subject: 2 *
  15).</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-independent,<span style="mso-spacerun:yes">&nbsp; </span>train val test split<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl69" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_independent_train_val_test_setting</td>
 </tr>
 <tr height="38" style="height:28.3pt">
  <td height="38" class="xl65" style="height:28.3pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt"><span style="font-variant-ligatures: normal;font-variant-caps: normal;orphans: 2;
  white-space:pre-wrap;widows: 2;-webkit-text-stroke-width: 0px;text-decoration-thickness: initial;
  text-decoration-style: initial;text-decoration-color: initial">For the 15
  subjects in the dataset, randomly select 9 subjects from the first session as
  the training set, 3 subjects as the validation set, and 3 subjects as the
  test set.</span></td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-dependent, front
  nine trials and back six trials<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_dependent_front_back_setting</td>
 </tr>
 <tr height="151" style="height:113.15pt">
  <td height="151" class="xl65" style="height:113.15pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  the 15 subjects in the dataset, a round of training and testing is performed
  for each subject's session data. Specifically, for one session of a subject,
  the first 9 trials out of the 15 trials are used as the training set, and the
  last 6 trials as the test set for training.<br>
    If using data from two sessions, the first 9 trials out of 15 from one
  session of a subject are used as the training set, and the last 6 trials as
  the test set for one round of training, resulting in a total of 30 training
  rounds (num_session * num_subject: 2 * 15). Therefore, using data from three
  sessions requires 45 training rounds, while one session requires 15 training
  rounds.</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-dependent, five
  fold cross-validation<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_dependent_5fold_setting</td>
 </tr>
 <tr height="189" style="height:141.45pt">
  <td height="189" class="xl65" style="height:141.45pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  the 15 subjects in the dataset, perform five rounds of training and testing
  for each individual subject's session data. Specifically, for one session of
  a subject, take three trials sequentially from the 15 trials as the training
  set, while the remaining trials serve as the test set for one training
  session. This process continues until all trials have been used as the test
  set, resulting in a total of five training sessions.<br>
    If using data from two sessions, for one subject’s session with 15 trials,
  first take the first three trials as the test set and the remaining 12 trials
  as the training set for one session. Then take trials 4, 5, and 6 as the test
  set, with the other 12 trials as the training set for another session. This
  continues until all trials have been used as the test set. The data from two
  sessions will require a total of 2 * 15 * 5 training rounds.</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-independent,
  leave one out:</td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_independent_leave_one_out_setting</td>
 </tr>
 <tr height="170" style="height:127.3pt">
  <td height="170" class="xl65" style="height:127.3pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  all the data in the dataset, perform 15 training rounds using all 15 subjects
  from one session. Specifically, use the data from one subject in a session as
  the test set, while the data from the remaining subjects serves as the
  training set for one training round. This process continues until all
  subjects have been used as the test set, resulting in a total of fifteen
  training rounds.<br>
    If using data from two sessions, start with the data from the first
  session, alternately selecting one subject's data as the test set while using
  the data from the other subjects as the training set for training. This
  results in 15 training sessions for one session. With two sessions, a total
  of 2 * 15 training rounds will be conducted.</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">cross-session<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_cross_session_setting</td>
 </tr>
 <tr height="75" style="height:56.6pt">
  <td height="75" class="xl65" style="height:56.6pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  all the data in the dataset, alternately select the data from one session as
  the test set while using the data from the other two sessions as the training
  set, continuing this process until all sessions have been used as the test
  set. A total of three training sessions will be conducted.</td>
 </tr>
 <!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="109" style="width:81pt"></td>
  <td width="122" style="width:91pt"></td>
  <td width="587" style="width:440pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>
</div>
...

### Detailed usage
To enable users to have more precise control and use of intermediate results, this section presents the detailed usage of the three main modules. If the settings class does not meet the requirements of your experiment, you can refer to the usage methods below.
#### Data loader
In the data loader, LibEER supports four EEG emotion recognition datasets: SEED, SEED-IV, DEAP, and HCI. It also provides support for various data preprocessing methods and a range of feature extraction techniques. The following example demonstrates how to use LibEER to load a dataset and preprocess the data. Specifically, it extracts 1-second DE (Differential Entropy) features from the DEAP dataset, after baseline removal and band-pass filtering between 0.3-50Hz, across five frequency bands.
```python
# get data, baseline, label, sample rate of data,  channels of data using get_uniform_data() function  
unified_data, baseline, label, sample_rate, channels = get_uniform_data(dataset="deap", dataset_path="DEAP/data_preprocessed_python")
# remove baseline  
data = baseline_removal(unified_data, baseline)  
# using a 0.3-50 Hz bandpass filter to process the data  
data = bandpass_filter(data, sample_rate,  pass_band=[0.3, 50])  
# a 1-second non-overlapping preprocess window to extract de_lds features on specified extract bands  
data = feature_extraction(data, sample_rate, extract_bands=[[0.5,4],[4,8],[8,14],[14,30],[30,50]] , time_window=1, overlap=0, feature_type="de_lds") 
# sliding window with a size of 1 and  a step size of 1 to segment the samples.  
data, feature_dim = segment_data(data, sample_length=1, stride=1)
# data format: (session, subject, trail, sample)
```
#### Data Split
In LibEER, the Data Split module is mainly responsible for data partitioning under different experimental tasks and split settings. It supports three mainstream experimental tasks: subject-dependent, cross-subject, and cross-session, and offers various data splitting methods. The following example demonstrates how to split the dataset into training, validation, and testing sets in a subject-dependent task, with a ratio of 0.6, 0.2, and 0.2, respectively.
```python
from data_utils.split import merge_to_part
data, label = merge_to_part(data, label, experiment_mode="subject_dependent") 
# further split each subject's subtask  
for idx, (data_i, label_i) in enumerate(zip(data,label)):  
    # according to the data format and label,  the test size is 0.2 and the validation size is 0.2   
    spi = get_split_index(data_i, label_i,  split_type="train-val-test", test_size=0.2, val_size=0.2)  
    for jdx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(spi['train'],spi['test'], spi['val'])):  
        # organize the data according to the resulting index  
        (train_data, train_label, val_data, val_label,  test_data, test_label) = index_to_data(data_i, label_i,  train_indexes, test_indexes, val_indexes)
```

#### Model training and evaluation
LibEER supports various mainstream emotion recognition methods. For details, please refer to the Support Methods section. We selected DGCNN for training and testing.
```python
from models.Models import Model
from Trainer.training import train
model = Model['DGCNN'](num_electrodes=channels, feature_dim=5,  num_classes=3, k=2, layers=[64], dropout_rate=0.5)  
# train and evaluate model, then output the metric  
round_metric = train(model,train_data,train_label,val_data,val_label,test_data,test_label)
```
## Supported Dataset
- [seed](https://bcmi.sjtu.edu.cn/home/seed/seed.html)
- [seediv](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)
- [deap](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- [hci](https://mahnob-db.eu/hci-tagging/)
## Supported Methods
### DNN methods
- [Ms-mda](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.778488/full)
- [DBN](https://ieeexplore.ieee.org/document/6890166)
### CNN methods
- [EEGNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c)
- [CDCN](https://ieeexplore.ieee.org/document/9011570)
- [Tsception](https://ieeexplore.ieee.org/document/9762054)
### GNN methods
- [DGCNN](https://ieeexplore.ieee.org/document/8320798)
- [RGNN](https://ieeexplore.ieee.org/document/9091308)
- [GCB-Net](https://ieeexplore.ieee.org/document/8815811)
### RNN methods
- [ACRNN](https://ieeexplore.ieee.org/abstract/document/9204431)
### Transformer methods
- [HSLT](https://www.sciencedirect.com/science/article/abs/pii/S0893608024005483)

