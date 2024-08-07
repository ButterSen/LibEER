
class Setting:
    def __init__(self, dataset, dataset_path, pass_band, extract_bands, time_window, overlap, sample_length, stride, seed,
                 feature_type, only_seg=False, cross_trail='true', experiment_mode="subject-dependent", train_part=None, eog_clean=True,
                 metrics=None, normalize=False, save_data=True, split_type="kfold", fold_num=5, fold_shuffle=True, front=9, test_size=0.2, val_size = 0.2, sessions=None, pr=None, sr=None, bounds=None,
                 onehot=False, label_used=None):
        # random seed
        self.seed = seed

        # dataset setting

        self.dataset = dataset
        self.dataset_path = dataset_path

        # preprocess setting

        # Data at indices 0 and 1 represent the lower and higher thresholds of bandpass filtering
        self.pass_band = pass_band
        # Two-dimensional array, with each element at an index representing the range of each frequency band
        self.extract_bands = extract_bands if extract_bands is None else extract_bands
        # The size of the time window during preprocessing, in num of data points
        self.time_window = time_window
        # the length of overlap for each preprocessing window
        self.overlap = overlap
        # The length of sample sequences input to the model at once
        self.sample_length = sample_length
        # the stride of a sliding window for data extraction
        self.stride = stride
        # Feature type of EEG signals
        self.feature_type = feature_type
        # whether remove the eye movement interference
        self.eog_clean = eog_clean
        # whether normalize
        self.normalize = normalize
        # whether save_data
        self.save_data = save_data

        self.only_seg = only_seg

        # train_test_setting

        # whether use cross trial setting
        self.cross_trail = cross_trail
        # subject-dependent or subject-independent or cross-session
        self.experiment_mode = experiment_mode
        # how to partition a dataset
        self.split_type = split_type
        # according to the split type, choose which part is used as the training set or testing set
        self.fold_num = fold_num
        self.fold_shuffle = fold_shuffle
        self.front = front
        self.test_size = test_size
        self.val_size = val_size
        self.sessions = sessions
        self.pr = pr
        self.sr = sr

        self.bounds = bounds
        self.onehot = onehot
        self.label_used = label_used



def set_setting_by_args(args):
    if args.dataset_path is None:
        print("Please set the dataset path")
    if args.dataset is None:
        print("Please select the dataset to train")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, cross_trail=args.cross_trail, experiment_mode=args.experiment_mode,
                   metrics=args.metrics, normalize=args.normalize, split_type=args.split_type, fold_num=args.fold_num,
                   fold_shuffle=args.fold_shuffle, front=args.front, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   bounds=args.bounds, onehot=args.onehot, label_used=args.label_used)


def seed_sub_dependent_front_back_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject dependent experiment mode,\n"
          "the first 9 trails for each subject were used as a training set and the last 6 as a test set")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='front-back', front=9, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def seed_early_stopping_sub_dep_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Seed subject dependent early stopping experiment mode, \n"
          "For each subject, nine random trails were used as training set, three random trails were used as verification"
          " set, last three trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)
def seediv_early_stopping_sub_dep_setting(args):
    if not args.dataset.startswith('seediv'):
        print('not using SEED IV dataset, please check your setting')
        exit(1)
    print("Using SeedIV subject dependent early stopping experiment mode, \n"
          "For each subject, sixteen random trails were used as training set, four random trails were used as verification"
          " set, last four trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def seed_sub_dependent_5fold_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject dependent experiment mode,\n"
          "Using a 5-fold cross-validation, three test sets are grouped in the Order of trail")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, cross_trail=args.cross_trail, experiment_mode="subject-dependent",
                   normalize=args.normalize, split_type='kfold', fold_num=5, fold_shuffle=False, sessions=args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)


def seed_sub_independent_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject independent early stopping experiment mode,\n"
          "Using the leave one out method, all samples of 15 trails for 1 subject were split "
          "into all samples as a test set, and all samples of 15 trails for 14 other round "
          "were split into all samples as a training set, cycle 15 times to report average results")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='leave-one-out', sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def seed_early_stopping_sub_independent_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using early stopping SEED subject independent experiment mode,\n"
          "The random nine subjects' data are taken as training set, random three subjects' data are taken as "
          "validation set, random three subject's data are taken as test set. We choose the best results in validation set,"
          "and test it in test set"
          )
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def hci_early_stopping_sub_dependent_setting(args):
    if not args.dataset.startswith('hci'):
        print('not using Hci dataset, please check your setting')
        exit(1)
    print("Using hci subject dependent early stopping experiment mode, \n"
          ""
          )
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot, bounds=args.bounds,
                   label_used=args.label_used)

def seediv_early_stopping_sub_independent_setting(args):
    if not args.dataset.startswith('seediv'):
        print('not using SEED IV dataset, please check your setting')
        exit(1)
    print("Using SeedIV subject dependent early stopping experiment mode, \n"
          "For each subject, sixteen random trails were used as training set, four random trails were used as verification"
          " set, last four trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2,
                   sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def deap_early_stopping_sub_independent_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using deap dataset, please check your setting')
        exit(1)
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot, bounds=args.bounds,
                   label_used=args.label_used)
def hci_early_stopping_sub_independent_setting(args):
    if not args.dataset.startswith('hci'):
        print('not using Hci dataset, please check your setting')
        exit(1)
    print("Using hci subject dependent early stopping experiment mode, \n"
          ""
          )
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot, bounds=args.bounds,
                   label_used=args.label_used)
def deap_early_stopping_sub_dependent_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using deap dataset, please check your setting')
        exit(1)
    print("Using deap subject dependent early stopping experiment mode")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot,bounds=args.bounds,
                   label_used=args.label_used)



def seed_cross_session_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED cross session experiment mode,\n"
          "Three sessions of data, one as the test dataset")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="cross-session", normalize=args.normalize,
                   split_type='leave-one-out', sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def deap_sub_independent_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using DEAP dataset, please check your setting')
        exit(1)
    print("Using Default DEAP sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-independent",
                   normalize=args.normalize, split_type='leave-one-out', pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

def deap_sub_dependent_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using DEAP dataset, please check your setting')
        exit(1)
    print("Using Default DEAP sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-dependent",
                   normalize=args.normalize, cross_trail=args.cross_trail, split_type='kfold', fold_num=10, pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

def dreamer_sub_independent_setting(args):
    if not args.dataset.startswith('dreamer'):
        print('not using Dreamer dataset, please check your setting')
        exit(1)
    print("Using Default Dreamer sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 13], [14, 30]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-independent",
                   normalize=args.normalize, split_type='leave-one-out', pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

def dreamer_sub_dependent_setting(args):
    if not args.dataset.startswith('dreamer'):
        print('not using Dreamer dataset, please check your setting')
        exit(1)
    print("Using Default Dreamer sub dependent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 13], [14, 30]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-dependent",
                   normalize=args.normalize, cross_trail=args.cross_trail, split_type='leave-one-out', pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

preset_setting = {
    # 记得改一下格式
    "seed_early_stopping_sub_dep_setting": seed_early_stopping_sub_dep_setting,
    "seediv_early_stopping_sub_dep_setting": seediv_early_stopping_sub_dep_setting,
    "seed_early_stopping_sub_independent_setting": seed_early_stopping_sub_independent_setting,
    "seediv_early_stopping_sub_independent_setting": seediv_early_stopping_sub_independent_setting,
    "deap_early_stopping_sub_dependent_setting" : deap_early_stopping_sub_dependent_setting,
    "hci_early_stopping_sub_dependent_setting" : hci_early_stopping_sub_dependent_setting,
    "deap_early_stopping_sub_independent_setting" : deap_early_stopping_sub_independent_setting,
    "hci_early_stopping_sub_independent_setting" : hci_early_stopping_sub_independent_setting,
    # ***********************************************************************
    "seed_sub_dependent_5fold_setting": seed_sub_dependent_5fold_setting,
    "seed_sub_dependent_front_back_setting": seed_sub_dependent_front_back_setting,
    "seed_sub_independent_setting": seed_sub_independent_setting,
    "seed_cross_session_setting": seed_cross_session_setting,
    "deap_sub_independent_setting": deap_sub_independent_setting,
    "deap_sub_dependent_setting": deap_sub_dependent_setting,
    "dreamer_sub_independent_setting": dreamer_sub_independent_setting,
    "dreamer_sub_dependent_setting": dreamer_sub_dependent_setting,

    None: set_setting_by_args
}
