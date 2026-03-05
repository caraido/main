import collections
import os

from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sympy.polys.numberfields.utilities import is_rat
from tensorly.decomposition import parafac
from tqdm import tqdm

from utils.utils import *


plt.rcParams['pdf.fonttype'] = 42

def align_data(data, cue, back, forward):
    data_aligned = []
    for i in range(data.shape[0]):
        trial = data[i]
        truncated = trial[:, int((cue[i] - back) * adjusted_fs):int((cue[i] + forward) * adjusted_fs)]
        data_aligned.append(truncated)
    trial_min_length = np.min([i.shape[1] for i in data_aligned])
    data_aligned = np.array([i[:, :trial_min_length] for i in data_aligned])
    return data_aligned

def objective(trial):
    svc_c = trial.suggest_float("svc_c", 1e-6, 1e10, log=True)
    svc_gamma = trial.suggest_float("svc_gamma", 1e-10, 1e6, log=True)

    classifier_obj = SVC(kernel='rbf',C=svc_c, gamma=svc_gamma, class_weight='balanced')
    accuracy=0.0
    for i in range(len(X_to_use)):
        X_train_to_tune, _, y_train_to_tune, _ = train_test_split(X_to_use[i], labels_to_use,test_size=0.001)
        X_train_to_tune=StandardScaler().fit_transform(X_train_to_tune)
        X_train_to_tune=extractor.fit_transform(X_train_to_tune,y_train_to_tune)
        score = cross_val_score(classifier_obj, X_train_to_tune, y_train_to_tune, n_jobs=-1, cv=5)
        accuracy=np.maximum(score.mean(),accuracy)
    return accuracy

class GeneralDecoder:
    def __init__(self,extractor, decoder):
        self.extractor=extractor # Extractor is used to extract feature. It can be an PCA, PLS, NCA, KNN or StandardScaler object
        self.decoder=decoder # Decoder is the classifier used to decode labels. Also an initialized object
        self.X_to_use=None # This should be a list
        self.y=None
        self.scaler=StandardScaler()

        self.train_accuracy=None
        self.test_accuracy=None
        self.chance=None
        self.mean_train_accuracy=None
        self.mean_test_accuracy=None
        self.mean_chance=None
        self.std_train_accuracy=None
        self.std_test_accuracy=None
        self.std_chance=None

    def decode(self,test_size=0.3,n_repeats=50,n_time_bin=None):
        if n_time_bin is None:
            print("We will build a decoder for each time bin")
            all_test_accuracy=[]
            all_train_accuracy=[]
            all_chance=[]
            for repeat in range(n_repeats):
                print(f"Start repeat {repeat+1}")
                for n_bin in range(len(self.X_to_use)):
                    # split the data
                    X_train, X_test, y_train, y_test = train_test_split(self.X_to_use[n_bin], self.y, test_size=test_size)

                    # scale the features

                    X_train=self.scaler.fit_transform(X_train)
                    X_test=self.scaler.transform(X_test)

                    # extract the features
                    self.extractor.fit(X_train,y_train)
                    X_train_low = self.extractor.transform(X_train)
                    X_test_low = self.extractor.transform(X_test)
                    # classification
                    self.decoder.fit(X_train_low, y_train)
                    y_test_predict = self.decoder.predict(X_test_low)
                    y_train_predict = self.decoder.predict(X_train_low)

                    # get the shuffled data
                    X_train_shuffle = np.random.permutation(X_train_low.flatten()).reshape(X_train_low.shape)
                    decoder.fit(X_train_shuffle, y_train)
                    y_shuffle = decoder.predict(X_test_low)

                    # calculate the accuracy
                    correct_train = np.sum(y_train_predict == y_train)
                    accuracy_train = correct_train / len(y_train_predict)
                    correct_test = np.sum(y_test_predict == y_test)
                    accuracy_test = correct_test / len(y_test_predict)

                    # calculate the chance
                    correct_shuffle = np.sum(y_shuffle == y_test)
                    chance = correct_shuffle / len(y_shuffle)

                    all_train_accuracy.append(accuracy_train)
                    all_test_accuracy.append(accuracy_test)
                    all_chance.append(chance)
            self.test_accuracy=np.array(all_test_accuracy).reshape(n_repeats,-1)
            self.train_accuracy=np.array(all_train_accuracy).reshape(n_repeats,-1)
            self.chance=np.array(all_chance).reshape(n_repeats,-1)

            self.mean_test_accuracy=np.mean(self.test_accuracy,0)
            self.mean_train_accuracy=np.mean(self.train_accuracy,0)
            self.mean_chance=np.mean(self.chance,0)

            self.std_test_accuracy=np.std(self.test_accuracy,0)
            self.std_train_accuracy=np.std(self.train_accuracy,0)
            self.std_chance=np.std(self.chance,0)
        else:
            assert isinstance(n_time_bin,int) & n_time_bin==0 & n_time_bin<=len(self.X_to_use), "n_time_bin should be an integer between 1 and the number of time bins"
            all_test_accuracy=[]
            all_train_accuracy=[]
            all_chance=[]
            for repeat in range(n_repeats):
                print(f"Start repeat {repeat+1}")
                # split the data
                X_train, X_test, y_train, y_test = train_test_split(self.X_to_use[n_time_bin], self.y, test_size=test_size)

                # scale the features
                X_train=self.scaler.fit_transform(X_train)
                X_test=self.scaler.transform(X_test)

                # extract the features
                self.extractor.fit(X_train,y_train)
                X_train_low = self.extractor.transform(X_train)
                X_test_low = self.extractor.transform(X_test)
                # classification
                self.decoder.fit(X_train_low, y_train)
                y_test_predict = self.decoder.predict(X_test_low)
                y_train_predict = self.decoder.predict(X_train_low)

                # get the shuffled data
                X_train_shuffle = np.random.permutation(X_train_low.flatten()).reshape(X_train_low.shape)
                decoder.fit(X_train_shuffle, y_train)
                y_shuffle = decoder.predict(X_test_low)

                # calculate the accuracy
                correct_train = np.sum(y_train_predict == y_train)
                accuracy_train = correct_train / len(y_train_predict)
                correct_test = np.sum(y_test_predict == y_test)
                accuracy_test = correct_test / len(y_test_predict)

                # calculate the chance
                correct_shuffle = np.sum(y_shuffle == y_test)
                chance = correct_shuffle / len(y_shuffle)

                all_train_accuracy.append(accuracy_train)
                all_test_accuracy.append(accuracy_test)
                all_chance.append(chance)
            self.test_accuracy = np.array(all_test_accuracy)
            self.train_accuracy = np.array(all_train_accuracy)
            self.chance = np.array(all_chance)

            self.mean_test_accuracy = np.mean(all_test_accuracy, 0)
            self.mean_train_accuracy = np.mean(all_train_accuracy, 0)
            self.mean_chance = np.mean(all_chance, 0)

            self.std_test_accuracy = np.std(all_test_accuracy, 0)
            self.std_train_accuracy = np.std(all_train_accuracy, 0)
            self.std_chance = np.std(all_chance, 0)

if __name__ == "__main__":
    import optuna
    from plotly.io import show,write_image,write_html
    import plotly.io as pio
    from utils.utils import *
    #from statsmodels.stats.multitest import multipletests
    #pio.renderers.default = "svg"  # avoid opening browser
    ### This is only for decoding tense
    data_folder = 'data'
    patient = 'MBF'

    n_bins_history = 5
    n_PC = 20
    extractor = NeighborhoodComponentsAnalysis(n_components=n_PC)
    all_patients = os.listdir(data_folder)
    patient_folder_index = np.where([patient in s for s in all_patients])[0][0]
    # get the path
    path = os.path.join(data_folder, all_patients[patient_folder_index], 'sentenceCompletion_all_data.mat')
    print(path)

    # load the data
    all_data=load_all_data(path)

    bin_size=100 # ms
    fs=2000

    data=all_data[:,0]
    trial_onset=all_data[:,1] # in sec. # here the phrase onset is the same as the trial onset
    go_cue=all_data[:,2] # in sec
    trial_offset=all_data[:,3] # in sec
    voice_onset=all_data[:,4] # in sec
    voice_offset=all_data[:,5] # in sec

    # get the labels
    target_label=all_data[:,6]
    answer_label=all_data[:,7]
    sentence_label=all_data[:,12]
    irregular_label=all_data[:,14]

    # for CTH and MBF
    bad_channels=np.sort(np.squeeze(all_data[0,8].astype(int))-1) # minus 1 to conform to the python index
    bad_trials=np.sort(np.squeeze(all_data[0,9].astype(int))-1) # minus 1 to conform to the python index

    # here we selected the first 64 channels: IFG grid
    data=np.array([np.array([d for d in dt]) for dt in data]).swapaxes(1,2)[:,:64,:] # n_trial x n_channels x n_time
    bad_channels=bad_channels[bad_channels<64]

    # truncate the data a little bit for reshaping
    min_length=data.shape[-1]//bin_size*bin_size
    data=data[:,:,:min_length]
    data_binned=data.reshape(data.shape[0],data.shape[1],-1,bin_size).mean(axis=3)

    n_trials=len(all_data)
    n_channels=data.shape[1]
    n_time=data.shape[2]
    n_bins=data_binned.shape[2]
    adjusted_fs=int(fs/bin_size)

    trial_onset=fix_index(trial_onset) # here the phrase onset is the same as the trial onset
    trial_offset=fix_index(trial_offset)
    go_cue=fix_index(go_cue)
    voice_onset=fix_index(voice_onset)
    voice_offset=fix_index(voice_offset)
    irregular_label=fix_index(irregular_label)
    word_onset=trial_onset+1.5 # word onset has no gitter after trial(phrase) onset


    # for v7.3 files. This should be the standard in the future
    target_label=np.array([tl[0].lower() for tl in target_label],dtype=object)
    answer_label=np.array([al[0] if len(al)>0 else '' for al in answer_label],dtype=object)
    sentence_label=np.array([a for a in sentence_label],dtype=object)

    # get the morphological labels for decoding
    morpho_label=np.array(['same' if t==a else 'not same' for t,a in zip(target_label,answer_label)],dtype=object)
    full_sentence_label=[str(add_space_after_comma(replace_underscores(replace_underscores(a, b),' '))) for a, b in zip(sentence_label, target_label)]

    # here we remove the bad channels and bad trials
    # for the data
    clean_data_binned = np.delete(np.delete(data_binned, bad_channels, axis=1), bad_trials, axis=0)

    # for the labels
    clean_voice_onset = np.delete(voice_onset, bad_trials)
    clean_voice_offset = np.delete(voice_offset, bad_trials)
    clean_go_cue = np.delete(go_cue, bad_trials)
    clean_target_label = np.delete(target_label, bad_trials)
    clean_answer_label = np.delete(answer_label, bad_trials)
    clean_trial_offset = np.delete(trial_offset, bad_trials)
    clean_morpho_label = np.delete(morpho_label, bad_trials)
    clean_full_sentence_label = np.delete(full_sentence_label, bad_trials)
    clean_irregular_label = np.delete(irregular_label, bad_trials)

    nan_trials = np.where(~np.isnan(clean_voice_onset))[0]
    clean_data_binned = clean_data_binned[nan_trials]
    clean_voice_onset = clean_voice_onset[nan_trials]
    clean_go_cue = clean_go_cue[nan_trials]
    clean_voice_offset = clean_voice_offset[nan_trials]
    clean_target_label = clean_target_label[nan_trials]
    clean_answer_label = clean_answer_label[nan_trials]
    clean_trial_offset = clean_trial_offset[nan_trials]
    clean_morpho_label = clean_morpho_label[nan_trials]
    clean_full_sentence_label = clean_full_sentence_label[nan_trials]
    clean_irregular_label = clean_irregular_label[nan_trials]

    n_clean_channels = clean_data_binned.shape[1]
    n_clean_trials = clean_data_binned.shape[0]
    print(collections.Counter(clean_morpho_label))

    # get sentence tense (not based on word conjugation)
    sentence_tense_label = []
    for i, s in enumerate(clean_full_sentence_label):
        tense = get_sentence_tense(s)
        sentence_tense_label.append(tense)
    sentence_tense_label = np.array(sentence_tense_label)
    print(collections.Counter(sentence_tense_label))

    # sentence subject number (not based on word conjugation)
    sentence_subject_number_label= np.array([get_sentence_subject_number(str(s)) for s in clean_full_sentence_label])
    print(collections.Counter(sentence_subject_number_label))

    # sentence subject person (not based on word conjugation)
    sentence_subject_person_label= np.array([get_sentence_subject_person(str(s)) for s in clean_full_sentence_label])
    print(collections.Counter(sentence_subject_person_label))

    # align the data to different cues
    # 1. phrase/ word onset. word is just 1.5 s after phrase onset
    back_phrase_onset = 0
    forward_phrase_onset = 6
    data_binned_phrase_onset = clean_data_binned[:, :, :int(forward_phrase_onset * adjusted_fs)] # 4s of data

    # 2. voice onset
    back_voice=np.nanmin(clean_voice_onset) # positive
    forward_voice=np.nanmin(clean_trial_offset-clean_voice_onset) # positive
    data_binned_voice_onset=align_data(clean_data_binned,clean_voice_onset,back_voice,forward_voice)

    # 2. voice offset
    back_voice_off = np.nanmin(clean_voice_offset)  # positive
    forward_voice_off = 1  # positive
    data_binned_voice_offset = align_data(clean_data_binned, clean_voice_offset, back_voice_off, forward_voice_off)

    # 3. go cue (this is mostly ignored by the patient anyway)
    back_go_cue = np.nanmin(clean_go_cue)
    forward_go_cue = np.nanmin(clean_trial_offset - clean_go_cue)
    data_binned_go_cue = align_data(clean_data_binned, clean_go_cue, back_go_cue, forward_go_cue)

    #######################################################################

    ################### analysis ########################
    ## parameters to use

    alpha = 0.05
    #
    # #decoder = SVC(kernel='rbf', class_weight='balanced')
    # ############### Decoding based on phrase onset ####################
    # # step 1: get the data for training
    #
    #
    # labels_to_use = sentence_tense_label
    # labels_to_use=labels_to_use[clean_irregular_label==1] # 1: irregular verbs, 0: regular verbs
    # data_to_use = np.array(data_binned_phrase_onset).swapaxes(1, 2)[clean_irregular_label==1][labels_to_use!='NA']
    # labels_to_use = labels_to_use[labels_to_use != 'NA']
    #
    #
    # n_bins = data_to_use.shape[1]
    #
    # X_to_use = reformat(data_to_use, n_bins_history)
    #
    # ###### here we try to do some quick hyperparameter tuning
    # # study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler())
    # # study.optimize(objective, n_trials=200,show_progress_bar=True) # be careful of this since the parameters are used from the variable space.
    # # print(study.best_params)
    # #
    # # fig = optuna.visualization.plot_optimization_history(study)
    # # write_html(fig, f'figures/{what_to_decode}/optuna_history_{patient}_phrase_onset_{extractor}.html')
    # # fig = optuna.visualization.plot_parallel_coordinate(study, params=["svc_gamma", "svc_c"])
    # # write_html(fig, f'figures/{what_to_decode}/optuna_parallel_coordinate_{patient}_phrase_onset_{extractor}.html')
    # # fig = optuna.visualization.plot_param_importances(study)
    # # write_html(fig, f'figures/{what_to_decode}/optuna_param_importances_{patient}_phrase_onset_{extractor}.html')
    # # fig = optuna.visualization.plot_contour(study, params=["svc_gamma", "svc_c"])
    # # write_html(fig, f'figures/{what_to_decode}/optuna_contour_{patient}_phrase_onset_{extractor}.html')
    #
    # ##### Now do decoding ######
    # decoder = SVC(kernel='rbf', class_weight='balanced')#, C=study.best_params['svc_c'],gamma=study.best_params['svc_gamma'])
    #
    # pipeline=GeneralDecoder(extractor,decoder)
    # pipeline.X_to_use=X_to_use
    # pipeline.y=labels_to_use
    # pipeline.decode()
    #
    # # compare to test accuracy distribution
    # percentile_bybin = np.percentile(pipeline.test_accuracy, alpha * 100, axis=0)
    # p_illustration = ['*' if percentile_bybin[i] > pipeline.mean_chance[i] else '' for i in
    #                   range(pipeline.chance.shape[-1])]
    #
    # # plot aligned to word onset
    # fig,ax=plot_accuarcy(pipeline.mean_test_accuracy, pipeline.mean_chance,
    #               lines=[1.5, np.nanmean(word_onset), np.nanmean(clean_voice_onset), np.nanmean(clean_voice_offset)],
    #               line_labels=['word onset', 'go cue', 'voice on', 'voice off'],
    #               back=0, forward=data_binned_phrase_onset.shape[-1] / adjusted_fs,
    #               data_labels=["test accuracy", "chance"],
    #               data_std=[pipeline.std_test_accuracy, 0],
    #               p=p_illustration, n_xticks=10,
    #               title=f"aligned to Phrase/word onset")
    # fig.savefig(f"figures/Irregular Tense/accuracy_aligned_to_{patient}_phrase_onset_{extractor}_{decoder}_{bin_size}ms_irregular.pdf")

    ############### Decoding based on voice onset ####################
    # step 1: find the best parameter

    labels_to_use = sentence_tense_label
    labels_to_use=labels_to_use[clean_irregular_label==0] # 1: irregular verbs, 0: regular verbs
    data_to_use = np.array(data_binned_voice_offset).swapaxes(1, 2)[clean_irregular_label==0][labels_to_use!='NA']
    labels_to_use = labels_to_use[labels_to_use != 'NA']


    X_to_use = reformat(data_to_use, n_bins_history)
    ###### here we try to do some quick hyperparameter tuning
    # study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler())
    # study.optimize(objective, n_trials=200,show_progress_bar=True) # be careful of this since the parameters are used from the variable space.
    # print(study.best_params)
    #
    # fig = optuna.visualization.plot_optimization_history(study)
    # write_html(fig, f'figures/{what_to_decode}/optuna_history_{patient}_voice_onset_{extractor}.html')
    # fig = optuna.visualization.plot_parallel_coordinate(study, params=["svc_gamma", "svc_c"])
    # write_html(fig, f'figures/{what_to_decode}/optuna_parallel_coordinate_{patient}_voice_onset_{extractor}.html')
    # fig = optuna.visualization.plot_param_importances(study)
    # write_html(fig, f'figures/{what_to_decode}/optuna_param_importances_{patient}_voice_onset_{extractor}.html')
    # fig = optuna.visualization.plot_contour(study, params=["svc_gamma", "svc_c"])
    # write_html(fig, f'figures/{what_to_decode}/optuna_contour_{patient}_voice_onset_{extractor}.html')


    ##### Now do decoding ######
    decoder=SVC(kernel='rbf',class_weight='balanced')#,C=study.best_params['svc_c'],gamma=study.best_params['svc_gamma'])

    pipeline=GeneralDecoder(extractor,decoder)
    pipeline.X_to_use=X_to_use
    pipeline.y=labels_to_use
    pipeline.decode()

    # compare to test accuracy distribution
    percentile_bybin = np.percentile(pipeline.test_accuracy, alpha * 100, axis=0)
    p_illustration = ['*' if percentile_bybin[i] > pipeline.mean_chance[i] else '' for i in
                      range(pipeline.chance.shape[-1])]

    # plot aligned to word onset
    fig,ax=plot_accuarcy(pipeline.mean_test_accuracy, pipeline.mean_chance,
                  lines=[-np.nanmean(clean_voice_offset),
                         -np.nanmean(clean_voice_offset)+1.5,

                         np.nanmean(clean_voice_onset - clean_voice_offset),
                         0],
                  line_labels=['phrase onset', 'word onset', 'voice on', 'voice off'],
                  back=back_voice_off, forward=forward_voice_off,
                  data_labels=["test accuracy", "chance"],
                  data_std=[pipeline.std_test_accuracy, 0],
                  p=p_illustration, n_xticks=10,
                  title=f"aligned to Voice offset")
    fig.savefig(f"figures/Irregular Tense/accuracy_aligned_to_{patient}_voice_offset_{extractor}_{decoder}_{bin_size}ms_regular.pdf")










