import collections
import os

import numpy as np
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

def transform(dpca_model, n_components, X):  # Accept num_trials, num_channels, num_times, Here self is the dpa
    # Get the size of the test data
    num_trials = X.shape[0]
    num_times = X.shape[-1]

    # Initialize the array to store transformed data
    X_Transformed = np.zeros((num_trials, n_components, num_times))
    for t in range(num_trials):
        temp = X[t, :, :][:, None, :]  # n_components, 1, num_times

        # Here, don't use the dPCA function transform
        # DON'T DO THIS: X_Transformed[t, :, :] = np.squeeze(self.dpca_model.transform(temp, 'c'))
        # Because there it make mean average, which centarize the temp every time
        # Makes curve height same across the class.
        # Instead, do this:
        D, Xr = dpca_model.D['s'], temp.reshape((temp.shape[0], -1))
        X_Transformed[t, :, :] = np.squeeze(np.dot(D.T, Xr).reshape((D.shape[1],) + temp.shape[1:]))

    return X_Transformed

if __name__ == "__main__":
    from scipy.stats import zscore
    from dPCA import dPCA
    from utils.utils import *
    from scipy.ndimage import gaussian_filter1d
    #from statsmodels.stats.multitest import multipletests
    #pio.renderers.default = "svg"  # avoid opening browser

    data_folder = 'data'
    patient = 'CTH'
    what_to_plot = 'SubjectNum'# this could be "SubjectNum", "SubjectPerson" "Morphology", "Tense"

    n_bins_history = 5
    n_components = 3
    #extractor = NeighborhoodComponentsAnalysis(n_component s=n_PC)
    extractor_name=PCA
    all_patients = os.listdir(data_folder)
    patient_folder_index = np.where([patient in s for s in all_patients])[0][0]
    # get the path
    path = os.path.join(data_folder, all_patients[patient_folder_index], 'sentenceCompletion2_all_data.mat')
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

    # 3. go cue (this is mostly ignored by the patient anyway)
    back_go_cue = np.nanmin(clean_go_cue)
    forward_go_cue = np.nanmin(clean_trial_offset - clean_go_cue)
    data_binned_go_cue = align_data(clean_data_binned, clean_go_cue, back_go_cue, forward_go_cue)

   ############ Here we are plotting dPCA differences for the conditions

    ################# phrase onset #####################################
    if what_to_plot == 'SubjectNum':
        labels_to_use = sentence_subject_number_label
    elif what_to_plot == 'SubjectPerson':
        labels_to_use=sentence_subject_person_label
    elif what_to_plot == 'Morphology':
        labels_to_use = clean_morpho_label
    elif what_to_plot == 'Tense':
        labels_to_use = sentence_tense_label
    else:
        raise ValueError("what_to_decode should be 'SubjectNum', 'Morphology', or 'Tense'")
    data_to_use = gaussian_filter1d(data_binned_phrase_onset.swapaxes(1, 2)[
        labels_to_use != 'NA' ],2,axis=1)
    labels_to_use = np.array(labels_to_use)[labels_to_use != 'NA']
    labels=np.unique(labels_to_use) # assume there are only two types of labels
    condition1 = np.array(data_to_use)[labels_to_use == labels[0]].swapaxes(1, 2)
    condition2 = np.array(data_to_use)[labels_to_use == labels[1]].swapaxes(1, 2)

    min_length = np.min([condition1.shape[0], condition2.shape[0]])
    condition1 = condition1[-min_length:]
    condition2 = condition2[-min_length:]
    # condition1 = (condition1 - np.mean(condition1.reshape(min_length, -1), axis=1)[:, None, None]) / np.std(condition1.reshape(min_length, -1), axis=1)[:, None, None]
    # condition2 = (condition2 - np.mean(condition2.reshape(min_length, -1), axis=1)[:, None, None]) / np.std(condition2.reshape(min_length, -1), axis=1)[:, None, None]
    condition1 = condition1 - np.mean(condition1, axis=2, keepdims=True)
    condition2 = condition2 - np.mean(condition2, axis=2, keepdims=True)
    # condition1 = zscore(condition1,axis=2)
    # condition2 = zscore(condition2,axis=2)
    # zscore the data before putting it in dPCA

    new_labels = np.array([labels[0]] * min_length + [labels[1]] * min_length)
    # the following is from the original ipynb file of dPCA
    trialR = np.stack([condition1, condition2], axis=-1).swapaxes(2, 3)  # n_samples, n_neurons, n_stimulus,n_timebins
    R = np.mean(trialR, 0)
    R -= np.mean(R.reshape(data_to_use.shape[-1], -1), 1)[:, None, None]
    dpca = dPCA.dPCA(labels='st', n_components=n_components,regularizer='auto')
    dpca.protect = ['t']
    Z = dpca.fit_transform(R, trialR)
    Z_single_trial = transform(dpca, n_components, np.concatenate([condition1, condition2], axis=0))


    fig, ax = plt.subplots(1,n_components, figsize=(4 * n_components,4))
    time = np.linspace(-back_phrase_onset,forward_phrase_onset, data_to_use.shape[1])
    for i in range(n_components):
        for s in range(2):
            ax[i].fill_between(time,
                                  Z_single_trial[:min_length, i, :].mean(0) - 1.96 * Z_single_trial[:min_length, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]),
                                  Z_single_trial[:min_length, i, :].mean(0) + 1.96 * Z_single_trial[:min_length, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]), color='b', alpha=0.1)
            ax[i].fill_between(time,
                                  Z_single_trial[-min_length:, i, :].mean(0) - 1.96 * Z_single_trial[-min_length:, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]),
                                  Z_single_trial[-min_length:, i, :].mean(0) + 1.96 * Z_single_trial[-min_length:, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]), color='g', alpha=0.1)
            ax[i].plot(time, Z_single_trial[:min_length, i, :].mean(0), color='b')
            ax[i].plot(time, Z_single_trial[-min_length:, i, :].mean(0), color='g')
            ax[i].axvline(0, color='blue', linestyle='--')
            ax[i].axvline(1.5, color='yellow', linestyle='--')
            ax[i].axvline(np.nanmean(clean_voice_onset), color='r', linestyle='--')
            ax[i].axvline(np.nanmean(clean_voice_offset), color='cyan', linestyle='--')
            ax[i].set_xlabel("time (s)")
    plt.title('Aligned to Phrase/Word onset')
    fig.savefig(f'figures/{what_to_plot}/dPCA_phrase_onset_{patient}.pdf')
    plt.tight_layout()

    fig, ax = plot_on_channel([gaussian_filter1d(data_to_use, 2, axis=1)[labels_to_use==np.unique(labels_to_use)[0]].mean(axis=0).swapaxes(0,1),
                              gaussian_filter1d(data_to_use, 2, axis=1)[labels_to_use == np.unique(labels_to_use)[1]].mean(axis=0).swapaxes(0,1)],
                              CI=[gaussian_filter1d(data_to_use, 2, axis=1)[
                                      labels_to_use == np.unique(labels_to_use)[0]].std(axis=0).swapaxes(0,1) / np.sqrt(
                                  sum(labels_to_use == np.unique(labels_to_use)[0])) * 1.96,
                                  gaussian_filter1d(data_to_use, 2, axis=1)[
                                      labels_to_use == np.unique(labels_to_use)[1]].std(axis=0).swapaxes(0,1) / np.sqrt(
                                      sum(labels_to_use == np.unique(labels_to_use)[1])) * 1.96
                                  ],
                              lines=[0,
                                     1.5,
                                     np.nanmean(clean_voice_onset),
                                     np.nanmean(clean_voice_offset)],
                              line_labels=['phrase onset', 'word onset', 'voice onset', 'voice offset'],
                              back=back_phrase_onset, forward=forward_phrase_onset,
                              title=f"aligned to Phrase/word onset: {bin_size} ms binned")
    fig.savefig(f'figures/{what_to_plot}/trial_averages_phrase_onset_{patient}.pdf')

############################## voice onset #######################################
    if what_to_plot == 'SubjectNum':
        labels_to_use = sentence_subject_number_label
    elif what_to_plot == 'SubjectPerson':
        labels_to_use=sentence_subject_person_label
    elif what_to_plot == 'Morphology':
        labels_to_use = clean_morpho_label
    elif what_to_plot == 'Tense':
        labels_to_use = sentence_tense_label
    else:
        raise ValueError("what_to_decode should be 'SubjectNum', 'Morphology', or 'Tense'")
    data_to_use = gaussian_filter1d(data_binned_voice_onset.swapaxes(1, 2)[
        labels_to_use != 'NA' ],2,axis=1)
    labels_to_use = np.array(labels_to_use)[labels_to_use != 'NA']

    labels=np.unique(labels_to_use) # assume there are only two types of labels

    condition1 = np.array(data_to_use)[labels_to_use == labels[0]].swapaxes(1, 2)
    condition2 = np.array(data_to_use)[labels_to_use == labels[1]].swapaxes(1, 2)

    min_length = np.min([condition1.shape[0], condition2.shape[0]])
    condition1 = condition1[-min_length:]
    condition2 = condition2[-min_length:]
    # condition1 = (condition1-np.mean(condition1.reshape(min_length,-1),axis=1)[:,None,None])/np.std(condition1.reshape(min_length,-1),axis=1)[:,None,None]
    # condition2 = (condition2-np.mean(condition2.reshape(min_length,-1),axis=1)[:,None,None])/np.std(condition2.reshape(min_length,-1),axis=1)[:,None,None]
    condition1 = condition1 - np.mean(condition1, axis=2, keepdims=True)
    condition2 = condition2 - np.mean(condition2, axis=2, keepdims=True)
    # zscore the data before putting it in dPCA

    new_labels = np.array([labels[0]] * min_length + [labels[1]] * min_length)
    # the following is from the original ipynb file of dPCA
    trialR = np.stack([condition1, condition2], axis=-1).swapaxes(2, 3)  # n_samples, n_neurons, n_stimulus,n_timebins
    R = np.mean(trialR, 0)
    R -= np.mean(R.reshape(data_to_use.shape[-1], -1), 1)[:, None, None]
    dpca = dPCA.dPCA(labels='st',  n_components=n_components,regularizer='auto',)
    dpca.protect = ['t']
    Z = dpca.fit_transform(R, trialR)
    Z_single_trial = transform(dpca, n_components, np.concatenate([condition1, condition2], axis=0))

    fig, ax = plt.subplots(1,n_components, figsize=(4 * n_components,4))
    time = np.linspace(-back_voice,forward_voice, data_to_use.shape[1])
    for i in range(n_components):
        for s in range(2):
            ax[i].fill_between(time,
                                  Z_single_trial[:min_length, i, :].mean(0) - 1.96 * Z_single_trial[:min_length, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]),
                                  Z_single_trial[:min_length, i, :].mean(0) + 1.96 * Z_single_trial[:min_length, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]), color='b', alpha=0.1)
            ax[i].fill_between(time,
                                  Z_single_trial[-min_length:, i, :].mean(0) - 1.96 * Z_single_trial[-min_length:, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]),
                                  Z_single_trial[-min_length:, i, :].mean(0) + 1.96 * Z_single_trial[-min_length:, i, :].std(
                                      0) / np.sqrt(Z_single_trial.shape[0]), color='g', alpha=0.1)
            ax[i].plot(time, Z_single_trial[:min_length, i, :].mean(0), color='b')
            ax[i].plot(time, Z_single_trial[-min_length:, i, :].mean(0), color='g')
            ax[i].axvline(-np.nanmean(clean_voice_onset), color='blue', linestyle='--')
            ax[i].axvline(-np.nanmean(clean_voice_onset)+1.5, color='yellow', linestyle='--')
            ax[i].axvline(0, color='red', linestyle='--')
            ax[i].axvline(np.nanmean(clean_voice_offset)-np.nanmean(clean_voice_onset), color='cyan', linestyle='--')
            ax[i].set_xlabel("time (s)")
    plt.title('Aligned to Voice onset')
    plt.tight_layout()
    fig.savefig(f'figures/{what_to_plot}/dPCA_voice_onset_{patient}.pdf')


    fig, ax = plot_on_channel([gaussian_filter1d(data_to_use, 2, axis=1)[labels_to_use==np.unique(labels_to_use)[0]].mean(axis=0).swapaxes(0,1),
                               gaussian_filter1d(data_to_use, 2, axis=1)[labels_to_use==np.unique(labels_to_use)[1]].mean(axis=0).swapaxes(0,1)
                               ],
                              CI=[gaussian_filter1d(data_to_use, 2, axis=1)[labels_to_use==np.unique(labels_to_use)[0]].std(axis=0).swapaxes(0,1)/np.sqrt(sum(labels_to_use==np.unique(labels_to_use)[0]))*1.96,
                               gaussian_filter1d(data_to_use, 2, axis=1)[labels_to_use==np.unique(labels_to_use)[1]].std(axis=0).swapaxes(0,1)/np.sqrt(sum(labels_to_use==np.unique(labels_to_use)[1]))*1.96
                               ],
                              lines=[-np.nanmean(clean_voice_onset),
                                     -np.nanmean(clean_voice_onset) + 1.5,
                                     0,
                                     -np.nanmean(clean_voice_onset - clean_voice_offset)],
                              line_labels=['phrase onset', 'word onset', 'voice onset', 'voice offset'],
                              back=back_voice, forward=forward_voice,
                              title=f"aligned to voice onset: {bin_size} ms binned")
    fig.savefig(f'figures/{what_to_plot}/trial_averages_voice_onset_{patient}.pdf')






    plt.show()


