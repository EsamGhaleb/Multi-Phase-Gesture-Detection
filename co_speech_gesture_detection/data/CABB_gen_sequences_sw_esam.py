markersbody = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
          'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
          'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
          'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
          'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

markershands = ['LEFT_WRIST', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
              'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP', 
               'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP', 
               'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_FINGER_MCP', 
               'LEFT_PINKY_FINGER_PIP', 'LEFT_PINKY_FINGER_DIP', 'LEFT_PINKY_FINGER_TIP',
              'RIGHT_WRIST', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
              'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP', 
               'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP', 
               'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_FINGER_MCP', 
               'RIGHT_PINKY_FINGER_PIP', 'RIGHT_PINKY_FINGER_DIP', 'RIGHT_PINKY_FINGER_TIP']
# from the markersbody, get me the indices of, nose, left eye, right eye, left shoulder, right shoulder, left hip, right hip, left elbow, right elbow
selected_markers = [0, 2, 5, 11, 12, 13, 14]# write the indices of markerhands plus 33 (the number of markersbody)
selected_markers.extend([33 + i for i in range(len(markershands))])

print(selected_markers)

import pickle
import sys
from collections import Counter
import numpy as np
import os
from get_gesture_info import (
    get_gestures_info, prepare_dialogue_shared_expressions_and_turns_info
)
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
from itertools import product
from einops import rearrange

# number of frames in a video to be considered
max_rgb_frame = 18 # 67% OF THE ICONIC GESTURES HAVE THIS LENGTH, which is the mean duration of the iconic gestures. We can keep this as it is since we are using the sliding windows approach, to determin the boundaris of the iconic gestures
sample_rate = 2
iou = 100
max_body_true = 1
max_frame = max_rgb_frame
num_channels = 3
keypoints_sampling_rate = 2 # this is the sampling rate of the keypoints, chose every 4th frame
max_seq_len = 40 # this is the upper bound of the number of frames in a video, given the confidence interval of 95% and the number of frames in the videos

def return_cross_validation_ids(labels, subject_ids, num_folds=2, complete_subject_based=False, kfold_cross_validation=True):
    subject_ids = np.array(subject_ids)
    labels = np.array(labels)
    unique_subjects = np.unique(subject_ids)
    if complete_subject_based:
        return subject_ids, subject_ids

    # I used Kfold function from Sklearn since it give equal number of samples for each fold
    if kfold_cross_validation:
        cv = KFold(num_folds, shuffle=True, random_state=1000)
        indices = np.zeros(subject_ids.shape)
        for ii, (tr, tt) in enumerate(cv.split(X=subject_ids, y=labels)):
            indices[tt] = ii
        indices = np.int32(indices)
        return indices, subject_ids

    cv = KFold(num_folds, shuffle=True, random_state=1000)
    indices = np.zeros(unique_subjects.shape)
    for ii, (tr, tt) in enumerate(cv.split(X=unique_subjects, y=unique_subjects)):
        indices[tt] = ii

    subject_based_CV_ids = np.zeros(subject_ids.shape)
    for i, unique_id in enumerate(unique_subjects):
        subject_based_CV_ids[np.where(subject_ids == unique_id)] = indices[i]
    subject_based_CV_ids = np.int32(subject_based_CV_ids)
    return subject_based_CV_ids, subject_ids

sys.path.extend(['../'])

selected_joints = {
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
   'CABB': selected_markers
}


def calculate_overlap(start1, end1, start2, end2):
    # Find the earliest and latest start and end times
    earliest_start = min(start1, start2)
    latest_end = max(end1, end2)
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)

    # Calculate the overlap, if any
    overlap = 0
    if latest_start < earliest_end:
        overlap = (earliest_end - latest_start)

    # Calculate the total duration
    total_duration = (end1-start1)

    # Calculate the percentage overlap
    if total_duration > 0:
        percentage_overlap = (overlap / total_duration) # * 100
    else:
        percentage_overlap = 0
    return percentage_overlap

      
def overlap_percentage(window, annotation):
    # Calculate the overlap as the intersection of the two ranges
    overlap = max(0, min(window[1], annotation[1]) - max(window[0], annotation[0]) + 1)

    # Calculate the length of the window and annotation
    window_length = window[1] - window[0] + 1
    annotation_length = annotation[1] - annotation[0] + 1

    # Check if the window is entirely within the annotation or vice versa
    if window[0] >= annotation[0] and window[1] <= annotation[1]:  # window within annotation
        percentage = overlap / window_length
        status = 'full'
        started_ended = 'inside'
    elif annotation[0] >= window[0] and annotation[1] <= window[1]:  # annotation within window
        percentage = overlap / annotation_length
        status = 'full'
        started_ended = 'inside'
    elif window[1] < annotation[1]:  # Window is on the left side of the annotation
        percentage = overlap / window_length
        if percentage < 0.05:
            status = 'outside'
        elif percentage < 0.25:
            status = 'starting'
        elif percentage < 0.5:
            status = 'early'
        elif percentage < 0.75:
            status = 'middle'
        else:
            status = 'full'
        started_ended = 'started'
    else:  # Window is on the right side of the annotation
        percentage = overlap / window_length
        if percentage < 0.05:
            status = 'outside'
        elif percentage < 0.25:
            status = 'ending'
        elif percentage < 0.5:
            status = 'late'
        elif percentage < 0.75:
            status = 'middle'
        else:
            status = 'full'
        started_ended = 'ended'

    # select the first 2 decimal points of the percentage
    percentage = int(percentage * 10000) / 10000
    return percentage, status, started_ended


def find_status(
        start_frame, 
        end_frame,
        data_start_frame,
        data_end_frame,
        gesture_type,
        gestures_info
        ):
    window = (start_frame, end_frame)
    annotation = (data_start_frame, data_end_frame)
    if gesture_type != 'iconic':
        # first calculate the overlap percentages the current window has with all the gestures 
        # if there is an overlap more than 10% with any of the gestures, then this window is not good
        overlap_percentages = gestures_info.apply(lambda x: calculate_overlap(start_frame, end_frame, x['start_frame'], x['end_frame']), axis=1)
        if overlap_percentages.max() > 0.1:
            return -1, "outsie", ""
        percentage = overlap_percentages.max()
        status = 'outside'
        started_ended = ""
    else:
        percentage, status, started_ended = overlap_percentage(window, annotation)
    return percentage, status, started_ended

def get_rows_info(data, gestures_info, keypoints, pair, speaker, 
                  all_speakers_fps,
                  sequences_lengths,
                  all_sample_names,
                  sequences_labels,
                  all_pair_speaker_referent,  
                  config, gesture_type='iconic',
                  maximum_number_of_sequences=5000):
    selected = selected_joints[config]
    num_joints = len(selected)
    sample_counter = 0
    for row_index, row in data.iterrows():
        # if gesture_type != 'iconic':
            # if sample_counter > maximum_number_of_sequences * 3:
            #     break
        data_start_frame = row['start_frame']
        data_end_frame = row['end_frame']
        # get sequences of keypoints for the iconic gesture, starting before the start of the gesture, and ending after the end of the gesture, with an upper limit of max_seq_len, and using the sampling rate keypoints_sampling_rate
        #TODO check if the sequence is long enough, ALSO whether it is better to start from the beginning of the gesture - max_seq_len or - max_seq_len/2 or max_frame
        number_of_snippets = 0
        #TODO check if there is another gesture in the snippet amq 
        this_gesture_fp = np.zeros((int(max_seq_len), max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)
        # create a list of labels for the gesture, with the same length as the number of snippets
        this_gesture_labels = []
        this_gesture_labels = []
        this_gesture_sample_names = []
        this_pair_speaker_referent = []
        for i in range(data_start_frame - max_frame*2, data_end_frame + max_frame, keypoints_sampling_rate):
            start_frame = i
            end_frame = i + max_frame
            if end_frame > len(keypoints) - max_frame or start_frame < 0:
                break
            if number_of_snippets > max_seq_len-1:
                break
            
            percentage, status, _ = find_status(
                start_frame, end_frame,
                data_start_frame, data_end_frame,
                gesture_type, gestures_info
            )
            if percentage == -1:
                break
            
            sample_name = str(pair) + '_' + str(speaker) + '_' + str(start_frame) + '_' + str(end_frame)
            # if percentage <= 0:
            #     #TODO check if there is another gesture in the snippet
            skel = keypoints[start_frame:end_frame, selected, :]
            this_label = 'overlap_percentage_' + str(percentage) + '_status_' + status
            # print(this_label)
            # data that are for every snippet in the sequence
            this_gesture_labels.append(this_label)
            this_gesture_fp[number_of_snippets,:,:,:,0] = skel
            this_gesture_sample_names.append(sample_name)
            this_pair_speaker_referent.append(str(pair) + '_' + str(speaker) + '_' + str(row['referent']))
            number_of_snippets += 1
        
        number_of_snippets = len(this_gesture_labels)
        # do not add the gesture if it is too short, for now if number_of_snippets < 2
        if number_of_snippets < 2:
            print(row_index, i, 'snippet too short')
            continue
        if len(this_gesture_labels) < max_seq_len:
            # pad the rest of the sequence with the 
            rest = max_seq_len - len(this_gesture_labels)
            L = len(this_gesture_labels)
            # print(L)
            rest = max_seq_len - L
            num = int(np.ceil(rest / L))
            
            # add the corresponding fps
            pad = np.concatenate([this_gesture_fp for _ in range(num)], 0)[:rest]
            this_gesture_fp[L:,:,:,:,:] = pad
            
            # pad the corresponding labels, sample names and pair_speaker_referent
            pad = this_gesture_labels * num
            pad = pad[:rest]
            this_gesture_labels.extend(pad)

            pad = this_gesture_sample_names * num
            pad = pad[:rest]
            this_gesture_sample_names.extend(pad)
            
            pad = this_pair_speaker_referent * num
            pad = pad[:rest]
            this_pair_speaker_referent.extend(pad)
            
        sequences_lengths.append(number_of_snippets)
        this_gesture_fp = np.transpose(this_gesture_fp, [0, 3, 1, 2, 4])
        all_speakers_fps.append(this_gesture_fp)
        all_sample_names.append(this_gesture_sample_names)
        all_pair_speaker_referent.append(this_pair_speaker_referent)
        sequences_labels.append(this_gesture_labels)
        sample_counter += 1
    return all_speakers_fps, sequences_lengths, all_sample_names, sequences_labels, all_pair_speaker_referent

def generate_data_sw(
        data_dict, 
        gestures_info,
        pair,
        speaker,
        all_speakers_fps,
        sequences_lengths,
        all_sample_names,
        sequences_labels,
        current_ind,
        history=40, 
        time_offset=2, 
        num_frames=18, 
        ):
    """
        Generates a dataset from a tensor ```data```;

        Args:
            data (np.ndarray, *required*): 
                input data;
            gestures_info
            config
            pair
            speaker
            history (int, *optional*, default to 40): 
                how many timestamps take into consideration;
            time_offset (int, *optinal*, default to 2): 
                take every ```time_offset``` timestamps into consideration;
            num_frames (int, *optional*, default to 18):
                how many frames to consider for each timestamp;

        Output:
            ...
    """

    label_format = "overlap_percentage_{:7.5f}_status_{:7}"
    sample_format = "{:6}_{:1}_{:06d}_{:06d}"

    data = data_dict[(pair, speaker)]
    start_ind = 0
    end_ind = data.shape[0] - (history + num_frames) * time_offset
    for t in tqdm(range(start_ind, end_ind, time_offset*history), leave=False):
        percentages = list()
        statuses = list()
        this_sequence_keypoints = []
        for i in range(t, t + history*time_offset, time_offset):
            start_frame = i
            end_frame = i + num_frames
            this_sequence_keypoints.append(data[start_frame:end_frame, :, :])
            window = (start_frame, end_frame)
            all_intersections = gestures_info.apply(
                lambda x: 
                overlap_percentage(window, (x["start_frame"], x["end_frame"])), axis=1
                )
            all_intersections = list(all_intersections)
            percentages_i = [elem[0] for elem in all_intersections]
            percentage_i_argmax = percentages_i.index(max(percentages_i))
            percentage = percentages_i[percentage_i_argmax]
            status = [elem[1] for elem in all_intersections][percentage_i_argmax]
            percentages.append(percentage)
            statuses.append(status)
        label = [label_format.format(p,s) for p, s in zip(percentages, statuses)]
        sample_names = [
            sample_format.format(pair, speaker, i, i+num_frames)
            for i in range(t, t + history*time_offset, time_offset)
        ]
        all_sample_names[current_ind] = sample_names
        sequences_labels[current_ind] = label
        this_sequence_keypoints = np.array(this_sequence_keypoints)
        this_sequence_keypoints = np.transpose(this_sequence_keypoints, [0, 3, 1, 2])
        all_speakers_fps[current_ind] = this_sequence_keypoints
        current_ind += 1
    return all_speakers_fps, sequences_lengths, all_sample_names, sequences_labels, [None]*len(sequences_lengths), current_ind

def load_data(pairs, speakers, config):
    data_dict = dict.fromkeys(product(pairs, speakers))
    for pair, speaker in product(pairs, speakers):
        gesture_keypoints_path = 'videos/npy3/{}_synced_pp{}.npy'.format(pair, speaker)
        keypoints = np.load(gesture_keypoints_path)
        selected = selected_joints[config]
        keypoints = keypoints[:, selected, :]
        data_dict[(pair, speaker)] = keypoints
    return data_dict

def get_data_size(data_dict, pair_speaker, history, time_offset, num_frames):
    total_num_samples = 0
    for pair in pair_speaker:
        x = data_dict[pair].shape[0]
        case_num_samples = (x - (history + num_frames) * time_offset) // (time_offset * history) + 1
        total_num_samples += case_num_samples
    return total_num_samples

def get_part_pair_speaker(data_df):
    return {tuple(elem) for elem in data_df[["pair", "speaker"]].values}

def gendata(all_data, label_path, out_path, video_paths='', part='train', config='27', save_video=False):
    history = 40
    time_offset = 2
    num_frames = 18
    pairs = np.unique(all_data['pair'].to_numpy())
    speakers = np.unique(all_data['speaker'].to_numpy())
    data_dict = load_data(pairs, speakers, config)
    pair_speaker = get_part_pair_speaker(all_data)
    total_num_samples = get_data_size(data_dict, pair_speaker, history, time_offset, num_frames)
    all_speakers_fps = np.zeros((total_num_samples, history, 3, num_frames, int(config)))
    sequences_lengths = history*np.ones(total_num_samples)
    all_sample_names = np.chararray((total_num_samples, history), itemsize=22)
    sequences_labels = np.chararray((total_num_samples, history), itemsize=47)
    all_pair_speaker_referent = []
    current_ind = 0
    for pair, speaker in tqdm(pair_speaker, leave=True):
        data = all_data[(all_data['pair'] == pair) & (all_data['speaker'] == speaker)]
        data = data.reset_index(drop=True)
        # get the number of iconic gestures
        # get the necessary information for the iconic gestures: start frame, end frame, referent, label, speaker, pair
        data = data[['start_frame', 'end_frame', 'referent', 'label', 'speaker', 'pair', 'is_gesture']]
        # remove duplicate rows
        data = data.drop_duplicates()
        iconic_gestures = data[data['is_gesture'] == 'gesture'] #TODO, here we use already generated instances of gestures. When we move the sliding window to the gestures, we will need to generate them on the fly and their labels will be based on the overlap percentage with the gesture see overlap_percentage
     
        ret_val = generate_data_sw(
            data_dict=data_dict,
            gestures_info=iconic_gestures,
            pair=pair,
            speaker=speaker,
            all_speakers_fps=all_speakers_fps,
            sequences_lengths=sequences_lengths,
            all_sample_names=all_sample_names,
            sequences_labels=sequences_labels,
            current_ind=current_ind,
            history=history,
            time_offset=time_offset,
            num_frames=num_frames
        )
        all_speakers_fps, sequences_lengths, all_sample_names, sequences_labels, _, current_ind = ret_val

    c = Counter()
    for label in sequences_labels.decode():
        c.update([elem.split('_')[-1] for elem in label])

    print(c)

    overhead = c['']//history

    print("overhead = ", overhead)
    if overhead > 0:
        all_speakers_fps = all_speakers_fps[:-overhead]
        sequences_lengths = sequences_lengths[:-overhead]
        all_sample_names = all_sample_names[:-overhead]
        sequences_labels = sequences_labels[:-overhead]

    with open(label_path, 'wb') as f:
        pickle.dump((all_sample_names, all_pair_speaker_referent, sequences_labels, sequences_lengths), f)
        # pickle.dump((sample_names, pair_speaker_reerent, labels), f)
    all_speakers_fps = np.array(all_speakers_fps) 
    # fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print(all_speakers_fps.shape)
    np.save(out_path, all_speakers_fps)
    print('saved to {}'.format(out_path))

if __name__ == '__main__':
    # the following paths are for the fribbles dataset, for now you can ignore them
    out_folder = 'data/'
    videos_path = 'videos/{}_synced_pp{}.mp4'
    full_data_path = "full_data"

    # save gestures info which is a pandas dataframe, before that check if it exists
    normalize = False
    if normalize:
        normalization_step = '_normalized'
    else:
        normalization_step = ''

    gestures_info_path = os.path.join(out_folder, 'gestures_info_mmpose_full{}.pkl'.format(normalization_step))
    new_gestures_info_path = os.path.join(full_data_path, 'gestures_info_mmpose.pkl')
    print(new_gestures_info_path)
    if os.path.exists(new_gestures_info_path):
        gestures_info = pd.read_pickle(new_gestures_info_path)
    else:
        dialign_output = '/Users/esamghaleb/important_surfdrive/Research/CDM/Projects/LinguisticAlignmentFullGestures/code/dialign/output_targets_riws_lemma/'
        turn_info_path = '/Users/esamghaleb/important_surfdrive/Research/CDM/Projects/LinguisticAlignmentFullGestures/code/dialign/targets_riws_lemma/'
        fribbles_path = "/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/Fribbles/{}.jpg"
        temp_videos = 'temp_lex_align_videos/{}/{}_{}.mp4'
        temp_videos_path = '/'.join(temp_videos.split('/')[0:-1])
        shared_constructions_info, turns_info = prepare_dialogue_shared_expressions_and_turns_info(dialign_output, turn_info_path)
        gestures_info, gesture_data = get_gestures_info(turns_info, detector='mmpose')
        gestures_info.to_pickle(gestures_info_path.format(normalization_step))
    # gestures_info contains the following columns: ['pair', 'speaker', 'start_frame', 'end_frame', 'referent', 'label', 'is_gesture', 'keypoints']
    out_folder = 'data/sw_esam_1606_final/{}'
    # drop the keypoint column
    labels = np.array(gestures_info['label'].to_list())
    unique_labels = np.unique(labels)
    gesture_ids = {}
    for i, label in enumerate(unique_labels):
        gesture_ids[label] = i
    discrete_labels = []
    for label in labels:
        discrete_labels.append(gesture_ids[label])
    gestures_info['pair_speaker'] = gestures_info.apply(lambda x: str(x['pair']) + '_' + str(x['speaker']), axis=1)    
    # gestures_info = gestures_info[gestures_info['is_gesture'] == 'gesture']
    subject_ids = np.array(gestures_info['pair_speaker'])

    folds, subject_ids = return_cross_validation_ids(discrete_labels, subject_ids, num_folds=5, kfold_cross_validation=False)
    unique_folds = np.unique(folds)
    for fold in unique_folds[:1]:
        print(fold)
        this_out_folder = out_folder.format(fold)
        if not os.path.exists(this_out_folder):
            os.makedirs(this_out_folder)
        print('*'*20)
        print('fold {}'.format(fold))
        for p in ['test', 'train']:
            if p == 'test':
                data = gestures_info[folds == fold]
            else:
                data = gestures_info[folds != fold]
            # CABB_dataset_length_%d_sample_rate_%d_iou_%f/train/gesture/1_1_snippet_id1.npz
            data_out_path = '{}/{}_data_27_joint{}_esam.npy'.format(this_out_folder, p, normalization_step)
            label_out_path = '{}/{}_27_label{}_esam.pkl'.format(this_out_folder, p, normalization_step)
            referents_speakers_path = '{}/{}_27_referent_speakers{}_esam.pkl'.format(this_out_folder, p, normalization_step)
            gendata(data, label_out_path, data_out_path, video_paths=videos_path, part=p)