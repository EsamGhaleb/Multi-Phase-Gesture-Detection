import numpy as np
import pandas as pd
import os
from scipy.stats import entropy
import math 
import pickle
import re 
from collections import defaultdict, Counter


class Turn:
    ''' This class holds information for each turn''' 
    def __init__(self, speaker, turn_ID, target_turn, utterance, gestures, duration, 
    from_ts, to_ts, round, trial, target, director, correct_answer, given_answer, accuracy, dataset):
        self.speaker = speaker
        self.ID = turn_ID
        self.utterance = utterance
        self.lemmas_with_pos = ''
        self.pos_sequence = ''
        self.lemmas_sequence = ''
        self.text_lemma_pos = ''
        self.gesture = gestures
        self.duration = duration
        self.from_ts = from_ts
        self.to_ts = to_ts 
        self.target = target
        self.trial = trial
        self.round = round
        self.director = director
        self.correct_answer = correct_answer
        self.given_answer = given_answer
        self.accuracy = accuracy
        self.dataset = dataset
        self.target_turn = target_turn
        self.utterance_speech = []
    def __str__(self) -> str:
        return 'Speaker is {} with utterance \"{}\". The trial is {} where the director is {} talking about {}'.format(self.speaker, 
        self.utterance, self.trial, self.director, self.target)
    def set_lemmas_with_pos(self, lemmas_with_pos):
        self.lemmas_with_pos = lemmas_with_pos
    def set_pos_sequence(self, pos_sequence):
        self.pos_sequence = pos_sequence
    def set_lemmas_sequence(self, lemmas_sequence):
        self.lemmas_sequence = lemmas_sequence
    def set_text_lemma_pos(self, text_lemma_pos):
        self.text_lemma_pos = text_lemma_pos
    def set_ID(self, ID):
        self.ID = ID
    def set_target_turn(self, target_turn):
        self.target_turn = target_turn
    def set_utterance_speech(self, utterance_speech):
        self.utterance_speech = utterance_speech
class Utterance:
    ''' This class holds information for each utterance'''
    def __init__(self, word, from_ts, to_ts, lemma='', pos='', lemma_pos=''):
        self.word = word
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.lemma = lemma
        self.pos = pos
        self.lemma_pos = lemma_pos
    def set_lemma(self, lemma):
        self.lemma = lemma
    def set_pos(self, pos):
        self.pos = pos
    def set_lemma_pos(self, lemma_pos):
        self.lemma_pos = lemma_pos
    def __str__(self) -> str:
        return 'word is \"{}\", from_ts is {}, to_ts is {}'.format(self.word, self.from_ts, self.to_ts)
class Gesture:
    ''' A class to include information about gestures
        In one turn, we can have multiple gestures. These gestures can also be from another speaker (when there is an overlap between speakers)
    '''
    def __init__(self, is_gesture, g_from_ts, g_to_ts, g_type, g_referent, g_comment, g_hand):
        self.is_gesture = is_gesture
        self.g_type = g_type
        self.g_referent = g_referent
        self.g_comment = g_comment
        self.g_hand = g_hand
        self.g_from_ts = g_from_ts
        self.g_to_ts = g_to_ts

def load_pickle(file_name):
   with open(file_name, 'rb') as reader:
        b = pickle.load(reader)
        return b

def calculate_entropy(targets):
    bins=np.arange(0, 17)
    max_entropy = math.log(len(bins)-1,2)
    exp_entropy = entropy(np.histogram(np.array(targets)-1, bins=bins)[0], base=2)/max_entropy
    return exp_entropy

def read_annotated_pos_data(dialign_output, turn_info_path):
   shared_constructions_info = defaultdict(list)
   targets_turns_info = defaultdict(list)
   turns_info = defaultdict()
   for f in os.listdir(dialign_output):
      if f.endswith('_tsv-lexicon.tsv') and not f.startswith('.'):
         filepath = os.path.join(dialign_output, f)
         files_parts = filepath.split('_')
         fribble_ID = files_parts[-2]
         pair_name = files_parts[-3].split('/')[1]
         all_turns_info_path = turn_info_path+pair_name+'.pickle'
         targets_turns_info_path = turn_info_path+pair_name+'_'+fribble_ID+'.pickle'
         pair_target_shared_expressions = pd.read_csv(filepath, sep='\t', header=0)
         with open(targets_turns_info_path, 'rb') as reader:
               targets_pair_turns_info = pickle.load(reader)
        
         for i, row in pair_target_shared_expressions.iterrows():
            turns =  [str(targets_pair_turns_info[int(turn)].ID) for turn in row['Turns'].split(',')]
            pair_target_shared_expressions.loc[i, 'Turns'] = ", ".join(turns)
            pair_target_shared_expressions.loc[i, 'Establishment turn'] = targets_pair_turns_info[int(row['Establishment turn'])].ID
            pair_target_shared_expressions.loc[i, 'Spanning'] = targets_pair_turns_info[int(row['Establishment turn'])].ID - int(turns[0])
         shared_constructions_info[pair_name].append(pair_target_shared_expressions)
         targets_turns_info[pair_name].extend(targets_pair_turns_info)
         with open(all_turns_info_path, 'rb') as reader:
               turns_info[pair_name] = pickle.load(reader)
   for pair_name in shared_constructions_info.keys(): 
      shared_constructions_info[pair_name]= pd.concat(shared_constructions_info[pair_name], axis=0)
      shared_constructions_info[pair_name].reset_index(drop=True, inplace=True)
   return shared_constructions_info, turns_info

def create_exp_info_row(expression, length, targets, target_fribbles, num_targets, exp_entropy, speakers, freq, free_freq, priming, spanning_rounds, spanning_time, duration_to_emerge, turns_to_emerge, rounds_to_emerge, turns, rounds, establishment_round, establishment_turn,  first_round, last_round, initiators, shared_exp_pos_seq, shared_exp_pos, pair, dataset):
    return { 'exp': expression, 'length': length, 'fribbles': targets, 'target_fribbles': target_fribbles,  '#fribbles': num_targets, 'fribbles entropy': exp_entropy, 'speakers': speakers, 'freq': freq, 'free. freq': free_freq, 'priming': priming, 'spanning_rounds': spanning_rounds, 'spanning_time': spanning_time, 'duration_to_emerge': duration_to_emerge, 'turns_to_emerge':turns_to_emerge, 'rounds_to_emerge': rounds_to_emerge, 'turns': turns, 'rounds': rounds, 'estab_round': establishment_round, 'estab_turn': establishment_turn, 'first_round': first_round, 'last_round': last_round, 'initiator': initiators, 'pos_seq': shared_exp_pos_seq, 'exp with pos': shared_exp_pos, 'pair': pair, 'dataset': dataset}

def extract_all_shared_exp_info(shared_constructions_info, turns_info):
    first_row = True
    exp_info = pd.DataFrame()
    for pair in shared_constructions_info:
        shared_expressions = shared_constructions_info[pair]['Surface Form'].to_list()
        all_turns = shared_constructions_info[pair]['Turns'].to_list()
        free_freq = shared_constructions_info[pair]['Free Freq.'].to_list()
        priming = shared_constructions_info[pair]['Priming'].to_list()
        spanning = shared_constructions_info[pair]['Spanning'].to_list()
        freqs = shared_constructions_info[pair]['Freq.'].to_list()
        lengths = shared_constructions_info[pair]['Size'].to_list()
        initiators = shared_constructions_info[pair]['First Speaker'].to_list()
        establishment_turns = shared_constructions_info[pair]['Establishment turn'].to_list()
        shared_exp_pos = shared_constructions_info[pair]['Exp with POS'].to_list()
        shared_exp_pos_seq = shared_constructions_info[pair]['POS'].to_list()
    
        for idx, turns in enumerate(all_turns):
            all_turns[idx] = np.array(turns.split(','), dtype=int)
        for idx, turns in enumerate(all_turns):
            targets = [int(turns_info[pair][turn].target) for turn in turns]
            dataset = turns_info[pair][turns[0]].dataset
            speakers = [turns_info[pair][turn].speaker for turn in turns]
            rounds = [int(turns_info[pair][turn].round) for turn in turns]
            num_targets = len(set(targets))
            establishment_turn = int(establishment_turns[idx])
            establishment_round = int(turns_info[pair][establishment_turn].round)         
            
            spanning_time = turns_info[pair][turns[-1]].to_ts - turns_info[pair][establishment_turn].from_ts 
            spanning_rounds = int(turns_info[pair][turns[-1]].round) - int(turns_info[pair][turns[0]].round)

            last_round = turns_info[pair][turns[-1]].round
            first_round = turns_info[pair][turns[0]].round
            rounds_to_emerge = int(turns_info[pair][establishment_turn].round) - int(turns_info[pair][turns[0]].round)
            turns_to_emerge = establishment_turn - turns[0]
            
            duration_to_emerge = turns_info[pair][turns[-1]].to_ts - turns_info[pair][establishment_turn].from_ts
            exp_entropy = calculate_entropy(targets)
            target_fribbles = np.unique(targets)
        

            this_exp_info = create_exp_info_row(shared_expressions[idx], lengths[idx], targets, target_fribbles, num_targets, exp_entropy, speakers, freqs[idx], free_freq[idx], priming[idx], spanning_rounds, spanning_time, duration_to_emerge, turns_to_emerge, rounds_to_emerge, turns, rounds, establishment_round, establishment_turn, first_round, last_round, initiators[idx], shared_exp_pos_seq[idx], shared_exp_pos[idx], pair, dataset)
            if first_row:
                exp_info = pd.Series(this_exp_info).to_frame().T
                first_row = False
            else:
                exp_info = pd.concat([exp_info, pd.Series(this_exp_info).to_frame().T])
                # print(exp_info)
          
                
    exp_info = exp_info.reset_index(drop=True)
    return exp_info

def prepare_dialogue_shared_expressions_and_turns_info(dialign_output, turn_info_path):
    shared_constructions_info, turns_info = read_annotated_pos_data(dialign_output, turn_info_path)
    for pair in shared_constructions_info:
        shared_exp_pos = shared_constructions_info[pair]['Surface Form'].copy()
        shared_exp_turns = shared_constructions_info[pair]['Turns'].copy()
        for idx, turns in enumerate(shared_exp_turns):
            shared_exp_turns[idx] = np.array(turns.split(','), dtype=int)
        shared_exp = []
        pos_seq_shared_exp = []
        expressions_with_pos = []
        for exp_idx, exp in enumerate(shared_exp_pos):
            pos_seq = []
            turns = shared_exp_turns[exp_idx]        
            exp_patterns = r"(" + r"#[\w]+ ".join(exp.split(' '))+ r"#[\w]+)"
            exp_patterns = exp_patterns.replace('##', '#')
            exp_patterns = exp_patterns.replace('?', '\?')
            # exp = exp_patterns.replace(', #PUNCT', ',#PUNCT')
            exp_patterns = exp_patterns.replace(')#', '\)#')
            exp_patterns = exp_patterns.replace('((', '(\(')
            exp_patterns = exp_patterns.replace('*', '\*')
            exp_with_pos_all_turns = []
            for turn in turns:
                turn = int(turn)
                exp_with_pos = re.findall(exp_patterns, turns_info[pair][turn].lemmas_with_pos)
                # print(turns_info[pair][turn].pos_sequence)
                exp_with_pos_all_turns.append(exp_with_pos[0])
            exp_with_pos_all_turns = Counter(exp_with_pos_all_turns)
            exp_with_pos = max(exp_with_pos_all_turns, key=exp_with_pos_all_turns.get)
            # exp_with_pos = exp_with_pos.replace('_', '#')
            expressions_with_pos.append(exp_with_pos)
            for word in exp_with_pos.split(' '):
                pos_seq.append(word.split('#')[1])
            pos_seq_shared_exp.append(" ".join(pos_seq))
        # shared_constructions_info[pair]['Surface Form'] = shared_exp
        assert len(pos_seq_shared_exp) == len(shared_exp_pos)
        shared_constructions_info[pair]['POS'] = pos_seq_shared_exp
        shared_constructions_info[pair]['Exp with POS'] = expressions_with_pos 
    return shared_constructions_info, turns_info

import pandas as pd
def get_gestures_info(turns_info, detector='mediapipe', normalize=False):
    keypoints_path = '/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_pp{}_landmarks.npy'
    pairs_name = ['pair10', 'pair17', 'pair15', 'pair21', 'pair18', 'pair24', 'pair16', 'pair13', 'pair07', 'pair22', 'pair20',  'pair08', 'pair05', 'pair11', 'pair23', 'pair04', 'pair09', 'pair12', 'pair14' ]
    gestures_info = []
    fps = 29.97002997002997
    abs_min_frame_length = 4
    for pair in pairs_name:
        if detector == 'mediapipe':
            gesture_keypoints_path_A = '/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_pp{}_landmarks.npy'.format(pair, pair, 'A')
            gesture_keypoints_path_B = '/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_pp{}_landmarks.npy'.format(pair, pair, 'B')
        elif detector == 'mmpose':
            gesture_keypoints_path_A = 'data/videos/npy3/{}_synced_pp{}.npy'.format(pair, 'A')
            gesture_keypoints_path_B = 'data/videos/npy3/{}_synced_pp{}.npy'.format(pair, 'B')

        gesture_keypoints_A = np.load(gesture_keypoints_path_A, allow_pickle=True)
        gesture_keypoints_B = np.load(gesture_keypoints_path_B, allow_pickle=True)
        for turn in turns_info[pair]:
            if turn.gesture:
                for gesture in turn.gesture:
                    # start frame 
                    start_frame = int(gesture.g_from_ts * fps)
                    end_frame = int(gesture.g_to_ts * fps)
                    if turn.speaker == 'A':
                        gesture_keypoints = gesture_keypoints_A[start_frame:end_frame, :, :]
                    else:
                        gesture_keypoints = gesture_keypoints_B[start_frame:end_frame, :, :]
                    gesture_keypoints = np.array(gesture_keypoints)
                    # divide the x and y coordinates by the width and height of the image, which is 1920 x 1080
                    if normalize:
                        gesture_keypoints[:, :, 0] = gesture_keypoints[:, :, 0] / 1920
                        gesture_keypoints[:, :, 1] = gesture_keypoints[:, :, 1] / 1080
                    gesture_info = {'pair': pair, 'turn': turn.ID, 'type': gesture.g_type, 'is_gesture': gesture.is_gesture, 'from_ts': gesture.g_from_ts, 'to_ts': gesture.g_to_ts, 'hand': gesture.g_hand, 'referent': gesture.g_referent, 'comment': gesture.g_comment, 'keypoints': gesture_keypoints, 'speaker': turn.speaker, 'start_frame': start_frame, 'end_frame': end_frame}
                    # label is iconic or not
                    if abs(end_frame - start_frame) < abs_min_frame_length:
                        continue
                    if gesture_keypoints.shape[0] < abs_min_frame_length:
                        continue
                    if gesture.g_type == 'iconic':
                        gesture_info['label'] = 'iconic'
                    elif gesture.is_gesture == 'gesture':
                        gesture_info['label'] = 'gesture'
                    else:
                        gesture_info['label'] = 'other'
                    gestures_info.append(gesture_info)
                
    temp_gestures_info = pd.DataFrame(gestures_info)
    max_frame_length = 60
    min_frame_length = abs_min_frame_length
    temp_gestures_info['duration'] = temp_gestures_info['to_ts'] - temp_gestures_info['from_ts']
    gestures_durations = temp_gestures_info['duration'].to_list()
    mle_rate_gestures_duration =  np.mean(gestures_durations)
    mle_std_gestures_duration = np.std(gestures_durations)
    import scipy.stats as st

    for pair in pairs_name:
        pair_turns_info = turns_info[pair]
        if detector == 'mmpose':
            gesture_keypoints_path_A = 'data/videos/npy3/{}_synced_pp{}.npy'.format(pair, 'A')
            gesture_keypoints_path_B = 'data/videos/npy3/{}_synced_pp{}.npy'.format(pair, 'B')
        elif detector == 'mediapipe':
            gesture_keypoints_path_A = '/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_pp{}_landmarks.npy'.format(pair, pair, 'A')
            gesture_keypoints_path_B = '/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_pp{}_landmarks.npy'.format(pair, pair, 'B')
        gesture_keypoints_A = np.load(gesture_keypoints_path_A, allow_pickle=True)
        gesture_keypoints_B = np.load(gesture_keypoints_path_B, allow_pickle=True)
        # generate a list of start and end frames randomly, and then get the keypoints for that frame
        # the start frame should be less than the length of the video
        starts_frames = []
        ends_frames = []
        A_gestures_info  = temp_gestures_info[(temp_gestures_info['pair'] == pair) & (temp_gestures_info['speaker'] == 'A')]
        B_gestures_info  = temp_gestures_info[(temp_gestures_info['pair'] == pair) & (temp_gestures_info['speaker'] == 'B')]
        for i in range(500):
            start_frame = np.random.randint(0, len(gesture_keypoints_A)-max_frame_length)
            # the end frame should be greater than the start frame, and less than the length of the video, and the difference between them should be greater than 30 frames and less than 100 frames
            duration = st.expon.rvs(scale=mle_rate_gestures_duration, size=1)
            # convert number of frames
            frames_length = int(duration * fps)
            end_frame = start_frame + frames_length
            if end_frame > len(gesture_keypoints_A):
                continue

            # end_frame = np.random.randint(start_frame + min_frame_length, start_frame + max_frame_length)
            # if the start and the end frame overlap with the iconic gesture, then skip this iteration
            if len(A_gestures_info[(A_gestures_info['start_frame'] >= start_frame) & (A_gestures_info['end_frame'] <= end_frame)]) > 0 or len(A_gestures_info[(start_frame >= A_gestures_info['start_frame']) & ( end_frame <= A_gestures_info['end_frame'])]) > 0:         
                # print('start frame {} and end frame {} is overlap with iconic gesture'.format(start_frame, end_frame))
                continue
            # else:
            #    print('start frame {} and end frame {} do not overlap with iconic gesture'.format(start_frame, end_frame))
            if end_frame - start_frame < min_frame_length:
                continue
            starts_frames.append(start_frame)
            ends_frames.append(end_frame)
        for start_frame, end_frame in zip(starts_frames, ends_frames):
            gesture_keypoints = gesture_keypoints_A[start_frame:end_frame, :, :]
            gesture_keypoints = np.array(gesture_keypoints)
            if gesture_keypoints.shape[0] < min_frame_length:
                continue
            # divide the x and y coordinates by the width and height of the image, which is 1920 x 1080
            if normalize:
                gesture_keypoints[:, :, 0] = gesture_keypoints[:, :, 0] / 1920
                gesture_keypoints[:, :, 1] = gesture_keypoints[:, :, 1] / 1080
            start_time = start_frame / fps
            end_time = end_frame / fps
            gesture_info = {'pair': pair, 'turn': None, 'type': 'non-iconic', 'is_gesture': 'not', 'from_ts': start_time, 'to_ts': end_time, 'hand': 'uknown', 'referent': 'unknown', 'comment': None, 'keypoints': gesture_keypoints, 'speaker': 'A', 'start_frame': start_frame, 'end_frame': end_frame, 'label': 'other'}
            gestures_info.append(gesture_info)
        # do the same for the other participant
        starts_frames = []
        ends_frames = []
        for i in range(500):
            start_frame = np.random.randint(0, len(gesture_keypoints_B)-max_frame_length)
            # the end frame should be greater than the start frame, and less than the length of the video, and the difference between them should be greater than 30 frames and less than 100 frames
            # end_frame = np.random.randint(start_frame + 15, start_frame + max_frame_length)
            # if the start and the end frame overlap with the iconic gesture, then skip this iteration
            duration = st.expon.rvs(scale=mle_rate_gestures_duration, size=1)
            # convert number of frames
            frames_length = int(duration * fps)
            end_frame = start_frame + frames_length
            if end_frame > len(gesture_keypoints_B):
                continue
            if len(B_gestures_info[(B_gestures_info['start_frame'] >= start_frame) & (B_gestures_info['end_frame'] <= end_frame)]) > 0 or len(B_gestures_info[(start_frame >= B_gestures_info['start_frame']) & ( end_frame <= B_gestures_info['end_frame'])]) > 0:
                # print('start frame {} and end frame {} is overlap with iconic gesture'.format(start_frame, end_frame))
                continue
            if end_frame - start_frame < min_frame_length:
                continue
            starts_frames.append(start_frame)
            ends_frames.append(end_frame)
        for start_frame, end_frame in zip(starts_frames, ends_frames):
            gesture_keypoints = gesture_keypoints_B[start_frame:end_frame, :, :]
            gesture_keypoints = np.array(gesture_keypoints)
            if gesture_keypoints.shape[0] < min_frame_length:
                continue
            # divide the x and y coordinates by the width and height of the image, which is 1920 x 1080
            if normalize:
                gesture_keypoints[:, :, 0] = gesture_keypoints[:, :, 0] / 1920
                gesture_keypoints[:, :, 1] = gesture_keypoints[:, :, 1] / 1080
            start_time = start_frame / fps
            end_time = end_frame / fps

            gesture_info = {'pair': pair, 'turn': None, 'type': 'non-iconic', 'is_gesture': 'not', 'from_ts': start_time, 'to_ts': end_time, 'hand': 'uknown', 'referent': 'unknown', 'comment': None, 'keypoints': gesture_keypoints, 'speaker': 'B', 'start_frame': start_frame, 'end_frame': end_frame, 'label': 'other'}

            gestures_info.append(gesture_info)

    gestures_info = pd.DataFrame(gestures_info)
    gestures_info['duration'] = gestures_info['to_ts'] - gestures_info['from_ts']
    gesture_data = gestures_info.groupby('type').count().reset_index().rename(columns={'pair': 'count'})[['type', 'count']]
    gesture_data['normalized_count'] = gesture_data['count'] / gesture_data['count'].sum()
    gesture_data['normalized_count'] = gesture_data['normalized_count'].apply(lambda x: round(x, 2))

    return gestures_info, gesture_data
    
def assert_match_betwee_shared_exp_and_actual_utterances(turns_info, shared_constructions_info):
    all_pairs = shared_constructions_info.keys()
    for pair in all_pairs:
        for row_idx, row in shared_constructions_info[pair].iterrows():
            exp = row['Exp with POS']
            exp_patterns = [r"[\w'?]+_"+exp_pos for exp_pos in exp.split(' ')]
            exp_patterns_to_extract_tokens = r"(" + " ".join(exp_patterns)+ r")"
            exp_patterns_to_extract_tokens = exp_patterns_to_extract_tokens.replace(r'?', r'\?')
            turns = row['Turns'].split(',')
            exp_lemmas = " ".join([exp_pos.split('#')[0] for exp_pos in exp.split(' ')])
            for turn in turns:
                assert exp_lemmas in turns_info[pair][int(turn)].lemmas_sequence

if __name__ == "__main__":
    dialign_output = '/Users/esamghaleb/important_surfdrive/Research/CDM/SharedConstructions/exp-rep-main/code/dialign/output_targets_riws_lemma/'
    turn_info_path = '/Users/esamghaleb/important_surfdrive/Research/CDM/SharedConstructions/exp-rep-main/code/dialign/targets_riws_lemma/'
    fribbles_path = "/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/Fribbles/{}.jpg"
    videos_path = '/Users/esamghaleb/Documents/ResearchData/CABB Small Dataset/processed_audio_video/{}/{}_synced_overview.mp4'
    temp_videos = 'temp_lex_align_videos/{}/{}_{}.mp4'
    
    temp_videos_path = '/'.join(temp_videos.split('/')[0:-1])


    shared_constructions_info, turns_info = prepare_dialogue_shared_expressions_and_turns_info(dialign_output, turn_info_path)
    gestures_info, gesture_data = get_gestures_info(turns_info)
   