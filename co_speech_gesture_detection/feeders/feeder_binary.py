import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import random
import tools

mediapipe_flip_index = np.concatenate(([0,2,1,4,3,6,5], [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, 27], ), axis=0) 

mmpose_flip_index = np.concatenate(([0,2,1,4,3,6,5],[17,18,19,20,21,22,23,24,25,26],[7,8,9,10,11,12,13,14,15,16]), axis=0) 

# number of frames in a video to be considered
max_rgb_frame = 18 # 67% OF THE ICONIC GESTURES HAVE THIS LENGTH, which is the mean duration of the iconic gestures. We can keep this as it is since we are using the sliding windows approach, to determin the boundaris of the iconic gestures
sample_rate = 2
iou = 100
max_body_true = 1
max_frame = max_rgb_frame
num_channels = 3
keypoints_sampling_rate = 2 # this is the sampling rate of the keypoints, chose every 4th frame
max_seq_len = 40 # this is the upper bound of the number of frames in a video, given the confidence interval of 95% and the number of frames in the videos


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=False, random_mirror=False, random_mirror_p=0.5, is_vector=False, fold=0, max_length=40):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param random_mirror: If true, randomly mirror the input sequence
        :param max_length: The maximum length of the input sequence
        """

        self.debug = debug
        self.data_path = data_path.format(fold)
        self.label_path = label_path.format(fold)
        self.fold = fold
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_mirror = random_mirror
        self.random_mirror_p = random_mirror_p
        self.load_data()
        self.is_vector = is_vector
        if normalization:
            self.get_mean_map()
        print(len(self.label))
        self.labels_dict = {'outside left': 0, 'starting': 0, 'early': 0, 'middle': 1, 'full': 1, 'outside right': 0, 'ending': 0, 'late': 0, 'outside': 0}

    def load_data(self):
        # data: N C V T M for each sequence, hence sequene data is a list of N C V T M arrays. Then data: S N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.pair_speaker_referent, self.label, self.lengths = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.pair_speaker_referent, self.label, self.lengths = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        # self.data = np.expand_dims(self.data, axis=-1)
        if self.debug:
            # choose randomly 1000 samples
            random.seed(0)
            idx = random.sample(range(len(self.label)), 1000)
            # self.label = self.label[idx]
            # self.data = self.data[idx]
            # self.sample_name = self.sample_name[idx]
            # self.pair_speaker_referent = self.pair_speaker_referent[idx]
            # self.lengths = self.lengths[idx]

            self.label = [self.label[i] for i in idx]
            self.data = np.array([self.data[i] for i in idx])
            self.sample_name = [self.sample_name[i] for i in idx]
            # self.pair_speaker_referent = [self.pair_speaker_referent[i] for i in idx]
            self.lengths = [self.lengths[i] for i in idx]
            

    def get_mean_map(self):
        data = self.data
        print(data.shape)
        N, S, C, T, V, M = data.shape # --> N, S, C, T, V, M , so we need to reshape it to N * S, C, T, V, M
        data = data.reshape((N * S, C, T, V, M))
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * S* T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
        

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        labels = self.label[index]
        lengths = int(self.lengths[index])
        labels = np.array([self.labels_dict[l.split('_')[4].rstrip()] for l in labels])

        # labels = np.array([self.labels_dict[l.tostring().split('_')[4]] for l in labels])
        # data: S N C V T M. first convert the data into SN C V T M
        data_numpy = np.array(data_numpy)
        S, C, V, T, M = data_numpy.shape
        # real labels 
        # actual_label = [l.split('_')[4] for l in self.label[index]]
        # speaker and referent
        # data['pair_speaker_referent'] = data.apply(lambda x: str(x['pair']) + '_' + str(x['speaker']) + '_' + str(x['referent']), axis=1)
        # pair_speaker_referent = data['pair_speaker_referent'].to_numpy()
        # sample_names  = data.apply(lambda x: str(x['pair']) + '_' + str(x['speaker']) + '_' + str(x['start_frame']) + '_' + str(x['end_frame']), axis=1).to_numpy()
        #TODO apply this for every speaker, as we have self.max_length speakers, convert them to np.array as well
        # speaker = '_'.join(self.pair_speaker_referent[index][0].split('_')[0:2])
        # referent = self.pair_speaker_referent[index][0].split('_')[2]
        start_frame = self.sample_name[index][0].split('_')[2]
        end_frame = self.sample_name[index][0].split('_')[3]
        # make a dictionary to store the data
        labels_dict = {'label': labels, 'start_frame': int(start_frame), 'end_frame': int(end_frame), 'fold': self.fold}

        # labels_dict = {'label': labels, 'actual_label': actual_label, 'start_frame': int(start_frame), 'end_frame': int(end_frame), 'keypoints': data_numpy, 'fold': self.fold}
        if self.random_mirror:
            if random.random() > self.random_mirror_p:
                if data_numpy.shape[3] == 49:
                    flip_index = mediapipe_flip_index
                if data_numpy.shape[3] == 27:
                    flip_index = mmpose_flip_index
                data_numpy = data_numpy[:,:,:,flip_index,:]
                if self.is_vector:
                    data_numpy[:, 0,:,:,:] = - data_numpy[:, 0,:,:,:]
                else: 
                    data_numpy[:, 0,:,:,:] = 512 - data_numpy[:, 0,:,:,:]
        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            assert data_numpy.shape[1] == 3
            S, C, V, T, M = data_numpy.shape
            assert data_numpy.shape[1] == 3
            if self.is_vector:
                data_numpy[:, 0,:,0,:] -= data_numpy[:, 0,:,0,0].mean(axis=1, keepdims=True)
                data_numpy[:, 1,:,0,:] -= data_numpy[:, 1,:,0,0].mean(axis=1, keepdims=True)
            else:
                mean_0 = data_numpy[:, 0,:,0,0].mean(axis=1, keepdims=True)
                mean_1 = data_numpy[:, 1,:,0,0].mean(axis=1, keepdims=True)
                data_numpy[:, 0,:,:,:] -= mean_0.reshape(mean_0.shape[0], 1, 1, 1)
                data_numpy[:, 1,:,:,:] -= mean_1.reshape(mean_1.shape[0], 1, 1, 1)

        if self.random_shift:
            if self.is_vector:
                data_numpy[:, 0,:,0,:] += random.random() * 20 - 10.0
                data_numpy[:, 1,:,0,:] += random.random() * 20 - 10.0
            else:
                data_numpy[:, 0,:,:,:] += random.random() * 20 - 10.0
                data_numpy[:, 1,:,:,:] += random.random() * 20 - 10.0
        for i in range(S):
            # TODO, if you want to apply preprocessing, do it here taken into account the number of segments
            if self.random_choose:
                data_numpy[i] = tools.random_choose(data_numpy[i], self.window_size)
            # if self.random_shift:
            #     data_numpy = tools.random_shift(data_numpy)

            # elif self.window_size > 0:
            #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
            if self.random_move:
                data_numpy[i] = tools.random_move(data_numpy[i])

        return data_numpy, labels, lengths, index, labels_dict

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
    