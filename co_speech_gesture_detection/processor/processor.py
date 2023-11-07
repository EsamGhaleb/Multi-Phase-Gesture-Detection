import os
import shutil
import time
from collections import OrderedDict
import shutil
import inspect
import pickle

import yaml
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchcrf import CRF
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import Levenshtein as lev
from sklearn.metrics import precision_recall_fscore_support

from .. import utils
from .. import loss


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, arg):
        super().__init__()
        arg.log_base_path = "/home/atuin/b105dc/data/work/iburenko/logs/gcn"
        arg.model_saved_name = os.path.join(
            arg.log_base_path,
            "save_models",
            arg.Experiment_name
        )
        arg.work_dir = os.path.join(
            arg.log_base_path,
            "work_dir",
            arg.Experiment_name
        )
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input(
                            'Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
        self.global_step = 0
        self.this_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.lr = self.arg.base_lr
        self.pad = 0
        self.best_f1 = 0
        # give the summary writer a name based on the fold, and put it in runs folder
        self.writer = SummaryWriter("logs/" + arg.Experiment_name)

    def markov_greedy(self,x, lengths):
        """
        Draws a number of samples from the model, each sample is a complete sequence.
        We impose a maximum number of steps, to avoid infinite loops.
        This procedure takes care of mapping sampled symbols to pad after the EOS symbol is generated.
        """
        batch_size = x.shape[0]
        max_length = x.shape[1]

        with torch.no_grad():
            # add the beginning we do not know the tag sequence
            # but NNs work with fixed dimensional tensors, 
            # so we allocate a tensor full of BOS codes
            y = torch.full((batch_size, max_length), self.bos, device=x.device) 
            # Per step
            embeddings = self.model(x, y, eval=True, get_embeddings=True, lengths=lengths)
            # generated_lenghts are the lengths of the sequences we have generated so far, so they are have the same shape as lengths, but initialized to 0
            generated_lenghts = torch.zeros_like(lengths) + 1
            
            for i in range(max_length):
                # we parameterise a cpd for Y[i]|X=x
                # note that the forward method takes care of not conditioning on y[i] itself
                # and only using the ngram_size-1 previous tags
                # at this point, the tag y[i] is a dummy code
                # the forward method recomputes all cds in the batch, this will include the cpd for Y[i]
                # [batch_size, max_len, C] 
                probs = self.model(x, y, eval=True, get_embeddings=False, get_predictions=True, u=embeddings, lengths=generated_lenghts)
                # probs are raw probabilities, we need to turn them into a distribution
                probs = F.softmax(probs, -1)
                generated_lenghts += 1
                for j in range(batch_size):
                    if generated_lenghts[j] > lengths[j]:
                        generated_lenghts[j] = lengths[j]
                # we get their modes via argmax
                # [batch_size, max_len]
                modes = torch.argmax(probs, -1)
                
                # Here we update the current token to the freshly obtained mode
                #  and also replace the token by 0 (pad) in case the sentence is already complete
                y[:, i] = modes[:, i]                
            # where we had a PAD token in x, we change the y token to PAD too
            # select only the first two dimensions from x
            # x = x[:,:, 0, 0, 0, 0]
            # y = torch.where(x != self.pad, y, torch.zeros_like(y) + self.pad)
            return y
        
    def crf_auto_decoder(self, x, lengths):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        # [batch_size, max_length]
        batch_size = x.shape[0]
        max_length = self.arg.labeler_args['max_seq_len']
        y = torch.full((batch_size, max_length), self.bos, device=x.device) 
        embeddings = self.model(x, y, eval=True, get_embeddings=True, lengths=lengths)
        with torch.no_grad():
            # add the beginning we do not know the tag sequence
            # but NNs work with fixed dimensional tensors, 
            # so we allocate a tensor full of BOS codes
            y = torch.full((batch_size, max_length), self.bos, device=x.device) 
            # Per step
            embeddings = self.model(x, y, eval=True, get_embeddings=True, lengths=lengths)
            # generated_lenghts are the lengths of the sequences we have generated so far, so they are have the same shape as lengths, but initialized to 0
            generated_lenghts = torch.zeros_like(lengths) + 1
            
            for i in range(max_length):
                # we parameterise a cpd for Y[i]|X=x
                # note that the forward method takes care of not conditioning on y[i] itself
                # and only using the ngram_size-1 previous tags
                # at this point, the tag y[i] is a dummy code
                # the forward method recomputes all cds in the batch, this will include the cpd for Y[i]
                # [batch_size, max_len, C] 
                try:
                    output = self.model(x, y, eval=True, get_embeddings=False, get_predictions=True, u=embeddings, lengths=generated_lenghts)
                except Exception as e:
                    print(e)
                    print("i: ", i)
                    print("y shape: ", y.shape)
                    print("embeddings shape: ", embeddings.shape)
                    print("generated_lenghts shape: ", generated_lenghts.shape)
                    print("lengths shape: ", lengths.shape)
                    print("x shape: ", x.shape)
                    raise e
                output = self.model(x, y, eval=True, get_embeddings=False, get_predictions=True, u=embeddings, lengths=generated_lenghts)
                mask = torch.arange(max_length).expand(len(generated_lenghts), max_length) < lengths.unsqueeze(1)
                preds = self.model_crf.decode(output, mask.to(self.this_device))
                # for preds in preds_list, that is less than i+1, we need to pad them to i+1
                for pred_index, pred in enumerate(preds):
                    if len(pred) < max_length:
                        pred = pred + [self.pad] * (max_length - len(pred))
                        preds[pred_index] = pred
                # preds is list of lists of length batch_size, so we need to convert it to a tensor
                # [batch_size, max_len]
                preds = torch.tensor(preds, device=self.this_device)
                y[:, :i+1] = preds[:, :i+1]
                # predict.extend([np.array(seq[:l]) for l, seq in zip(lengths, preds)]) 
                generated_lenghts += 1
                for j in range(batch_size):
                    if generated_lenghts[j] > lengths[j]:
                        generated_lenghts[j] = lengths[j]
            return y
        
    def load_data(self):
        Feeder = utils.import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            dataset = Feeder(**self.arg.train_feeder_args)
            sums = [
                sum([float(elem.split("_")[2]) for elem in row])
                  for row in dataset.label
                ]
            zeros = len([elem for elem in sums if elem == 0])
            non_zeros = len(sums) - zeros
            weights = np.ones(len(sums))
            denumerator = len(sums)
            for i in range(len(sums)):
                numerator = non_zeros if sums[i] == 0 else zeros
                weights[i] = numerator / denumerator
            weighted_sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=2*non_zeros,
                replacement=False
            )
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=weighted_sampler,
                batch_size=self.arg.batch_size,
                # shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=utils.init_seed)
            self.targets = self.data_loader['train'].dataset.label
            self.targets = np.array(self.targets)
        
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=utils.init_seed)
        if self.arg.phase =='test':
            self.targets = self.data_loader['test'].dataset.label
            self.targets = np.array(self.targets)
        # print the ratio of positive to negative samples
        self.pos_weight = np.sum(self.targets == 'other') / len(self.targets)
        # make pose_weight a torch tensor
        self.pos_weight = torch.tensor(self.pos_weight)
        # self.labels_dict = {'outside_left': 0, 'starting': 1, 'early': 1, 'middle': 2, 'full': 2, 'outside_right': 0, 'ending': 3, 'late': 3}
        self.classes = self.arg.labeler_args['classes']
        self.num_classes = self.arg.labeler_args['num_classes']
        # for now make it 0, but it should be something outside the classes

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = utils.import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        if self.this_device.type == 'cpu':
            self.model = Model(**self.arg.model_args)
        else:
            self.model = Model(**self.arg.model_args).cuda(output_device)
        if self.arg.loss_function == 'BCEWithLogitsLoss':
            print('BCEWithLogitsLoss with pos_weight = {}'.format(self.pos_weight))
            if self.this_device.type == 'cpu':
                self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight).cuda(output_device)
        elif self.arg.loss_function == 'WeightedFocalLoss': 
            if self.this_device.type == 'cpu':
                self.loss = loss.WeightedFocalLoss(gamma=2, alpha=[0.1, 0.3, 0.4, 0.3])
            else:
                self.loss = loss.WeightedFocalLoss(gamma=2, alpha=[0.1, 0.3, 0.4, 0.3]).cuda(output_device)

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights.format(self.arg.train_feeder_args['fold'])))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                print('weights path is: ', self.arg.weights.format(self.arg.train_feeder_args['fold']))
                weights = torch.load(self.arg.weights.format(self.arg.train_feeder_args['fold']))
            if self.this_device.type == 'cpu':
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])
            else:
                pass
                weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
                # weights = OrderedDict([['.'.join(k.split('.')[1:]), v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))
            if self.arg.pretrained and self.arg.pretrained_model == 'SLR':
                if self.arg.loss_function == 'BCEWithLogitsLoss':
                    self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
                    if self.this_device.type == 'cpu':
                        self.model = utils.reinitialize_classification_layer(
                            self.model, num_classes=self.num_classes
                            )
                    else:
                        self.model = utils.reinitialize_classification_layer(
                            self.model, num_classes=self.num_classes
                            ).cuda(output_device)
                else:
                    if self.arg.labeler_args['classifier'] == 'binary':
                        self.loss = loss.WeightedFocalLoss(gamma=2, alpha=[0.5, 0.5])
                    else:
                        self.loss = loss.WeightedFocalLoss(gamma=2, alpha=[0.1, 0.3, 0.4, 0.3])
                    if self.this_device.type == 'cpu':
                        self.model = utils.reinitialize_classification_layer(
                            self.model, num_classes=self.num_classes
                            )
                    else:
                        self.model = utils.reinitialize_classification_layer(
                            self.model, num_classes=self.num_classes
                            ).cuda(output_device)
                    
        self.arg.labeler = self.arg.labeler.format(self.arg.labeler_args['labeler_name'])
        Labeler = utils.import_class(self.arg.labeler)
        self.arg.labeler_args['gcns_model'] = self.model
        shutil.copy2(inspect.getfile(Labeler), self.arg.work_dir)
        if self.this_device.type == 'cpu':
            self.model = Labeler(**self.arg.labeler_args)
        else:
            self.model = Labeler(**self.arg.labeler_args).cuda(output_device)
        if self.this_device.type == 'cpu':
            self.model_crf = CRF(num_tags=len(self.classes), batch_first=True)
        else:
            self.model_crf = CRF(num_tags=len(self.classes), batch_first=True).cuda(output_device)

        if self.arg.weights and "save_models" in self.arg.weights:
            try:
                weights = torch.load(self.arg.weights)
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        self.bos = self.model.bos
        # TODO: check this carefully
        # self.bos = 0
        if torch.cuda.is_available():
            self.model.cuda()

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]

            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir + '/eval_results')

        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = loader
        predict = []
        true = []
        MED_distances = []
        iou_scores = []
        if epoch >= self.arg.only_train_epoch:
            print('only train part, require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            print('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')
        for batch_idx, (data, label, lengths, index, labels_dict) in enumerate(process):
            self.global_step += 1
            data = Variable(data.float().to(self.this_device), requires_grad=False)
            label = Variable(label.long().to(self.this_device), requires_grad=False)
            timer['dataloader'] += self.split_time()
            if epoch < 100:
                keep_prob = -(1 - self.arg.keep_rate) / 100 * epoch + 1.0
            else:
                keep_prob = self.arg.keep_rate
            
            output = self.model(data, label, eval=False, lengths=lengths, keep_prob=keep_prob)   
            
            max_len = self.arg.labeler_args['max_seq_len']
            num_classes = self.arg.labeler_args['num_classes']
            mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

            # Apply the mask to the output tensor. We flatten the sequence dimension to apply the mask directly.
            output_flat = output.view(-1, num_classes)[mask.view(-1), :]
            labels_flat = label.view(-1)[mask.view(-1)]

            if self.arg.labeler_args['training_seq'] == 'crf':
                loss = - self.model_crf(output, label, mask.to(self.this_device), reduction='token_mean')
            else:
                loss = self.loss(output_flat, labels_flat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()            
            
            loss_value.append(loss.data.cpu().numpy())
            timer['model'] += self.split_time()
            if self.arg.labeler_args['labeler_name'] == 'Independent':
                if self.arg.labeler_args['training_seq'] == 'crf':
                    preds = self.model_crf.decode(output, mask.to(self.this_device))
                else:
                    preds = torch.argmax(output, dim=-1)
                # need to pad them to max_len
                for pred_index, pred in enumerate(preds):
                    if len(pred) < max_len:
                        pred = pred + [self.pad] * (max_len - len(pred))
                        preds[pred_index] = pred
                # preds is list of lists of length batch_size, so we need to convert it to a tensor
                # [batch_size, max_len]
                preds = torch.tensor(preds, device=self.this_device)
            elif self.arg.labeler_args['labeler_name'] == 'Markov' or self.arg.labeler_args['labeler_name'] == 'Autoregressive':
                if epoch+1 %  self.arg.viterabi_interval == 0:
                    preds = self.crf_auto_decoder(data, lengths) 
                else:
                    preds = self.markov_greedy(data, lengths) 
                  
            predict.extend([seq[:l] for l, seq in zip(lengths, preds.cpu().numpy())])  
            true.extend([seq[:l] for l, seq in zip(lengths, label.data.cpu().numpy())])

            for pred, target in zip(preds, label):
                pred_str = ''.join(map(str, pred.cpu().numpy()))
                target_str = ''.join(map(str, target.cpu().numpy()))
                distance = lev.distance(pred_str, target_str)
                MED_distances.append(distance)
            iou_scores.append(utils.iou_score(preds.cpu(), label.cpu()))


            self.lr = self.optimizer.param_groups[0]['lr']

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx, len(loader), loss.data, self.lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        if epoch % self.arg.save_interval == 0:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])
            
            torch.save(weights, self.arg.model_saved_name +
                    '-' + str(epoch) + '.pt')
            if self.arg.labeler_args['training_seq'] == 'crf':
                crf_model_path = self.arg.model_saved_name + '-' + str(epoch) + '-crf.pt'
                crf_weights = self.model_crf.state_dict()
                crf_weights = OrderedDict([[k.split('module.')[-1],
                                            v.cpu()] for k, v in crf_weights.items()])
                torch.save(crf_weights, crf_model_path)
                # crf_weights = torch.load(crf_model_path)
                # self.model_crf.load_state_dict(crf_weights)
        # calculate f1, precision, recall, and accuracy
        mean_MED = np.mean(MED_distances)
        mean_iou = np.mean(iou_scores)
        true = utils.flatten(true)
        predict = utils.flatten(predict)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, predict, average='macro')
        print('precision: {}, recall: {}, f1: {}'.format(
            precision, recall, f1))
        print('MED: {}, IoU: {}'.format(mean_MED, mean_iou))
        # calculate the auc
        accuracy = np.mean(np.equal(predict, true))
        print('accuracy: {}'.format(accuracy))
        self.writer.add_scalar("Loss/train", np.mean(loss_value), epoch)
        self.writer.add_scalar("Accuracy/train", accuracy, epoch)
        self.writer.add_scalar("Precision/train", precision, epoch)
        self.writer.add_scalar("Recall/train", recall, epoch)
        self.writer.add_scalar("F1/train", f1, epoch)
        self.writer.add_scalar("MED/train", mean_MED, epoch)
        self.writer.add_scalar("IOU/train", mean_iou, epoch)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, predict)
        print('class: {}'.format(self.classes))
        for i in range(len(precision)):
            self.writer.add_scalar("Precision/train_{}".format(self.classes[i]), precision[i], epoch)
            self.writer.add_scalar("Recall/train_{}".format(self.classes[i]), recall[i], epoch)
            self.writer.add_scalar("F1/train_{}".format(self.classes[i]), f1[i], epoch)
            print('class: {} --> precision: {}, recall: {}, f1: {}'.format(
                self.classes[i], precision[i], recall[i], f1[i]))
            
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None, results={}, output_dict={}):   
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        with torch.no_grad():
            self.print_log('Eval epoch: {}'.format(epoch + 1))
            for ln in loader_name:
                loss_value = []
                step = 0
                process = self.data_loader[ln]
                true = []
                predict = []
                MED_distances = [] 
                iou_scores = []
                for batch_idx, (data, label, lengths, index, labels_dict) in enumerate(process):
                   
                    data = Variable(data.float().to(self.this_device), requires_grad=False)
                    label = Variable(label.long().to(self.this_device), requires_grad=False)

                    with torch.no_grad():
                        output = self.model(data, label, eval=False, lengths=lengths)
                    max_len = self.arg.labeler_args['max_seq_len']
                    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
                    if self.arg.labeler_args['training_seq'] == 'crf': 
                        loss = -self.model_crf(output, label, mask.to(self.this_device), reduction='token_mean')
                    else:
                        loss = self.loss(output, label)
                    loss_value.append(loss.data.cpu().numpy())
                    if self.arg.labeler_args['labeler_name'] == 'Independent':
                        if self.arg.labeler_args['training_seq'] == 'crf':
                            preds = self.model_crf.decode(output, mask.to(self.this_device))
                        else:
                            preds = torch.argmax(output, dim=-1)
                        # need to pad them to max_len
                        for pred_index, pred in enumerate(preds):
                            if len(pred) < max_len:
                                pred = pred + [self.pad] * (max_len - len(pred))
                                preds[pred_index] = pred
                        # preds is list of lists of length batch_size, so we need to convert it to a tensor
                        # [batch_size, max_len]
                        preds = torch.tensor(preds, device=self.this_device)
                        
                    elif self.arg.labeler_args['labeler_name'] == 'Markov' or self.arg.labeler_args['labeler_name'] == 'Autoregressive':
                        if epoch+1 %  self.arg.viterabi_interval == 0:
                            preds = self.crf_auto_decoder(data, lengths) 
                        else:
                            preds = self.markov_greedy(data, lengths)     
                    predict.extend([seq[:l] for l, seq in zip(lengths, preds.cpu().numpy())])
                    true.extend([seq[:l] for l, seq in zip(lengths, label.data.cpu().numpy())])
                    step += 1
                    
                    for pred, target in zip(preds, label):
                        pred_str = ''.join(map(str, pred.cpu().numpy()))
                        target_str = ''.join(map(str, target.cpu().numpy()))
                        distance = lev.distance(pred_str, target_str)
                        MED_distances.append(distance)
                    iou_scores.append(utils.iou_score(preds.cpu(), label.cpu()))

                mean_MED = np.mean(MED_distances)
                mean_iou = np.mean(iou_scores)
                                        
                # # calculate f1, precision, recall
                true = utils.flatten(true)
                predict = utils.flatten(predict)
                macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                    true, predict, average='macro')
                print('precision: {}, recall: {}, f1: {}'.format(
                    macro_precision, macro_recall, macro_f1))
                print('MED: {}, IOU: {}'.format(mean_MED, mean_iou))
                # calculate accuracy
                accuracy = np.mean(np.equal(true, predict))
                print('accuracy: {}'.format(accuracy))
                print('Eval Accuracy: ', accuracy,
                    ' model: ', self.arg.model_saved_name)
        if self.arg.phase == 'train':     
            self.writer.add_scalar("Loss/test", np.mean(loss_value), epoch)
            self.writer.add_scalar("Accuracy/test", accuracy, epoch)
            self.writer.add_scalar("Precision/test", macro_precision, epoch)
            self.writer.add_scalar("Recall/test", macro_recall, epoch)
            self.writer.add_scalar("F1/test", macro_f1, epoch)
            self.writer.add_scalar("MED/test", mean_MED, epoch)
            self.writer.add_scalar("IOU/test", mean_iou, epoch)
        results['macro_avg']['precision'].append(macro_precision)
        results['macro_avg']['recall'].append(macro_recall)
        results['macro_avg']['f1'].append(macro_f1)
        # fill all_predict with random 0 or 1 values 

        precision, recall, f1, _ = precision_recall_fscore_support(
            true, predict)
        results_dict = {
            'accuracy': accuracy, 
            'macro_precision': macro_precision, 
            'macro_recall': macro_recall, 
            'macro_f1': macro_f1, 
            'mean_MED': mean_MED, 
            'mean_iou': mean_iou
            }
        for i in range(len(precision)):
            if self.arg.phase == 'train': 
                self.writer.add_scalar("Precision/test_{}".format(self.classes[i]), precision[i], epoch)
                self.writer.add_scalar("Recall/test_{}".format(self.classes[i]), recall[i], epoch)
                self.writer.add_scalar("F1/test_{}".format(self.classes[i]), f1[i], epoch)
            print('class: {} --> precision: {}, recall: {}, f1: {}'.format(
                self.classes[i], precision[i], recall[i], f1[i]))
            results[self.classes[i]]['precision'].append(precision[i])
            results[self.classes[i]]['recall'].append(recall[i])
            results[self.classes[i]]['f1'].append(f1[i])
            results_dict[self.classes[i]] = {'precision': precision[i], 'recall': recall[i], 'f1': f1[i]}
            # save scores 
        if epoch == self.arg.num_epoch - 1:
            results_dict['true'] = true
            results_dict['predict'] = predict
        results_dict['true'] = true
        results_dict['predict'] = predict
        filename = (
            self.arg.log_base_path + 
            f"/work_dir/{self.arg.Experiment_name}"
            f"/eval_results/epoch_{epoch}_{self.arg.train_feeder_args['fold']}"
            f"_{macro_f1}_{macro_precision}"
            f"_{macro_recall}_test_wo_crf_decoding.pkl"
        )
        with open(filename,'wb') as f:
            pickle.dump(results_dict, f)
        return results, output_dict
    
    def start(self, results, output_dict):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * \
                len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                results, output_dict = self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'], results=results, output_dict=output_dict)
   
            print('best f1: ', self.best_f1,
                  ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            results, output_dict = self.eval(epoch=self.arg.start_epoch, save_score=self.arg.save_score,
                      loader_name=['test'], wrong_file=wf, result_file=rf, results=results, output_dict=output_dict)
            self.print_log('Done.\n')
        return results, output_dict