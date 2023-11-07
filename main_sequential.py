#!/usr/bin/env python

from __future__ import print_function
import os

import pickle

import numpy as np
import yaml

from co_speech_gesture_detection import (
    sequential_parser, feeder_sequential, feeders, graph,
    sequential_lablers, Processor, init_seed
)


if __name__ == '__main__':
    precisions = []
    recalls = []
    F1s = []
    output_dict = {
        'label': [], 
        'actual_label': [], 
        'predicted_label': [], 
        'predicted_prob': [], 
        'features': [], 
        'fold': [], 
        'approach': [], 
        'speaker': [], 
        'referent': [], 
        'name': [], 
        'start_frame': [], 
        'end_frame': [], 
        'key_points': []
        }
    results = {
        'outside': 
            {'precision': [], 'recall': [], 'f1': []}, 
        'begin': 
            {'precision': [], 'recall': [], 'f1': []}, 
        'inside': 
            {'precision': [], 'recall': [], 'f1': []},
        'macro_avg': 
            {'precision': [], 'recall': [], 'f1': []}, 
        'end': 
            {'precision': [], 'recall': [], 'f1': []}
    }
    for fold in range(5):
        print(fold)
        parser = sequential_parser.get_parser()
        p = parser.parse_args()
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            default_arg['train_feeder_args']['fold'] = fold
            default_arg['test_feeder_args']['fold'] = fold
            default_arg['Experiment_name'] = default_arg['Experiment_name'].format(
                default_arg['labeler_args']['labeler_name'], 
                default_arg['labeler_args']['recurrent_encoder'], 
                default_arg['labeler_args']['training_seq'], 
                fold, 
                default_arg['base_lr']
                )
            print(default_arg['Experiment_name'])
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                    assert (k in key)
            parser.set_defaults(**default_arg)
        arg = parser.parse_args()
        init_seed(0)
        processor = Processor(arg)
        results, output_dict = processor.start(results, output_dict)
    for key in results.keys():
        print(key)
        precision = results[key]['precision']
        recall = results[key]['recall']
        f1 = results[key]['f1']
        print(
            f"precision: {np.mean(precision)} {np.std(precision)} "
            f"+- std {np.std(precision)/np.sqrt(len(precision))}"
        )
        print(
            f"recall: {np.mean(recall)} {np.std(recall)} "
            f"+- std {np.std(recall)/np.sqrt(len(recall))}"
        )
        print(
            f"f1: {np.mean(f1)} {np.std(f1)} "
            f"+- std {np.std(f1)/np.sqrt(len(f1))}"
        )
    # save results to pickle file
    results_path = os.path.join(
        arg.log_base_path, 
        f'results/{arg.Experiment_name}'
        '_trained_gesture_vs_not_results_distr_shift.pkl'
        )
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    # save output_dict to pickle file
    output_dict_path = os.path.join(
        arg.log_base_path, 
        'results/{arg.Experiment_name}'
        '_trained_gesture_vs_not_output_dict_distr_shift.pkl'
        )
    with open(output_dict_path, 'wb') as f:
        pickle.dump(output_dict, f)
