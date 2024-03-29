# -*- coding: utf-8 -*-
"""Plot Results.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zclQ_nf9SbLGfqZMEN00Wf2wNzRwFNWY

## Import Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys, traceback

"""### Global Configs"""

"""### A simple object to parse one log file"""

class Experiment:
    
    def __init__(self, experiment, results_folder):
        """
            Takes in an experiment string like:
                experiment = 'exp_6__1_10_10000__1e-06__20200706192647.csv'
            And builds an object with the experiment configuration. 
        """
        experiment_csv = f"{results_folder}/{experiment}"
        assert os.path.exists(experiment_csv)

        experiment_config = experiment.split('__')

        # Make sure the experiment configuration is as expected
        assert len(experiment_config) == 4
        experiment_number, game_parameters, learning_rate, timestamp = experiment_config
        game_parameters = game_parameters.split('_')

        # Make sure the game parameters are as expected
        assert len(game_parameters) == 3
        game_length, num_trajs, num_epo = game_parameters

        # Read the first line of the results file to get the name of the policies of the two agents. 
        with open(experiment_csv, 'r') as file:
            policies_names = file.readline().strip().split(',')
            # Assume that there are only two agents
            assert len(policies_names) == 2
            agent_1_policy, agent_2_policy = policies_names

        # Collect all the experiment configuration into one dictionary
        self.experiment_config = {
            'results_folder' : results_folder,
            'experiment' : experiment,
            'experiment_csv' : experiment_csv,
            'experiment_number' : experiment_number,
            'game_length' : int(game_length),
            'num_trajs' : int(num_trajs),
            'num_epo' : int(num_epo),
            'learning_rate' : float(learning_rate),                 
            'timestamp' : timestamp,                     
            'agent_1_policy' : agent_1_policy,
            'agent_2_policy' : agent_2_policy}
    
    def get_experiment_data(self):
        """ Read the .csv file"""
        if self.experiment_config is not None:
            if 'data' not in self.experiment_config:    
                self.experiment_config['data'] = pd.read_csv(self.experiment_config['experiment_csv'], 
                                                             skiprows=1, 
                                                             header=None, 
                                                             names=['avg_agent_1','std_agent_1', 
                                                                    'avg_agent_2','std_agent_2'])
        return self.experiment_config['data']
    
    def __repr__(self):
        return f"{self.experiment_config['experiment']}"
    
    @staticmethod
    def partition_experiments(comparator, results_folder):
        """
            Partition the experiments folder into equivalence classes. 
        """
        list_of_expts = []
        items = os.listdir(results_folder)
        for name in items:
            if name.endswith(".csv"):
                list_of_expts.append(Experiment(name, results_folder))

        partition = []
        while len(list_of_expts) > 0:
            current_exp = list_of_expts[0]
            partition += [list(filter(lambda e, cur=current_exp: comparator(e, cur), list_of_expts))]
            list_of_expts = list(filter(lambda e, cur=current_exp: not comparator(e, cur), list_of_expts))
        return partition        

    @staticmethod
    def equal_game_params(e1, e2):
        return e1.experiment_config['game_length'] == e2.experiment_config['game_length'] and \
        e1.experiment_config['num_trajs'] == e2.experiment_config['num_trajs'] and \
        e1.experiment_config['num_epo'] == e2.experiment_config['num_epo']
    
    @staticmethod
    def equal_method_1(e1, e2):
        """
        For all files with the same <game params> and <learning rate>, 
        graph agent1 on the same graph (i.e. fix the game and learning rate, then compare different algorithms).
        """
        return Experiment.equal_game_params(e1, e2) and \
        e1.experiment_config['learning_rate'] == e2.experiment_config['learning_rate']

    @staticmethod
    def equal_method_2(e1, e2):
        """
        For all files with the same <experiment> and <game params>,  
        graph agent1 on the same graph (i.e. fix the game and algo, then compare learning rates).        """
        return Experiment.equal_game_params(e1, e2) and \
        e1.experiment_config['experiment_number'] == e2.experiment_config['experiment_number']
    
    @staticmethod
    def plot_one_elem_of_partition(output_folder, plot_var, 
                                    x_range_min, x_range_max, 
                                    y_range_min, y_range_max,
                                    elem, title_set, filename_prefix):
        for exp in elem:
            if '_Uniform'.lower() in exp.experiment_config['agent_1_policy'].lower():
                print("Skipping {} because it contains a uniform distr. policy.".format(exp.experiment_config['experiment_csv']))
                continue
            x = [i for i in range(0, exp.experiment_config['num_epo'])]
            y = exp.get_experiment_data()['avg_agent_1']
            e = exp.get_experiment_data()['std_agent_1']
            # Safety check: if data ends with FAILED, then specify that the policy failed.
            # Perhaps we should do something better?
            if (type(y[len(y)-1]) is str) and (y[len(y)-1].lower() == 'failed'):
                y = y[:-1]
                e = e[:-1]
                y = [np.clip(float(item), -500, 500) for item in y]
                e = [np.clip(float(item), -500, 500) for item in e]
                exp.experiment_config['agent_1_policy'] = exp.experiment_config['agent_1_policy'] + " (FAILED)"
            # A safe fail here: if the data is incomplete, graph what is available.
            # We could do something better here!
            if len(y) != exp.experiment_config['num_epo']:
                x = x[:len(y)]
            
            # x = x[:1000]
            # y = y[:1000]
            # e = e[:1000]
            try:
                if not plot_var:
                    e = None
                plt.errorbar(x, y, e, linestyle='None', marker='.',
                            label=exp.experiment_config['agent_1_policy'])

            except Exception as e:
                print("Failed to plot {}".format(exp.experiment_config['experiment_csv']))
                raise e

        e = elem[0]
        plt.title(', '.join([f"{x} : {e.experiment_config[x]} " for x in title_set]))
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('cumulative reward avg', fontsize=14)
        plt.ylim([-25, 25])
        plt.legend(loc='lower right')
        try:
            os.mkdir(output_folder)
        except:
            pass
        plt.savefig(output_folder + "/" + filename_prefix + '_' + '_'.join([f"{x}_{e.experiment_config[x]}" for x in title_set]) + '.png', bbox_inches='tight', dpi=200)
        plt.show()

def method_1(results_folder, output_folder, plot_var, 
             x_range_min, x_range_max, y_range_min, y_range_max):
    partition = Experiment.partition_experiments(Experiment.equal_method_1, results_folder)
    for elem in partition:
        Experiment.plot_one_elem_of_partition(output_folder, plot_var, 
                                                x_range_min, x_range_max, 
                                                y_range_min, y_range_max,
                                                elem, 
                                                ['game_length', 'num_trajs', 'num_epo', 'learning_rate'], 
                                                'algorithms_comparison')

def method_2(results_folder, output_folder, plot_var,
             x_range_min, x_range_max, y_range_min, y_range_max):
    partition = Experiment.partition_experiments(Experiment.equal_method_2, results_folder)
    for elem in partition:
        Experiment.plot_one_elem_of_partition(output_folder, plot_var, 
                                                x_range_min, x_range_max, 
                                                y_range_min, y_range_max,
                                                elem, 
                                                ['game_length', 'num_trajs', 'num_epo', 'learning_rate', 'experiment_number'], 
                                                'learning_rates_comparison')


if __name__ == '__main__':
    method = sys.argv[1]
    results_folder = sys.argv[2]
    output_folder = sys.argv[3]
    try:
        plot_var_str = sys.argv[4].strip().lower()
        if (plot_var_str == 'f') or (plot_var_str == 'false'):
            plot_var = False
        else:
            plot_var = True
    except:
        plot_var = True
    try:
        x_range_min = sys.argv[5]
    except:
        x_range_min = None
    try:
        x_range_max = sys.argv[6]
    except:
        x_range_max = None
    try:
        y_range_min = sys.argv[7]
    except:
        y_range_min = None
    try:
        y_range_max = sys.argv[8]
    except:
        y_range_max = None
    function_mappings = {
        'method_1': method_1,
        'method_2': method_2
    }
    try:
        func = function_mappings[method]
    except KeyError:
        raise ValueError('invalid input')
    func(results_folder, output_folder, plot_var, 
         x_range_min, x_range_max, y_range_min, y_range_max)