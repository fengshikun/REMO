import re
import os
import pandas as pd
import numpy as np
import argparse

def log_check(name, data_dict, model_name, seed, best_res='always'):
    with open(name, "r") as f:
        content = f.read()
    
    if best_res == 'depends':
        tgt = r"====epoch 300\n====Evaluation\ntrain: ([0-9]+\.[0-9]+) val: ([0-9]+\.[0-9]+) test: ([0-9]+\.[0-9]+) test cliff: ([0-9]+\.[0-9]+)\nThe best epoch for (\w+): (\d+)\ntrain: (\d+\.\d+) val: (\d+\.\d+) test: (\d+\.\d+) test cliff: (\d+\.\d+)"
        match = re.findall(tgt, content)
        if match:
            for result in match:
                if int(result[5]) > 150:
                    target = result[4]
                    test_rmse = result[8]
                    test_rmse_cliff = result[9]
                    data_dict['model'].append(model_name)
                    data_dict['seed'].append(seed)
                    data_dict['target'].append(target)
                    data_dict['test_rmse'].append(test_rmse)
                    data_dict['test_cliff_rmse'].append(test_rmse_cliff)

                else:
                    target = result[4]
                    test_rmse = result[2]
                    test_rmse_cliff = result[3]
                    data_dict['model'].append(model_name)
                    data_dict['seed'].append(seed)
                    data_dict['target'].append(target)
                    data_dict['test_rmse'].append(test_rmse)
                    data_dict['test_cliff_rmse'].append(test_rmse_cliff)
        
    elif best_res == 'always':    
        tgt = r"The best epoch for (\w+): \d+\ntrain: (\d+\.\d+) val: (\d+\.\d+) test: (\d+\.\d+) test cliff: (\d+\.\d+)"
        match = re.findall(tgt, content)
        if match:
            for result in match:
                print(result)
                target = result[0]
                test_rmse = result[3]
                test_rmse_cliff = result[4]
                data_dict['model'].append(model_name)
                data_dict['seed'].append(seed)
                data_dict['target'].append(target)
                data_dict['test_rmse'].append(test_rmse)
                data_dict['test_cliff_rmse'].append(test_rmse_cliff)
    else: #best_res == 'never'
        tgt = r"====epoch 300\n====Evaluation\ntrain: ([0-9]+\.[0-9]+) val: ([0-9]+\.[0-9]+) test: ([0-9]+\.[0-9]+) test cliff: ([0-9]+\.[0-9]+)\nThe best epoch for (\w+):"
        match = re.findall(tgt, content)
        if match:
            for result in match:
                print(result)
                target = result[4]
                test_rmse = result[2]
                test_rmse_cliff = result[3]
                data_dict['model'].append(model_name)
                data_dict['seed'].append(seed)
                data_dict['target'].append(target)
                data_dict['test_rmse'].append(test_rmse)
                data_dict['test_cliff_rmse'].append(test_rmse_cliff)
        

    
   
    return data_dict


def main():
    parser = argparse.ArgumentParser(description='Command for result analysis')
    parser.add_argument('--log_dir', type=str, default="/share/project/Chemical_Reaction_Pretraining/finetune/logs", help="Directory of output log files")
    parser.add_argument('--best_res', type=str, default="depend", help="'always' for the best validation result, 'never' for results of the last epoch, 'depends' for the case to take the best res of the result after 100 epochs only, otherwise the last epoch")
    parser.add_argument('--output', type=str, default="/share/project/Chemical_Reaction_Pretraining/scripts/activity_cliff_result.csv", help="Directory for the output csv")
    args = parser.parse_args()
    
    log_dir = args.log_dir
    log_list= os.listdir(log_dir)
    data_dict = {"model":[],
                 "seed": [],
                 "target":[],
                 "test_rmse":[],
                 "test_cliff_rmse": []}
    for i in log_list:
        name_tokens = i.split('.')[0].split('_')
        model_name = '_'.join(name_tokens[3:-1])
        seed_num = name_tokens[-1][-1]
        data_dict = log_check(os.path.join(log_dir, i), data_dict, model_name, seed_num, args.best_res)
    # print(data_dict)
    tab_res = pd.DataFrame(data=data_dict)
    tab_res.to_csv(args.output, index=False)
if __name__ == "__main__":
    main()