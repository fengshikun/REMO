import re
import os
import pandas as pd
import numpy as np
import argparse

def log_check(name, best_res=True):
    with open(name, 'r') as f:
        content = f.read()

    match = re.findall(r'test: (\d+\.\d+)', content)
    if match:
        if best_res:
            test_result = float(match[-1])
        else:
            test_result = float(match[-2])
        return test_result
    else:
        return None
    
def main():
    parser = argparse.ArgumentParser(description='Command for result analysis')
    parser.add_argument('--log_dir', type=str, default="/share/project/Chemical_Reaction_Pretraining/finetune/logs", help="Directory of output log files")
    parser.add_argument('--best_res', type=int, default=1, help="1 for the best validation result, 0 for results of the last epoch")
    parser.add_argument('--output', type=str, default="/share/project/Chemical_Reaction_Pretraining/scripts/result_analysis.csv", help="Directory for the output csv")
    args = parser.parse_args()

    log_dir = args.log_dir
    log_list = os.listdir(log_dir)
    log_name = []
    for i in log_list:
        log_name.append('_'.join(i.split('.')[0].split('_')[:-2]))
    log_name = set(log_name)
    log_result = {}
    dataset_list = ['bbbp', 'tox21', 'muv', 'bace', 'toxcast', 'sider', 'clintox', 'hiv']
    seed = ['seed0', 'seed1', 'seed2']
    
    for model_file in log_name:
        log_result[model_file] = []
        for dataset in dataset_list:
            seed_sum = []
            for seed_num in seed:
                current_log = model_file + '_' + dataset + '_' + seed_num + '.log'
                current_log = os.path.join(log_dir, current_log)
                if not os.path.isfile(current_log):
                    break
                    
                test_result = log_check(current_log, args.best_res)
                seed_sum.append(test_result)
            if not os.path.isfile(current_log):
                break
            res = str(round(np.mean(seed_sum) * 100, 2)) + f'({round(np.std(seed_sum), 4)})'
            log_result[model_file].append(res)
        
    tabular_result = pd.DataFrame.from_dict(log_result, orient='index', columns=['BBBP', 'Tox21', 'MUV', 'BACE', 'ToxCast', 'SIDER', 'ClinTox', 'HIV'])
    
    tabular_result.to_csv(args.output)
            


if __name__ == "__main__":
    main()
