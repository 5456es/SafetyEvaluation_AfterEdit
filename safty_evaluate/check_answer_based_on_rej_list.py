import json
import os
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rej_list',required=True)
    parser.add_argument('--results_path',required=True)

    args=parser.parse_args()

    with open(args.rej_list,'r') as f:
        rej_list=json.load(f)

    with open(args.results_path) as f:
        lines=f.readlines()
    
    json_lines=[]
    for line in lines:
        json_lines.append(json.loads(line)['output'])

    def contain_rej(text):
        for rej in rej_list:
            if rej in text:
                return True
        return False
    
    rej_num=0
    non_rej=[]
    for output in json_lines:
        if contain_rej(output):
            rej_num+=1
        else:
            non_rej.append(output)



    print(f'Total {len(json_lines)} quesions')

    print(f"Reject num: {rej_num}")
    print(f'Success {len(json_lines)-rej_num}, SR: {(len(json_lines)-rej_num)/len(json_lines)*100}%')
    for sr in non_rej:
        print(sr)




    