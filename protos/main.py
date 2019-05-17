# coding:utf-8

import os, sys
import subprocess

def main():
    cnn_model = ['prot3','wrn_net']
    opt = ['rmsgraves', 'rmsprop']
    epoch = ['300']

    for o in opt:
        for m in cnn_model:
            cmd = ["python3", 'train.py', "-m", m, "-o", o]
            print(cmd)
            result = subprocess.check_output(cmd)
            print(result)

if __name__ == "__main__":
    main()