# coding:utf-8

import os, sys
import subprocess

def main():
    cnn_model = ['prot3','wrn_net','resnet', 'prot2']
    opt = ['adaboud','sgd','rmsprop']
    z_score = ['True', 'False']
    batchsize = ['256', '128', '64', '512']

    for z in z_score:
        for o in opt:
            for b in batchsize:
                for m in cnn_model:
                    cmd = ["python3", 'train.py', "-m", m, "-b", b, "-o", o, "-z", z]
                    print(cmd)
                    result = subprocess.check_output(cmd)
                    print(result)

if __name__ == "__main__":
    main()