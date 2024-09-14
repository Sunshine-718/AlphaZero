#! usr/local/bin/python3.12
# -*- coding: utf-8 -*-
# @Author: Sunshine
# @Time: 2024/5/13 上午5:56
import os

if __name__ == '__main__':
    print('Compling utils.c...', end='')
    os.system(r'gcc ./dependency/utils.c -o utils.o -c -O3')
    print('\rutils.c compiled')
    print('Compling heuristic.c...', end='')
    os.system(r'gcc ./dependency/heuristic.c -o heuristic.o -c -O3')
    print('\rheuristic.c compiled')
    print('Compling Negamax.c...', end='')
    os.system(r'gcc ./dependency/Negamax.c -o Negamax.o -c -O3')
    print('\rNegamax.c compiled')
    print('Generating NBTTTAgent.dll...', end='')
    os.system(r'gcc -dynamiclib -O3 -o NBTTTAgent.dylib Negamax.o heuristic.o utils.o')
    print('\rNBTTTAgent.dll generated')
    os.system(r'rm *.o')
    print("Compile complete!")
