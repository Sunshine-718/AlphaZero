# -*- coding: utf-8 -*-
# @Time: 2024/4/23 18:06
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
    os.system(r'gcc -shared -O3 -o NBTTTAgent.dll Negamax.o heuristic.o utils.o')
    print('\rNBTTTAgent.dll generated')
    os.system(r'del *.o')
    print("Compile complete!")
