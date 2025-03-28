import os
import platform
from .config import training_config, network_config, env_config

system = platform.system()
root = './environments/NBTTT'
mapping = {'Windows': 'dll',
           'Darwin': 'dylib',
           'Linux': 'so'}


def compile_file(source_file, output_file):
    os.system(f'gcc {source_file} -o {output_file} -c -O3')


def cleanup():
    os.system(f'rm {root}/*.o' if platform.system()
              != 'Windows' else f'del {root}/*.o')


if f'NBTTTAgent.{mapping[system]}' not in os.listdir(root):
    compile_file(f'{root}/dependency/utils.c', f'{root}/utils.o')
    compile_file(f'{root}/dependency/heuristic.c', f'{root}/heuristic.o')
    compile_file(f'{root}/dependency/Negamax.c', f'{root}/Negamax.o')

    os.system(f"gcc -shared -O3 -o {root}/NBTTTAgent.{mapping[system]} "
              f"{root}/Negamax.o {root}/heuristic.o {root}/utils.o")

    cleanup()
    print("Compile complete!")

from .utils import instant_augment, inspect
from .env import Env
from .Network import CNN
