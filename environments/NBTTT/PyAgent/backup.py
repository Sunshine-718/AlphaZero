# -*- coding: utf-8 -*-
# @Time: 2024/3/12 18:48
import os
import shutil
import datetime

if __name__ == '__main__':
    files = os.listdir('./')
    folder = f"./backup/{str(datetime.datetime.now()).split('.')[0]}".replace(':', '')
    os.mkdir(folder)
    for i in files:
        if '.' in i and i != '.idea' and i != '.vscode':
            shutil.copy(f'./{i}', f'{folder}/{i}')
