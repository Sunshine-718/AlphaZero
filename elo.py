#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 06/Aug/2024  04:45


class Elo:
    def __init__(self, init_A=1500, init_B=1500):
        self.R_A = init_A
        self.R_B = init_B
    
    def update(self, result_a, k=32):
        """
        :param result_a: 选手A的比赛结果(1胜，0.5平，0负)
        :param k: K系数，影响评分变化的幅度，默认值32 
        """
        expected_a = 1 / (1 + pow(10, (self.R_B - self.R_A) / 400))
        expected_b = 1 / (1 + pow(10, (self.R_A - self.R_B) / 400))
        self.R_A += k * (result_a - expected_a)
        self.R_B += k * ((1 - result_a) - expected_b)
        return self.R_A, self.R_B
        
if __name__ == '__main__':
    pass
