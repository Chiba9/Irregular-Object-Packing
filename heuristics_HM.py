# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:34:07 2022

@author: Chiba
"""

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import os
import volume

class HM_heuristic(object):
    def __init__(object, c=0.01):
        self.c = c
        
    def pack(env, item):
        Hc = box_hm()
        if item in env.unpacked:
            print("iten {} already packed!".format(item))
            return False
        pitch, roll = np.meshgrid(np.arange(0,2*np.pi,np.pi/2),np.arange(0,2*np.pi,np.pi/2))
        pitch_rolls = np.array([pitch.reshape(-1), roll.reshape(-1)]).T
        Trans = []
        BoxW, BoxH = env.resolution, env.resolution
        for pitch_roll in pitch_rolls:
            transforms = np.concatenate((np.repeat([pitch_roll],4,axis=0).T,[np.arange(0,2*np.pi,np.pi/2)]),axis=0).T
            for trans in transforms:       
                Ht, Hb = env.item_hm(item, trans)
                w,h = Ht.shape
                for X in range(0, BoxW-w+1):
                    for Y in range(0, BoxH-h+1):
                        Z = np.max(Hc[X:X+w, Y:Y+h]-Hb)
                        Update = np.maximum((Ht>0)*(Ht+Z), Hc[X:X+w,Y:Y+h])
                        if np.max(Update) <= env.box_size[2]:
                            score = c*(X+Y)+np.sum(Hc)+np.sum(Update)-np.sum(Hc[X:X+w,Y:Y+h])
                            Trans.append(np.array(list(trans)+[X,Y,Z,score]))
        if len(Trans)!=0:
        Trans = Trans[np.argsort(Trans[:,6])]
        trans = Trans[0]
        if type(trans)!=type(None):
            print("Pos:%d,%d,%f" % (trans[3],trans[4],trans[5]))
            S = env.pack_item(item_id, trans)