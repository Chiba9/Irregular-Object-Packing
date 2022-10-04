# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:53:21 2021

@author: Chiba
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pybullet as p
import time
import pybullet_data
import os

class Env(object):
    def __init__(self, obj_dir, is_GUI=True, box_size=(0.4,0.4,0.3), resolution = 40):
        self.obj_dir = obj_dir
        self.obj_info = pd.read_csv(obj_dir + 'objects.csv')
        self.box_size = box_size
        self.resolution = resolution
        wall_width = 0.1
        boxL, boxW, boxH = box_size
        
        if p.isConnected():
            p.disconnect()
            
        if is_GUI:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        p.setGravity(0,0,-10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf')
        self.create_box([boxL,wall_width,boxH],[boxL/2,-wall_width/2,boxH/2])
        self.create_box([boxL,wall_width,boxH],[boxL/2,boxW+wall_width/2,boxH/2])
        self.create_box([wall_width,boxW,boxH],[-wall_width/2,boxW/2,boxH/2])
        self.create_box([wall_width,boxW,boxH],[boxL+wall_width/2,boxW/2,boxH/2])
        
        self.loaded_ids = []
        
        
    def create_box(self, size, pos):
        size = np.array(size)
        shift = [0, 0, 0]
        color = [1,1,1,1]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        rgbaColor=color,
                                        visualFramePosition=shift,
                                        halfExtents = size/2)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                              collisionFramePosition=shift,
                                              halfExtents = size/2)
        p.createMultiBody(baseMass=100,
                          baseInertialFramePosition=[0, 0, 0],
                          baseCollisionShapeIndex=collisionShapeId,
                          baseVisualShapeIndex=visualShapeId,
                          basePosition=pos,
                          useMaximalCoordinates=True)
        
    def load_items(self, item_ids):
        flags = p.URDF_USE_INERTIA_FROM_FILE
        for count in range(len(item_ids)):
            file_dir = self.object_info['dir'][item_ids[count]]
            loaded_id = p.loadURDF(self.obj_dir + file_dir, 
                                 [(count//5)/4+2.2, (count%5)/4+0.2, 0.1], flags=flags)
            self.loaded_ids.append(loaded_id)
            
    def remove_all_items(self):
        for loaded in self.loaded_ids:
            p.removeBody(loaded)
        self.loaded_ids = []
        
    def box_heightmap(self):
        sep = self.box_size[0]/self.resolution
        xpos = np.arange(sep/2,self.box_size[0]+sep/2,sep)
        ypos = np.arange(sep/2,self.box_size[1]+sep/2,sep)
        xscan, yscan = np.meshgrid(xpos,ypos)
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Start = np.insert(ScanArray,2,self.box_size[2],0).T
        End = np.insert(ScanArray,2,0,0).T
        RayScan = np.array(p.rayTestBatch(Start, End))
        Height = (1-RayScan[:,2].astype('float64'))*self.box_size[2]
        HeightMap = Height.reshape(self.resolution,self.resolution).T
        return HeightMap  
    
    def item_hm(self, item_id ,orient):
        old_pos, old_quater = p.getBasePositionAndOrientation(item_id)
        quater = p.getQuaternionFromEuler(orient)
        p.resetBasePositionAndOrientation(item_id,[1,1,1],quater)
        AABB = p.getAABB(item_id)
        sep = self.box_size[0]/self.resolution
        xpos = np.arange(AABB[0][0]+sep/2,AABB[1][0],sep)
        ypos = np.arange(AABB[0][1]+sep/2,AABB[1][1],sep)
        xscan, yscan = np.meshgrid(xpos,ypos)
        ScanArray = np.array([xscan.reshape(-1),yscan.reshape(-1)])
        Top = np.insert(ScanArray,2,AABB[1][2],axis=0).T
        Down = np.insert(ScanArray,2,AABB[0][2],axis=0).T
        RayScanTD = np.array(p.rayTestBatch(Top, Down))
        RayScanDT = np.array(p.rayTestBatch(Down, Top))
        Ht = (1-RayScanTD[:,2])*(AABB[1][2]-AABB[0][2])
        RayScanDT = RayScanDT[:,2]
        RayScanDT[RayScanDT==1] = np.inf
        Hb = RayScanDT*(AABB[1][2]-AABB[0][2])
        Ht = Ht.astype('float64').reshape(len(ypos),len(xpos)).T
        Hb = Hb.astype('float64').reshape(len(ypos),len(xpos)).T
        p.resetBasePositionAndOrientation(item_id,old_pos,old_quater)
        return Ht,Hb
    
    def bbox_order(self):
        volume = []
        for item in self.loaded_ids:
            AABB = np.array(p.getAABB(item))
            volume.append(np.product(AABB[1]-AABB[0]))
        bbox_order = np.argsort(volume)[::-1]
        return bbox_order
    
    def pack_item(self, item_id, transform):
        z_shift = 0.005
        
        target_euler = transform[0:3]
        target_pos = transform[3:6]
        target_pos[0] = target_pos[0]/self.resolution*self.box_size[0]
        target_pos[1] = target_pos[1]/self.resolution*self.box_size[1]
        pos, quater = p.getBasePositionAndOrientation(item_id)
        new_quater = p.getQuaternionFromEuler(target_euler)
        p.resetBasePositionAndOrientation(item_id, pos, new_quater)
        AABB = p.getAABB(item_id)
        shift = np.array(pos)-(np.array(AABB[0])+np.array([self.box_size[0]/2/self.resolution,
                         self.box_size[1]/2/self.resolution,z_shift]))
        new_pos = target_pos+shift
        p.resetBasePositionAndOrientation(item_id, new_pos, new_quater)
        for i in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        curr_pos, curr_quater = p.getBasePositionAndOrientation(item_id)
        curr_euler = p.getEulerFromQuaternion(curr_quater)
        stability = np.linalg.norm(new_pos-curr_pos)<0.02 and curr_euler.dot(target_euler)/(np.linalg.norm(curr_euler)*np.linalg.norm(target_euler)) > np.pi*2/3
        return stability
            
    def drawAABB(self, aabb, width=1):
        aabbMin = aabb[0]
        aabbMax = aabb[1]
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 0, 0], width)
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [0, 1, 0], width)
        f = [aabbMin[0], aabbMin[1], aabbMin[2]]
        t = [aabbMin[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [0, 0, 1], width)
        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMin[0], aabbMin[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMax[0], aabbMin[1], aabbMin[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMax[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMin[0], aabbMax[1], aabbMin[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMin[0], aabbMax[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMin[1], aabbMax[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        f = [aabbMax[0], aabbMax[1], aabbMax[2]]
        t = [aabbMax[0], aabbMax[1], aabbMin[2]]
        p.addUserDebugLine(f, t, [1, 1, 1], width)
        
    def draw_box(self, width=5):
        xmax = self.box_size[0]
        ymax = self.box_size[1]
        p.addUserDebugLine([0,0,0],[0,0,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,ymax,0],[0,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([xmax,0,0],[xmax,0,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([xmax,ymax,0],[xmax,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,0,0],[xmax,0,0], [1, 1, 1], width)
        p.addUserDebugLine([0,ymax,0],[xmax,ymax,0], [1, 1, 1], width)
        p.addUserDebugLine([0,0,0],[0,ymax,0], [1, 1, 1], width)
        p.addUserDebugLine([xmax,0,0],[xmax,ymax,0], [1, 1, 1], width)
        p.addUserDebugLine([0,0,self.box_size[2]],[xmax,0,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,ymax,self.box_size[2]],[xmax,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([0,0,self.box_size[2]],[0,ymax,self.box_size[2]], [1, 1, 1], width)
        p.addUserDebugLine([xmax,0,self.box_size[2]],[xmax,ymax,self.box_size[2]], [1, 1, 1], width)