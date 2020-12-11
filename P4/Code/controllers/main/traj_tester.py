#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:36:28 2020

@author: connorcolombo
"""
import numpy as np
from scipy import signal, linalg
from util import *
from warnings import warn
from scipy.ndimage import gaussian_filter1d

import numpy as np
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *

import matplotlib.pyplot as plt

pos_traj = getTrajectory('buggyTrace.csv')

F_max = 15737 # [N] Maximum vehicle thrust
m = 1888.6
decel_max = 9.08 # no brakes: 1.1 # [m/s/s] Maximum capable longitudinal deceleration rate under any condition (empirical from coasting down)
accel_max = F_max / m # [m/s/s] Maximum vehicle acceleration
vcm = 16.5        

aa = accel_max
dd = decel_max
"""
t0 = 100
s0 = 20
sf = 100
v0 = 10
vf = 20
"""

def path_length(pos_traj, start, end):
    """
    Compute the path distance between two points on a trajectory.
    
    Computes the path_length distance along the given position trajectory 
    between the points at the given indices. If end < start, the distance 
    will be negative.

    Parameters
    ----------
    pos_traj
        Nx2 numpy array containing the [x,y] coordinates of each waypoint.
    start
        Index of the first point.
    end
        Index of the second point.

    Returns
    -------
    path_distance
        Path length distance between points at given indices.

    """
    idx_1, idx_2 = tuple(map(int, (start, end)))
    path_sign = 1.0
    if idx_1 > idx_2:
        # Ensure proper ordering. Swap order and sign if necessary.
        path_sign = -1.0
        idx_1, idx_2 = idx_2, idx_1
    section = pos_traj[idx_1:(idx_2+1)] # Section of trajectory between idx_1 and idx_2, inclusive
    point_distances = np.sqrt(np.sum(np.diff(section, axis=0)**2, axis=1)) # Distances between each point
    path_distance = path_sign * np.sum(point_distances)
    return path_distance

def s_traj(t0,s0,v0,sf,vf,a,d):
    """
    Computes the parameters for an s-parameterized pyramid trajectory which 
    starts at (t0,s0,v0) and ends at (sf,vf) while accelerating at a and 
    decelerating at d. If a speed is unreachable, this trajectory hits the 
    closest possible speed.
    """
    sf = sf - s0 # zero
    tp = ( np.sqrt( (d*v0**2 + a*vf**2 + 2*a*d*sf) / (a+d) ) - v0) / a
    vp = v0 + a*tp
    sp = 0.5 * (v0+vp)*tp
    tf = 2*(sf-sp)/(vp+vf) + tp
    
    sp = sp + s0 # reoffset
    tp = tp + t0 # offset
    tf = tf + t0 # offset
    
    return tp, tf, vp, sp

def s_traj_follow(s, a,d, t0,s0,v0, tp,sp,vp):
    """
    For an s-parameterized pyramid trajectory with the given parameters 
    produced by s_traj, this returns the target time and velocity at position 
    s.
    """
    if s <= sp:
        Dt = (np.sqrt(v0**2 + 2*a*(s-s0)) - v0) / a
        t = t0 + Dt
        v = v0 + a*Dt
    else:
        Dt = (vp - np.sqrt(vp**2 - 2*d*(s-sp))) / d
        t = tp + Dt
        v = vp - d*Dt
        
    return v,t

def s_traj_waypoints(pos_traj, a, d, vel_waypoints):
    if vel_waypoints[0][0] != 0:
        vel_waypoints = [(0,0)] + vel_waypoints
    
    if vel_waypoints[-1][0] != pos_traj.shape[0]:
        vel_waypoints = vel_waypoints + [(pos_traj.shape[0],0)]
    
    traj_info = []
    vavg_ratio_max = 0 # How close is vavg to requiring one of the trapezoids to exceed amax
    t0 = 0
    s0 = 0
    for w in range(1,len(vel_waypoints)):
        i_prev, v_prev = vel_waypoints[w-1]
        i, vi = vel_waypoints[w]
        
        # Path length of section:
        Ds = path_length(pos_traj, i_prev, i)
        sf = s0 + Ds
        
        # Compute Trajectory Parameters:
        tp, tf, vp, sp = s_traj(t0,s0,v_prev, sf,vi, a,d)
        
        # Append info defining the sub trapezoidal trajectory:
        traj_info.append((t0,s0,v_prev, tp,sp,vp, tf,sf,vi))
        
        # Update s0,t0
        s0 = sf
        t0 = tf
        
    
    #print(vel_waypoints)
    #print(traj_info)
    
    
    dist_traj = np.zeros((pos_traj.shape[0],1))
    time_traj = np.zeros((pos_traj.shape[0],1))
    vel_traj = np.zeros((pos_traj.shape[0],1))
    vel_traj[0] = vel_waypoints[0][1]
    
    t = 0 # trajectory info index
    s = 0 # total distance into trajectory of current waypoint
    (t0,s0,v_prev, tp,sp,vp, tf,sf,vi) = traj_info[t] # parameters of current trapezoid
    for i in range(1,pos_traj.shape[0]):
        s = s + np.sqrt(np.sum((pos_traj[i] - pos_traj[i-1])**2))
        
        # If s is outside of pyramid, advance pyramid index t until it's inside
        while s > sf:
            t = t+1
            (t0,s0,v_prev, tp,sp,vp, tf,sf,vi) = traj_info[t]
        
        assert(s0 <= s <= sf)
        #assert(sm <= se)
        
        vel_traj[i], time_traj[i] = s_traj_follow(s, a,d, t0,s0,v_prev, tp,sp,vp)
        dist_traj[i] = s
        
    #print(vel_traj)
    return vel_traj, time_traj, dist_traj, tf, traj_info

def plot_s_traj(t0,s0,v0,sf,vf,a,d):
    tp, tf, vp, sp = s_traj(t0,s0,v0,sf,vf,a,d)
    
    ss = np.linspace(s0,sf, num=1000)
    ts = np.zeros(ss.shape)
    vs = np.zeros(ss.shape)
    
    for i,s in enumerate(ss):
        vs[i], ts[i] = s_traj_follow(s, a,d, t0,s0,v0, tp,sp,vp)
    
    plt.figure()
    plt.plot(ss,vs)
    plt.axvline(s0, linestyle="--", color='blue')
    plt.axvline(sp, linestyle=":", color='black')
    plt.axvline(sf, linestyle="--", color='red')
    
    plt.axhline(v0, linestyle="--", color='blue')
    plt.axhline(vf, linestyle="--", color='red')
    plt.axhline(vp, linestyle=":", color='black')
    
    plt.xlabel('s')
    plt.ylabel('v(s)')
    plt.show()
    
    plt.figure()
    plt.plot(ts,vs)
    
    plt.axvline(t0, linestyle="--", color='blue')
    plt.axvline(tp, linestyle=":", color='black')
    plt.axvline(tf, linestyle="--", color='red')
    
    plt.axhline(v0, linestyle="--", color='blue')
    plt.axhline(vf, linestyle="--", color='red')
    plt.axhline(vp, linestyle=":", color='black')
    
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.show()
    

vel_waypoints = [
    (0,0),
    (1400,vcm/1.38),#(1700,vcm/1.38),
    (2453,vcm/2.0/1.38),
    #(3236,2*vcm), # <-
    #(5217,vcm),
    (5835,vcm/1.1),
    #(6574,2*vcm),
    (7799,vcm/1.42),
    (8203,vcm) # Floor it at the end
]

vel_traj, time_traj, dist_traj, tf, traj_info = s_traj_waypoints(pos_traj, aa, dd, vel_waypoints)

plt.figure()
plt.plot(dist_traj,vel_traj)
for t0,s0,v0, tp,sp,vp, tf,sf,vf in traj_info: 
    plt.axvline(s0, linestyle="--", color='blue')
    plt.axvline(sp, linestyle=":", color='black')
    plt.axvline(sf, linestyle="--", color='red')
    
    plt.axhline(v0, linestyle="--", color='blue')
    plt.axhline(vf, linestyle="--", color='red')
    plt.axhline(vp, linestyle=":", color='black')
plt.xlabel('s')
plt.ylabel('v(s)')
plt.show()

plt.figure()
plt.plot(time_traj,vel_traj)
for t0,s0,v0, tp,sp,vp, tf,sf,vf in traj_info: 
    plt.axvline(t0, linestyle="--", color='blue')
    plt.axvline(tp, linestyle=":", color='black')
    plt.axvline(tf, linestyle="--", color='red')
    
    plt.axhline(v0, linestyle="--", color='blue')
    plt.axhline(vf, linestyle="--", color='red')
    plt.axhline(vp, linestyle=":", color='black')
plt.xlabel('t')
plt.ylabel('v(t)')
plt.show()

    
    