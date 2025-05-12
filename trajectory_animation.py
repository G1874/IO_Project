import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

trajectory = np.load("Trajectory/trajectory_2025-05-06_13-28-33.npy", allow_pickle=True)
t = trajectory[:,2]
hand_1_traj = trajectory[:,0]
hand_2_traj = trajectory[:,1]

def draw_hand(kp):
    x = kp[:,0]
    y = 1 - kp[:,1]
    plt.scatter(x,y)
    plt.plot(x[0:5],y[0:5])
    idx = [0] + list(range(5,8+1))
    plt.plot(x[idx],y[idx])
    idx = [0] + list(range(9,12+1))
    plt.plot(x[idx],y[idx])
    idx = [0] + list(range(13,16+1))
    plt.plot(x[idx],y[idx])
    idx = [0] + list(range(17,20+1))
    plt.plot(x[idx],y[idx])
    idx = [5,9,13,17]
    plt.plot(x[idx],y[idx])
    plt.xlim(0,1)
    plt.ylim(0,1)

plt.ion()
plt.figure()

for i, (hand_1_kpts, hand_2_kpts) in enumerate(zip(hand_1_traj,hand_2_traj)):
    plt.cla()
    if hand_1_kpts.size != 0:
        draw_hand(hand_1_kpts)
    if hand_2_kpts.size != 0:
        draw_hand(hand_2_kpts)
    plt.show()
    plt.pause(t[1]-t[0])