import matplotlib.pyplot as plt
import numpy as np

z = np.array([12.8862, 13.1128, 17.5264, 17.1846, 12.9345, 13.0189, 11.4193, 12.7262, 12.9003, 12.4271, 11.9444, 10.2754, 9.24364, 9.27054, 8.40269, 9.14267, 12.2499, 7.80588])
t = np.arange(1, 19, 1)

ttcCam = np.array([12.7377, 13.7577, 13.8578, 13.9104, 14.7384, 14.1695, 15.6235, 14.6821, 13.1281, 11.6663, 12.1952, 10.6396, 10.8458, 9.96229, 9.33382, 9.79594, 9.52098, 8.83997])

plt.xlim([0, 19])
plt.ylim([7.5, 17])
# plt.ylim([8, 17])
plt.xticks(np.arange(min(t), max(t)+1, 1.0))
plt.yticks(np.arange(min(z), max(z)+1, 0.8))
#plt.yticks(np.arange(min(ttcCam), max(ttcCam)+1, 0.6))
plt.grid(color='r', linestyle='-', linewidth=0.25)

#plt.title("TTC Camera AKAZE-FREAK")
plt.title("TTC LiDAR")
plt.xlabel("Frame")
#plt.ylabel("TTC Camera (s)")
plt.ylabel("TTC LiDAR (s)")
#plt.plot(t, ttcCam)
plt.plot(t, z)
plt.show()
