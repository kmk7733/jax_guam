import numpy as np
import matplotlib.pyplot as plt

# h2c1 = np.array([[0, 0, 0], [0, 0, -80], [150, 0, -100]])
# # h2c2 = np.array([[0, 0, 0], [0, 0, -80], [200, 0, -80], [200, 200, -80], [0, 200, -80], [0, 200, -80]])

# ax = plt.figure().add_subplot(projection='3d')
# ax.set_facecolor('white')
# ax.plot(h2c1[:,0],h2c1[:,1],-h2c1[:,2])
# ax.set_xlabel('North [ft]')
# ax.set_ylabel('East [ft]')
# ax.set_zlabel('Height [ft]')
# # ax.set_aspect("equal")
# ax.view_init()
# plt.autoscale(enable=True)
# plt.show()

T_t = np.arange(0, 21, 1)
sinusoidal = np.vstack((T_t, 10*np.cos(2*np.pi/300*T_t), 10*np.sin(2*np.pi/300*T_t))).T

ax = plt.figure().add_subplot(projection='3d')
ax.set_facecolor('white')
ax.plot(sinusoidal[:,0],sinusoidal[:,1],sinusoidal[:,2])
ax.set_xlabel('North [ft]')
ax.set_ylabel('East [ft]')
ax.set_zlabel('Height [ft]')
# ax.set_aspect("equal")
ax.view_init()
plt.autoscale(enable=True)
plt.show()
