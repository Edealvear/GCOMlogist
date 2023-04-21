# -*- coding: utf-8 -*-
"""
@author: Robert
"""

#from mpl_toolkits import mplot3d

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

vuestra_ruta = ""

os.getcwd()
#os.chdir(vuestra_ruta)


# """
# Ejemplo1
# """

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
# cset = ax.contour(X, Y, Z, 16, extend3d=True)
# ax.clabel(cset, fontsize=9, inline=1)
# plt.show()

# """
# Ejemplo2
# """

# def g(x, y):
#     return np.sqrt(1-x ** 2 - y ** 2)

# x = np.linspace(-1, 1, 30)
# y = np.linspace(-1, 1, 30)

# X, Y = np.meshgrid(x, y)
# Z = g(X, Y)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 10, cmap='binary')
# ax.contour3D(X, Y, -1*Z, 10, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')


# """
# Ejemplo3
# """

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# t2 = np.linspace(1, 0, 100)
# x2 = t2 * np.sin(20 * t2)
# y2 = t2 * np.cos(20 * t2)
# z2 = np.sqrt(1-x2**2-y2**2)

# c2 = x2 + y2

# ax.scatter(x2, y2, z2, c=c2)
# ax.plot(x2, y2, z2, '-b')


"""
2-sphere
"""

u = np.linspace(0.05, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 60)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))



# t2 = np.linspace(0.01, 1, 200)
# x2 = abs(t2) * np.sin(80 * t2/2)
# z2 =abs(t2) * np.cos(80 * t2/2)
# y2 = np.sqrt(1-x2**2-z2**2)
# c2 = x2 + y2


t2 = np.linspace(-np.pi+0.001, np.pi-0.001, 500)
x2 =np.cos(t2)**2
z2 =np.cos(t2)*np.sin(t2)
y2 =np.sin(t2)

c2 = x2 + y2


fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.plot(x2, y2, z2, '-b',c="gray")

ax.plot_surface(x, y, z, rstride=1, cstride=1,
              cmap='viridis', edgecolor='none')

ax.plot(x2, y2, z2, '-b',c="gray",zorder=3)
#ax.plot_wireframe(x2, y2, z2)

ax.set_title('surface')



"""
2-esfera proyectada
"""

def proj(x,z,z0=1, alpha=1):
    z0 = z*0+z0
    eps = 1e-16
    #x_trans = x/(abs(z0-z)**alpha+eps)
    x_trans = x/((abs(1-z)**.5) + eps)
    return(x_trans)
    #Nótese que añadimos un épsilon para evitar dividi entre 0!!
    

z0 = 1

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

c2 = np.sqrt(x2**2+y2**2)
col = plt.get_cmap("hot")(c2/np.max(c2))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
#ax.plot(x2, y2, z2, '-b',c="gray",zorder=3)
ax.scatter(x2, y2, z2, '-b',c=col,zorder= 3,s=0.1)
ax.set_title('2-sphere')
#ax.text(0.5, 90, 'PCA-'+str(i), fontsize=18, ha='center')

ax = fig.add_subplot(2, 2, 2, projection='3d')
#ax.set_xlim3d(-8,8)
#ax.set_ylim3d(-8,8)
#ax.set_zlim3d(0,1000)
ax.plot_surface(proj(x,z,z0=z0), proj(y,z,z0=z0), z*0-1, rstride=1, cstride=1,
                cmap='viridis', alpha=0.5, edgecolor='purple')

ax.plot(proj(x2,z2,z0=z0), proj(y2,z2,z0=z0), -1, '-b',c="gray",zorder=1)
#ax.scatter(proj(x2,z2,z0), proj(y2,z2,z0), -1, '-b',c="gray",zorder= 3,s=0.01)
ax.set_title('Stereographic projection')
plt.show()
fig.savefig('stereo1.png', dpi=250)   # save the figure to file
plt.close(fig) 



"""
2-esfera proyectada - familia paramétrica - FORMA INCORRECTA
"""

t = 1
z0 = 1

xt = proj(x,z,z0)*(t)+x*(1-t)
yt = proj(y,z,z0)*t+y*(1-t)
zt = (z*0-z0)*t + z*(1-t)
#Esto es para la curva
x2t = proj(x2,z2,z0)*t+x2*(1-t)
y2t = proj(y2,z2,z0)*t+y2*(1-t)
z2t = -z0*t+z2*(1-t)

fig = plt.figure(figsize=(6,6))
#fig.subplots_adjust(hspace=0.4, wspace=0.2)
ax = plt.axes(projection='3d')


# ax.set_xlim3d(-8,8)
# ax.set_ylim3d(-8,8)
ax.plot_surface(xt, yt, zt, rstride=1, cstride=1,
                cmap='viridis',alpha=0.5 ,edgecolor='purple')
ax.plot(x2t,y2t, -1, '-b',c="gray",zorder=3)

plt.show()
fig.savefig('stereo2.png')   # save the figure to file
plt.close(fig) 



"""
HACEMOS LA ANIMACIÓN
"""

from matplotlib import animation
#from mpl_toolkits.mplot3d.axes3d import Axes3D

def proj2(x,z,z0=1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = 2*x/(abs(z0-z)+eps)
    return(x_trans)

def animate(t):

    #Esto primero es para la esfera
    xt = (2/(2*(1-t)+(1-z)*t))*x
    yt = (2/(2*(1-t)+(1-z)*t))*y
    zt =  - t +z*(1-t)
    #Esto es para la curva
    x2t = (2/(2*(1-t)+(1-z2)*t))*x2
    y2t = (2/(2*(1-t)+(1-z2)*t))*y2
    z2t = - t +z2*(1-t)

    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, alpha=0.5,
                    cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b',c="gray")
    return ax

def init():
    return animate(0)



fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.01), init_func=init,
                              interval=60)
ani.save("familiaparamEEAD.gif", fps = 10) 
#ani.save("ejemplo.gif", fps = 5)

"""
Alternativas para hacer animación
#celluloid
pypi.org/project/celluloid

#APNG
from apng import APNG
APNG.from_files(['atleta-01.jpg',
                 'atleta-02.jpg', 
                 'atleta-03.jpg',
                 'atleta-04.jpg',
                 'atleta-05.jpg'], 
                 delay=100).save('animatleta1.png')

"""
#Alternativas:
#







