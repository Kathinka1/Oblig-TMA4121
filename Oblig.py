# RIKTIG
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

N_x= 15
N_y = 15
N_t = 10000

L_x = 1 # lengde stav
L_y = 1
T = 10 # total lengde tid

h = L_x/(N_x-1) 
p = L_y/(N_y-1)
k = T/(N_t-1)

gamma_x = k/(h**2)
gamma_y = k/(p**2)

u = np.zeros((N_x,N_y, N_t)) 

x = np.linspace(0, L_x, N_x)
y = np.linspace(0, L_y, N_y)

## Randkrav og initialbetingelser
# u(x,y,0) = f(x,y)
# u(x,y,t) = 0 på randen til omega.

for i in range(1, N_x-1):
    for j in range(1, N_y-1):
        u[i,j,0] = np.sin(np.pi * x[i]) + np.sin(np.pi * y[i])

u[0, :, :] = u[-1, :, :] = 0
u[:, 0, :] = u[:, -1, :] = 0

# Eksplisitt metode
for l in range(N_t-1):
    for i in range(1, N_x-1): 
        for j in range(1, N_y-1):
            u[i,j, l+1] = u [i,j,l] + gamma_x *(-2*u[i,j,l] + u[i+1,j,l] + u[i-1,j,l]) + gamma_y * (-2*u[i,j,l] + u[i,j+1,l] + u[i,j-1,l])
            

fig, ax = plt.subplots() 
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) 

def init(): 
    global cax  
    cax = ax.imshow(u[:, :, 0], extent=(0, L_x, 0, L_y), origin='lower', vmin=np.min(u), vmax=np.max(u)) 
    fig.colorbar(cax) 
    return cax, time_text 

def update(frame): 
    cax.set_data(u[:, :, frame]) 
    time_text.set_text('Tid: {:.2f}s'.format(frame * k)) 
    return cax, time_text 

ani = animation.FuncAnimation(fig, update, frames=N_t, init_func=init, blit=False, interval=50) 
plt.title("Temperaturfordeling over tid") 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.show()