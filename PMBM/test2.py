import numpy as np
import matplotlib.pyplot as plt
import math
# Create an array of theta values in degrees (e.g., from 0 to 113*360 degrees)
theta_degrees = np.linspace(0, 113*360, 10000)

# Convert degrees to radians
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("black")
# ax.gca().set_facecolor('black') # Set background color to black
# ax.gca().set_aspect('equal')  # Equal aspect ratio
# ax.grid(False)  # Turn off the grids
ax.set_xlim(-2.5, 2.5) # X-axis limit
ax.set_ylim(-2.5, 2.5) # Y-axis limit
i=0
for deg in theta_degrees:
    theta_radians = np.deg2rad(deg)

    a=np.pi
    a=math.e
    #a=-np.pi
    # a=10/3
    #a=1/3
    #a =5
    # a = 2.15615614186541865418641561561561
    # a=2.5641686484865418964153489615648951654165146851
    # a=math.sin(theta_radians)
    # Calculate z(theta) using the formula , 1j is imaginary number
    z = np.exp(theta_radians * 19.98j) + np.exp(a * theta_radians * 19.98j) #+ np.exp((a**2) * theta_radians * 1j)
    if i==0:
        prev_x=np.real(z)
        prev_y=np.imag(z)
        i=-1
    # Separate the real and imaginary parts of z
    x = np.real(z)
    y = np.imag(z)

    # Create a plot with specific settings
    allX=np.array([prev_x,x])
    allY=np.array([prev_y,y])
    ax.plot(allX, allY, color='white', linewidth=0.5)  # Set line color to white and line width to 0.5
    plt.pause(0.00000001)
    prev_x=x
    prev_y=y



# plt.show() # Display the plot