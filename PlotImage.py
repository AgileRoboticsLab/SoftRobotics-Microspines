import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import cv2

# Plot the data
fig, p_ax = plt.subplots(1) #Plot axis
T1_plot, = p_ax.plot([], [], '*', color='r', lw=2)
T2_plot, = p_ax.plot([], [], '+', color='b', lw=2)
T3_plot, = p_ax.plot([], [], '-', color='g', lw=2)
centroid_plot, = p_ax.plot([],[],'H',color='black',lw=2)

# Titles
p_ax.set_title('Tracker Positions')

#Set the x and y limits of the plot based on the camera resolution
p_ax.set_xlim(0, 640) 
p_ax.set_ylim(0, 480)
p_ax.invert_yaxis()

canvas = FigureCanvasAgg(plt.gcf())

def update_plot(T1_plot_x, T1_plot_y, T2_plot_x, T2_plot_y, T3_plot_x, T3_plot_y, centroid_x_data, centroid_y_data):
    T1_plot.set_xdata(T1_plot_x)
    T1_plot.set_ydata(T1_plot_y)
    T2_plot.set_xdata(T2_plot_x)
    T2_plot.set_ydata(T2_plot_y)
    T3_plot.set_xdata(T3_plot_x)
    T3_plot.set_ydata(T3_plot_y)
    centroid_plot.set_xdata(centroid_x_data)
    centroid_plot.set_ydata(centroid_y_data)

    canvas.draw()
    plot_image = np.array(canvas.renderer.buffer_rgba())
    plot_image = cv2.resize(plot_image, (640,480))
    
    return plot_image