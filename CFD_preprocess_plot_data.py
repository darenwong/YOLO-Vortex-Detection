# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:37:14 2018

@author: Daren
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def load_CFD_data(angle, i):
    """Load data of i th slice of the blade CFD simulation data at specific blade angle. 
    
    Parameters
    ----------     
    angle : float 
        Blade angle at which the CFD simulation ran at
        
    i : float 
        i th slice of data in the simulation data
    
    Returns
    -------
    y, z, vsec_t, vsec_r : numpy array of size (lz0,ly0) where:
        ly0 : length of original velocity vector field in y direction
        lz0 : length of original velocity vector field in z direction
        Uneven, irregular mesh arrays
        
    """ 
    
    y = np.load('CFD data/y_ang_' + str(angle) + '_i_' + str(i) + '.npy')
    z = np.load('CFD data/z_ang_' + str(angle) + '_i_' + str(i) + '.npy')
    vsec_t = np.load('CFD data/vsec_t_ang_' + str(angle) + '_i_' + str(i) + '.npy')
    vsec_r = np.load('CFD data/vsec_r_ang_' + str(angle) + '_i_' + str(i) + '.npy')
    return y, z, vsec_t, vsec_r


def interpolate(ly, lz, y, z, vsec_t, vsec_r):
    """Interpolate data onto an even, regular mesh grid
    
    Parameters
    ----------     
    ly, lz : int
        Desired shape of velocity vector array. Eg, (90, 80) if ly = 80, lz = 90
    
    y, z, vsec_t, vsec_r : numpy array of size (lz0,ly0) where:
        ly0 : length of original velocity vector field in y direction
        lz0 : length of original velocity vector field in z direction
        Uneven, irregular mesh arrays
    
    Returns
    -------
    yi, zi, vsec_t1, vsec_r1 : numpy array of size (lz,ly) 
        Interpolated mesh arrays
        
    """ 
    
    # Set up even, regular mesh grid
    yi = np.linspace(np.min(y),np.max(y),ly)
    zi = np.linspace(np.min(z),np.max(z),lz)
    yi, zi = np.meshgrid(yi, zi)
    
    # Apply cubic interpolation
    vsec_t1 = griddata((y.flatten(), z.flatten()), vsec_t.flatten(), (yi, zi), method='cubic')
    vsec_r1 = griddata((y.flatten(), z.flatten()), vsec_r.flatten(), (yi, zi), method='cubic')
    return yi, zi, vsec_t1, vsec_r1


def remove_nan(y, z, v, u):
    """Remove Nan values as a result of error from interpolating data that are near the edge or corner
    
    Parameters
    ----------     
    y, z, v, u : numpy array of size (lz0,ly0) where:
        ly0 : length of original velocity vector field in y direction
        lz0 : length of original velocity vector field in z direction
    
    Returns
    -------
    y, z, v, u : numpy array of size (lyi,lzi) where:
        lyi : length of resulting velocity vector field in y direction
        lzi : length of resulting velocity vector field in z direction
        and lyi <= ly, lzi <= lz
        
    """ 
    
    # Plot original vsec_t array
    plt.imshow(v)
    plt.title("vsec_t")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    # Plot original vsec_r array
    plt.imshow(u)
    plt.title("vsec_r")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    
    # Plot quiver plot of original velocity vector field
    plt.quiver(y, z, v, u)
    plt.title("Vortex (Interpolated)")
    plt.xlabel("y")
    plt.ylabel("z")
    
    plt.xticks()
    plt.yticks()
    plt.axis()
    plt.grid()
    plt.show()
    
    
    # Check all four sides of array if there is Nan values except at the four corner
    while any(np.isnan(v[0,1:-1])) or any(np.isnan(u[0,1:-1])) or any(np.isnan(v[1:-1,0])) or any(np.isnan(u[1:-1,0])) or any(np.isnan(v[1:-1,-1])) or any(np.isnan(u[1:-1,-1])) or any(np.isnan(v[-1,1:-1])) or any(np.isnan(u[-1,1:-1])):
        
        # Check top side
        if any(np.isnan(v[0,1:-1])) or any(np.isnan(u[0,1:-1])):
            v = v[1:,:]
            u = u[1:,:]
            y = y[1:,:]
            z = z[1:,:]

        # Check left side
        if any(np.isnan(v[1:-1,0])) or any(np.isnan(u[1:-1,0])):
            v = v[:,1:]
            u = u[:,1:]
            y = y[:,1:]
            z = z[:,1:]

        # Check right side
        if any(np.isnan(v[1:-1,-1])) or any(np.isnan(u[1:-1,-1])):
            v = v[:,:-1]
            u = u[:,:-1]
            y = y[:,:-1]
            z = z[:,:-1]

        # Check bottom side
        if any(np.isnan(v[-1,1:-1])) or any(np.isnan(u[-1,1:-1])):
            v = v[:-1,:]
            u = u[:-1,:]
            y = y[:-1,:]
            z = z[:-1,:]

    # Plot original vsec_t array
    plt.imshow(v)
    plt.title("vsec_t")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    # Plot original vsec_r array
    plt.imshow(u)
    plt.title("vsec_r")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    
    # Plot quiver plot of original velocity vector field
    plt.quiver(y, z, v, u)
    plt.title("Vortex (Trimmed)")
    plt.xlabel("y")
    plt.ylabel("z")
    
    plt.xticks()
    plt.yticks()
    plt.axis()
    plt.grid()
    plt.show()

    return y, z, v, u


def reshape_remove_nan(ly, lz, yi, zi, vsec_t1, vsec_r1, angle, i):
    """Generate data arrays of shape (lz, ly) after applying interpolation and removing Nan values
    
    Parameters
    ----------     
    ly, lz : int
        Desired shape of velocity vector array. Eg, (90, 80) if ly = 80, lz = 90
    
    yi, zi, vsec_t1, vsec_r1 : numpy array of size (lz0,ly0) where:
        ly0 : length of original velocity vector field in y direction
        lz0 : length of original velocity vector field in z direction
        Interpolated mesh arrays
    
    angle : float 
        Blade angle at which the CFD simulation ran at
        
    i : float 
        i th slice of data in the simulation data
    
    
    Returns
    -------
    yi, zi, vsec_t1, vsec_r1 : numpy array of size (lz, ly)
        
    """ 
    
    # Apply remove_nan for the first time and then find the size reduction (or the number of row and column removed because of nan values)
    yi, zi, vsec_t1, vsec_r1 = remove_nan(yi, zi, vsec_t1, vsec_r1)

    if yi.shape[0] < lz:
        row_change = lz - yi.shape[0]
    if yi.shape[1] < ly:
        col_change = ly - yi.shape[1]
        
    # Load CFD data again to run remove_nan function again to give the correct shape
    y, z, vsec_t, vsec_r = load_CFD_data(angle, i)    
    
    # Account for extra column or row reduction due to Nan values
    ly += col_change
    lz += row_change 
    
    # Apply interpolation and remove_nan function and this time the resulting arrays should have the shape (lz, ly)
    yi, zi, vsec_t1, vsec_r1 = interpolate(ly, lz, y, z, vsec_t, vsec_r)    
    yi, zi, vsec_t1, vsec_r1 = remove_nan(yi, zi, vsec_t1, vsec_r1)
    
    return yi, zi, vsec_t1, vsec_r1 


def save_data(yi, zi, vsec_t1, vsec_r1, angle, i):
    """Save image and data file in .npy form to be loaded and fed into the neural network
    
    Parameters
    ----------         
    yi, zi, vsec_t1, vsec_r1 : numpy array of size (lz0, ly0) where:
        ly0 : length of original velocity vector field in y direction
        lz0 : length of original velocity vector field in z direction

    angle : float 
        Blade angle at which the CFD simulation ran at
        
    i : float 
        i th slice of data in the simulation data

    Returns
    -------
    image, data : numpy array of size (lz, ly)
        
    """ 
    
    # Generate image and input data array 
    image = []
    data = []
    conc_data = np.zeros((yi.shape[0], yi.shape[1], 2))
    
    for ii in range(yi.shape[0]):
        for jj in range(yi.shape[1]):
            conc_data[ii,jj] = (vsec_t1[ii,jj], vsec_r1[ii,jj])
    
    image.append((vsec_t1, vsec_r1))
    data.append((conc_data))
    
    data = np.array(data)
    image = np.array(image)
    
    # Save image and input data in .npy file format
    np.save('CFD data/CFD_Harrison_image_ang_' + str(angle) + '_i_' + str(i)  + '.npy' ,image)
    np.save('CFD data/CFD_Harrison_data_ang_' + str(angle) + '_i_' + str(i)  + '.npy' ,data)
    return image, data

def plot_CFD_data(yi, zi, vsec_t1, vsec_r1, y, z, vsec_t, vsec_r):
    """Plot all before and after arrays and quiver plots"""
    
    # Plot original vsec_t array
    plt.imshow(vsec_t)
    plt.title("vsec_t")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    # Plot final processed vsec_t array
    plt.imshow(vsec_t1)
    plt.title("vsec_t1")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    # Plot original vsec_r array
    plt.imshow(vsec_r)
    plt.title("vsec_r")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    # Plot final processed vsec_r array
    plt.imshow(vsec_r1)
    plt.title("vsec_r1")
    plt.colorbar()
    plt.axis()
    plt.show()
    
    # Plot quiver plot of original velocity vector field
    plt.quiver(y, z, vsec_t, vsec_r)
    plt.title("Vortex (Original)")
    plt.xlabel("y")
    plt.ylabel("z")
    
    plt.xticks()
    plt.yticks()
    plt.axis()
    plt.grid()
    plt.show()
    
    # Plot quiver plot of final velocity vector field
    plt.quiver(yi, zi, vsec_t1, vsec_r1)
    plt.title("Vortex (Interpolated)")
    plt.xlabel("yi")
    plt.ylabel("zi")
    
    plt.xticks()
    plt.yticks()
    plt.axis()
    plt.grid()
    plt.show()

def main(angle, i, ly, lz): 
    """Main function to run all procedure from loading data to processing and plotting final data arrays and finally saving data ready to be fed into neural network for vortex detection
    
    Parameters
    ----------   
    angle : float 
        Select the blade angle at which the CFD simulation ran at
        
    i : float 
        Select the i th slice of data in the simulation data
        
    ly, lz : int
        Desired shape of velocity vector array. Eg, (90, 80) if ly = 80, lz = 90
        
    """     
    
    y, z, vsec_t, vsec_r = load_CFD_data(angle, i)
    yi, zi, vsec_t1, vsec_r1 = interpolate(ly, lz, y, z, vsec_t, vsec_r)
    yi, zi, vsec_t1, vsec_r1 = reshape_remove_nan(ly, lz, yi, zi, vsec_t1, vsec_r1, angle, i)
    image, data = save_data(yi, zi, vsec_t1, vsec_r1, angle, i)
    plot_CFD_data(yi, zi, vsec_t1, vsec_r1, y, z, vsec_t, vsec_r)
    
    
    

#Run main function
main(angle = 40, i = 45, ly = 80, lz = 80)
