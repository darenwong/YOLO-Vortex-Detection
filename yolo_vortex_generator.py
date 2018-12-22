# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:54:55 2018

@author: Daren
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter


def vortex(R1, R2, x, y, bx, by, direction):
    """Generate 2-D velocity vector field of a circular vector field

    Parameters
    ----------
    R1, R2 : float
        R1 and R2 defines the size of vortex where:
            1. For radius < RC (critical radius), velocity is proportional to R1 
            2. For radius > RC (critical radius), velocity is inversely proportional to R2
            
    x, y : numpy array
        A mesh grid of coordinates in x and y direction
        
    bx, by : float
        Location of center of vortex
        
    direction: str
        Direction of rotation of vortex

    Returns
    -------
    u, v : numpy array
        velocity vector field in x and y direction
        
    br : float
        critical radius of the vortex
    """    
    
    u,v = np.zeros((y.shape[0], x.shape[0])), np.zeros((y.shape[0], x.shape[0]))
    
    br = np.sqrt(R2/R1)    
    
    if direction == 'ac':
        direction = 1
    elif direction == 'c':
        direction = -1
    
    for ni, i in enumerate(x):
        for nj, j in enumerate(y):
            if np.hypot(i-bx,j-by) < br:
                u[nj,ni], v[nj,ni] = -R1*(j-by)*direction, R1*(i-bx) *direction              
            else:
                u[nj,ni], v[nj,ni] = -R2*(j-by)*direction/((i-bx)**2+(j-by)**2), R2*(i-bx)*direction/((i-bx)**2+(j-by)**2)
    return u, v, br


def generate_random_vortex(l):
    """Randomly resize the 2-D velocity vector field of a circular vortex in x and y direction 
    and then randomly rotate it to generate a 2-D velocity vector field of a non-circular vortex 
    rotated at an arbitrary angle.

    Parameters
    ----------
    l : float
        Size of the velocity vector field array. Eg, if l = 80 then shape of velocity vector field array is (80,80)
            
    x, y : numpy array
        A mesh grid of location in x and y direction

    Returns
    -------
    u, v : numpy array of size (l,l)
        Velocity vector field in x and y direction
        
    bx, by : float
        Location of center of vortex
        
    bw, bh : float
        Width and height of vortex
        
    br : float
        Radius of smallest circle that circumscribe the final vortex
        
    ba : float
        Angle of rotation
        
    """  
    
    # ud and lr set the initial mesh grid size to generate a circular vortex which will then be resized back to (200, 200)
    # for eg, if ud = 100, lr = 40, then the resulting vortex will be resized by a factor of 2 bigger in v direction and a factor of 5 bigger in u direction
    ud = np.random.randint(100,150)
    lr = np.random.randint(40,100)
    
    # setting up x and y mesh grid
    x = np.linspace(0, lr-1, lr)
    y = np.linspace(0, ud-1, ud)
    
    # randomly choose the center location of vortex
    bx = lr*np.random.rand()
    by = ud*np.random.rand()
    
    # calculate distance of center of vortex from center of grid
    bl = np.hypot(200/2-bx*(200/lr), 200/2-by*(200/ud))
    
    # regenerate new bx, by and bl parameters if bl exceeds l/2 so that the vortex still stays within the frame of the grid even after rotated and cropped from size of (200,200) to (80,80) 
    while bl > l/2:
        bx = lr*np.random.rand()
        by = ud*np.random.rand()
        bl = np.hypot(200/2-bx*(200/lr), 200/2-by*(200/ud))
        
    # generate random R1 and R2 parameters which define the critical radius of the vortex
    R1 = np.random.rand()
    R2 = R1*(14*(np.random.rand() + 0.1)/2)**2
    
    # randomly choose the direction of rotation of vortices
    if np.random.rand() > 0.5:
        direction = 'c'
    else:
        direction = 'ac'
        
    # generate velocity vector field (u & v) and critical radius (br) of the vortex
    u, v, br = vortex(R1, R2, x, y, bx, by, direction)
    
    # normalise both u and v
    k = (u,v)
    if abs(np.amax(k)) > abs(np.amin(k)):
        u = u/abs(np.amax(k))
        v = v/abs(np.amax(k))
    else:
        u = u/abs(np.amin(k))
        v = v/abs(np.amin(k)) 
    
    print('u.shape', u.shape)
    plt.imshow(u)
    plt.title("u")
    plt.colorbar()
    plt.axis([0, lr, 0, ud])
    plt.show()    
    
    print('v.shape', v.shape)
    plt.imshow(v)
    plt.title("v")
    plt.colorbar()
    plt.axis([0, lr, 0, ud])
    plt.show() 
    
    ### construct velocity and angle scalar field from the velocity vector field.
    
    #vel : 
    #    Velocity scalar field obtained by calculating the magnitude of velocity in u and v
    
    #ang, ang1 : 
    #    Velocity angle scalar field obtained by calculating the angle of velocity, v relative to u
    #    As angle is in the range of 0 to 360 degree, there will always be discontinuity in the ang array
    #    This discontinuity can cause error in interpolation when resizing or rotating the ang array.
    #    This discontinuity can be treated by finding another angle array (ang1) with discontinuity occuring at 
    #    180 degree relative to (ang). Since both ang and ang1 have discontinuities at different angle, the parts of 
    #    ang and ang1 that are not affected by discontinuity, the resulting vel array can be concatenated to obtain a 
    #    velocity array that is not affected by the discontinuity in ang and ang1.
    
    vel = np.hypot(u,v)
    ang = np.arctan((v)/(u+1e-20))
    ang1 = np.arctan((v)/(u+1e-20))
    
    for i in range(ang.shape[0]):
        for j in range(ang.shape[1]):
            if u[i,j] <= 0:
                ang[i,j] += np.pi
                ang1[i,j] -= np.pi
    
    print('vel.shape', vel.shape)
    plt.imshow(vel)
    plt.title("vel")
    plt.colorbar()
    plt.axis([0, vel.shape[1], 0, vel.shape[0]])
    plt.show()    
            
    print('ang.shape', ang.shape)
    plt.imshow(ang)
    plt.title("ang")
    plt.colorbar()
    plt.axis([0, ang.shape[1], 0, ang.shape[0]])
    plt.show()    
               
    print('ang1.shape', ang1.shape)
    plt.imshow(ang1)
    plt.title("ang1")
    plt.colorbar()
    plt.axis([0, ang1.shape[1], 0, ang1.shape[0]])
    plt.show() 
    
    # separate ang, ang1 to two separate graphs for resize operation to prevent interpolation error between 0 and 360 angle (discontinuity in angle)
    ang_top = ang[int(by)+1:, :]
    ang_bot = ang[:int(by)+1, :]
    
    ang_top1 = ang1[int(by)+1:, :]
    ang_bot1 = ang1[:int(by)+1, :]
    
    print('ang_top.shape', ang_top.shape)
    plt.imshow(ang_top)
    plt.title("ang_top")
    plt.colorbar()
    plt.axis([0, ang_top.shape[1], 0, ang_top.shape[0]])
    plt.show()
    
    print('ang_bot.shape', ang_bot.shape)
    plt.imshow(ang_bot)
    plt.title("ang_bot")
    plt.colorbar()
    plt.axis([0, ang_bot.shape[1], 0, ang_bot.shape[0]])
    plt.show()
    
    print('ang_top1.shape', ang_top1.shape)
    plt.imshow(ang_top1)
    plt.title("ang_top1")
    plt.colorbar()
    plt.axis([0, ang_top1.shape[1], 0, ang_top1.shape[0]])
    plt.show()
    
    print('ang_bot1.shape', ang_bot1.shape)
    plt.imshow(ang_bot1)
    plt.title("ang_bot1")
    plt.colorbar()
    plt.axis([0, ang_bot1.shape[1], 0, ang_bot1.shape[0]])
    plt.show()    

    # Resize vel, ang_top, ang_bot, ang_top1, ang_bot1 to an everall size of 200
    vel = cv2.resize(vel, (200,200))
    ang_top = cv2.resize(ang_top, (200, (200-int(by*200/ud))))
    ang_bot = cv2.resize(ang_bot, (200, int(by*200/ud)))
    ang_top1 = cv2.resize(ang_top1, (200, (200-int(by*200/ud))))
    ang_bot1 = cv2.resize(ang_bot1, (200, int(by*200/ud)))

    print('vel.shape', vel.shape)
    plt.imshow(vel)
    plt.title("vel")
    plt.colorbar()
    plt.axis([0, vel.shape[1], 0, vel.shape[0]])
    plt.show()

    print('ang_top.shape', ang_top.shape)
    plt.imshow(ang_top)
    plt.title("ang_top")
    plt.colorbar()
    plt.axis([0, ang_top.shape[1], 0, ang_top.shape[0]])
    plt.show()
    
    print('ang_bot.shape', ang_bot.shape)
    plt.imshow(ang_bot)
    plt.title("ang_bot")
    plt.colorbar()
    plt.axis([0, ang_bot.shape[1], 0, ang_bot.shape[0]])
    plt.show()
    
    print('ang_top1.shape', ang_top1.shape)
    plt.imshow(ang_top1)
    plt.title("ang_top1")
    plt.colorbar()
    plt.axis([0, ang_top1.shape[1], 0, ang_top1.shape[0]])
    plt.show()
    
    print('ang_bot1.shape', ang_bot1.shape)
    plt.imshow(ang_bot1)
    plt.title("ang_bot1")
    plt.colorbar()
    plt.axis([0, ang_bot1.shape[1], 0, ang_bot1.shape[0]])
    plt.show()

    # Concatenate the separated ang graphs back to ang of size (200,200)
    ang = np.concatenate((ang_bot ,ang_top), axis = 0)
    ang1 = np.concatenate((ang_bot1 ,ang_top1), axis = 0)
    
    print('ang.shape', ang.shape)
    plt.imshow(ang)
    plt.title("ang")
    plt.colorbar()
    plt.axis([0, ang.shape[1], 0, ang.shape[0]])
    plt.show()    
               
    print('ang1.shape', ang1.shape)
    plt.imshow(ang1)
    plt.title("ang1")
    plt.colorbar()
    plt.axis([0, ang1.shape[1], 0, ang1.shape[0]])
    plt.show() 
    
    # Randomly choose an angle between 0 to 180 degree to rotate the vortex by
    angle = np.random.randint(0,180)
    
    # Rotate vel, ang and ang1 array by the angle defined earlier 
    vel = rotate(vel, angle, reshape = False)
    ang = rotate(ang, angle, reshape = False)
    ang1 = rotate(ang1, angle, reshape = False)
    
    print('vel.shape', vel.shape)
    plt.imshow(vel)
    plt.title("vel")
    plt.colorbar()
    plt.axis([0, vel.shape[1], 0, vel.shape[0]])
    plt.show()   
    
    print('ang.shape', ang.shape)
    plt.imshow(ang)
    plt.title("ang")
    plt.colorbar()
    plt.axis([0, ang.shape[1], 0, ang.shape[0]])
    plt.show()    
               
    print('ang1.shape', ang1.shape)
    plt.imshow(ang1)
    plt.title("ang1")
    plt.colorbar()
    plt.axis([0, ang1.shape[1], 0, ang1.shape[0]])
    plt.show() 
    
    # Convert angle from degree to radian
    angle = (angle/180)*np.pi
    
    # Subtract the predefined angle from the ang, ang1 array which is equivalent to rotating the velocity field by the predefined angle
    ang -= angle
    ang1 -= angle
    
    # Crop a grid of size (80,80) from the resized vel, ang, ang1 array
    vel = vel[int(200/2-l/2):int(200/2+l/2), int(200/2-l/2):int(200/2+l/2)]
    ang = ang[int(200/2-l/2):int(200/2+l/2), int(200/2-l/2):int(200/2+l/2)]
    ang1 = ang1[int(200/2-l/2):int(200/2+l/2), int(200/2-l/2):int(200/2+l/2)]

    print('vel.shape', vel.shape)
    plt.imshow(vel)
    plt.title("vel")
    plt.colorbar()
    plt.axis([0, vel.shape[1], 0, vel.shape[0]])
    plt.show()   
    
    print('ang.shape', ang.shape)
    plt.imshow(ang)
    plt.title("ang")
    plt.colorbar()
    plt.axis([0, ang.shape[1], 0, ang.shape[0]])
    plt.show()    
               
    print('ang1.shape', ang1.shape)
    plt.imshow(ang1)
    plt.title("ang1")
    plt.colorbar()
    plt.axis([0, ang1.shape[1], 0, ang1.shape[0]])
    plt.show() 

    # Calculate the u and v components of velocity vector field
    u = vel*np.cos(ang)
    v = vel*np.sin(ang)
    u1 = vel*np.cos(ang1)
    v1 = vel*np.sin(ang1)   
    
    print('u.shape', u.shape)
    plt.imshow(u)
    plt.title("u")
    plt.colorbar()
    plt.axis([0, u.shape[1], 0, u.shape[0]])
    plt.show()    
    
    print('v.shape', v.shape)
    plt.imshow(v)
    plt.title("v")
    plt.colorbar()
    plt.axis([0, v.shape[1], 0, v.shape[0]])
    plt.show() 
    
    print('u1.shape', u1.shape)
    plt.imshow(u1)
    plt.title("u1")
    plt.colorbar()
    plt.axis([0, u1.shape[1], 0, u1.shape[0]])
    plt.show()    
    
    print('v1.shape', v1.shape)
    plt.imshow(v1)
    plt.title("v1")
    plt.colorbar()
    plt.axis([0, v1.shape[1], 0, v1.shape[0]])
    plt.show() 
    
    # Calculate the location of center of vortex in the uncropped, resized, unrotated velocity vector field with center of grid as origin
    bxc = bx*(200/lr) - 200/2
    byc = by*(200/ud) - 200/2
    
    # Calculate the location of center of vortex in the uncropped, resized, rotated velocity vector field with center of grid as origin
    bxc1 = bxc*np.cos(angle) + byc*np.sin(angle)
    byc1 = - bxc*np.sin(angle) + byc*np.cos(angle)
    
    # Calculate the location of center of vortex in the cropped, resized, unrotated velocity vector field with center of grid as origin
    bx = bxc1 + 200/2 - (200-l)/2
    by = byc1 + 200/2 - (200-l)/2
    
    # Concatenate u, u1, v, v1 so that the resulting u and v are not affected by discontinuities in angle (between 0 and 360 degree) in ang, ang1 array
    if direction == 'c':
        if 0 <= angle < np.pi*0.5 or angle == np.pi*2:
            u = np.concatenate((u[:, :int(bx)], u1[:, int(bx):]), axis = 1)
            v = np.concatenate((v[:, :int(bx)], v1[:, int(bx):]), axis = 1)
        elif np.pi*0.5 <= angle < np.pi:
            u = np.concatenate((u1[:int(by), :], u[int(by):, :]), axis = 0)
            v = np.concatenate((v1[:int(by), :], v[int(by):, :]), axis = 0)
        elif np.pi <= angle < np.pi*1.5:
            u = np.concatenate((u1[:, :int(bx)], u[:, int(bx):]), axis = 1)
            v = np.concatenate((v1[:, :int(bx)], v[:, int(bx):]), axis = 1)
        elif np.pi*1.5 <= angle < np.pi*2:
            u = np.concatenate((u[:int(by), :], u1[int(by):, :]), axis = 0)
            v = np.concatenate((v[:int(by), :], v1[int(by):, :]), axis = 0)
        else:
            print('angle: ', angle*180/np.pi, 'out of range!')
    
    elif direction == 'ac':
        if 0 <= angle < np.pi*0.5 or angle == np.pi*2:
            u = np.concatenate((u1[:, :int(bx)], u[:, int(bx):]), axis = 1)
            v = np.concatenate((v1[:, :int(bx)], v[:, int(bx):]), axis = 1)
        elif np.pi*0.5 <= angle < np.pi:
            u = np.concatenate((u[:int(by), :], u1[int(by):, :]), axis = 0)
            v = np.concatenate((v[:int(by), :], v1[int(by):, :]), axis = 0)
        elif np.pi <= angle < np.pi*1.5:
            u = np.concatenate((u[:, :int(bx)], u1[:, int(bx):]), axis = 1)
            v = np.concatenate((v[:, :int(bx)], v1[:, int(bx):]), axis = 1)
        elif np.pi*1.5 <= angle < np.pi*2:
            u = np.concatenate((u1[:int(by), :], u[int(by):, :]), axis = 0)
            v = np.concatenate((v1[:int(by), :], v[int(by):, :]), axis = 0)
        else:
            print('angle: ', angle*180/np.pi, 'out of range!')
        
    # Apply gaussian filter on u and v to smoothen any residual effect caused by the discontinuity in angle
    u = gaussian_filter(u, sigma=1)
    v = gaussian_filter(v, sigma=1)
    
    print('u.shape', u.shape)
    plt.imshow(u)
    plt.title("u")
    plt.colorbar()
    plt.axis([0, u.shape[1], 0, u.shape[0]])
    plt.show()    
    
    print('v.shape', v.shape)
    plt.imshow(v)
    plt.title("v")
    plt.colorbar()
    plt.axis([0, v.shape[1], 0, v.shape[0]])
    plt.show() 
    
    # Calculate the width, height and angle of vortex
    bw = 2*br*(200/lr)
    bh = 2*br*(200/ud)
    ba = angle

    # Find the radius of smallest circle that circumscribe the vortex
    br = max(bw,bh)/2
    return u, v, bx, by, bw, bh, br, ba


def generate_samples(num, l): 
    """Randomly generate samples with image data, neural network input data and unprocessed labels. 

    Parameters
    ----------
    num : int
        Number of samples to be generated
    
    l : float
        Size of the velocity vector field array. Eg, if l = 80 then shape of velocity vector field array is (80,80)
            
    Returns
    -------
    image : numpy array of size (num, 2, l, l)
        image contains the u and v array separately which can be plotted easily
        
    data : numpy array of size (num, l, l, 2)
        data is saved in the form that can be input directly into neural network
        
    label : numpy array of size (num, 9, 5)
        label contains data about 5 parameters of the 9 vortices (max): bx, by, bw, bh, ba
        where bx, by : location of center of vortex,
              bw, bh : width and height of vortex,
              ba : angle at which the vortex is rotated by
        
    """  
    # Create empty arrays for image, data and label data 
    image = []
    data = []
    label = []
    
    
    for i in range(num):
        # Randomly selects number of vortex to be present in each sample
        n = np.random.randint(1,4)
        n = 1
        # Set tol so that vortices won't be too close to one another
        tol = 5
            
        # Generate all parameters of up to 9 vortices (u, v, bx, by, bw, bh, br, ba) 
        # Regenerate vortex if the generated vortex is too close or overlap with other vortices
        if n == 0:
            u1, v1, bx1, by1, bw1, bh1, ba1 = (np.zeros((l,l)) + -2*np.random.rand() + 1), (np.zeros((l,l)) + -2*np.random.rand() + 1), 0, 0, 0, 0, 0
            u2, v2, bx2, by2, bw2, bh2, ba2 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            u3, v3, bx3, by3, bw3, bh3, ba3 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            u4, v4, bx4, by4, bw4, bh4, ba4 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            u5, v5, bx5, by5, bw5, bh5, ba5 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            u6, v6, bx6, by6, bw6, bh6, ba6 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            u7, v7, bx7, by7, bw7, bh7, ba7 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            u8, v8, bx8, by8, bw8, bh8, ba8 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            u9, v9, bx9, by9, bw9, bh9, ba9 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
        
        if n >= 1:
            u1, v1, bx1, by1, bw1, bh1, br1, ba1 = generate_random_vortex(l)
        else:
            u1, v1, bx1, by1, bw1, bh1, ba1 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
        
        if n >= 2:
            u2, v2, bx2, by2, bw2, bh2, br2, ba2 = generate_random_vortex(l)
            while np.hypot((bx2-bx1), (by2-by1)) < (br1 + br2 + tol):
                u2, v2, bx2, by2, bw2, bh2, br2, ba2 = generate_random_vortex(l)   
        else:
            u2, v2, bx2, by2, bw2, bh2, ba2 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            
        if n >= 3:
            u3, v3, bx3, by3, bw3, bh3, br3, ba3 = generate_random_vortex(l)
            while np.hypot((bx2-bx3), (by2-by3)) < (br2 + br3 + tol) or np.hypot((bx1-bx3), (by1-by3)) < (br1 + br3 + tol):
                u3, v3, bx3, by3, bw3, bh3, br3, ba3 = generate_random_vortex(l) 
        else:
            u3, v3, bx3, by3, bw3, bh3, ba3 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
        
        if n >= 4:
            u4, v4, bx4, by4, bw4, bh4, br4, ba4 = generate_random_vortex(l)  
            while np.hypot((bx2-bx4), (by2-by4)) < (br2 + br4 + tol) or np.hypot((bx1-bx4), (by1-by4)) < (br1 + br4 + tol) or np.hypot((bx3-bx4), (by3-by4)) < (br3 + br4 + tol):
                u4, v4, bx4, by4, bw4, bh4, br4, ba4  = generate_random_vortex(l)
        else:
            u4, v4, bx4, by4, bw4, bh4, ba4 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
     
        if n >= 5:
            u5, v5, bx5, by5, bw5, bh5, br5, ba5 = generate_random_vortex(l)  
            while np.hypot((bx2-bx5), (by2-by5)) < (br2 + br5 + tol) or np.hypot((bx1-bx5), (by1-by5)) < (br1 + br5 + tol) or np.hypot((bx3-bx5), (by3-by5)) < (br3 + br5 + tol) or np.hypot((bx4-bx5), (by4-by5)) < (br4 + br5 + tol):
                u5, v5, bx5, by5, bw5, bh5, br5, ba5 = generate_random_vortex(l) 
        else: 
            u5, v5, bx5, by5, bw5, bh5, ba5 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
        
        if n >= 6:
            u6, v6, bx6, by6, bw6, bh6, br6, ba6 = generate_random_vortex(l) 
            while np.hypot((bx2-bx6), (by2-by6)) < (br2 + br6 + tol) or np.hypot((bx1-bx6), (by1-by6)) < (br1 + br6 + tol) or np.hypot((bx3-bx6), (by3-by6)) < (br3 + br6 + tol) or np.hypot((bx4-bx6), (by4-by6)) < (br4 + br6 + tol) or np.hypot((bx5-bx6), (by5-by6)) < (br5 + br6 + tol):
                u6, v6, bx6, by6, bw6, bh6, br6, ba6 = generate_random_vortex(l)
        else: 
            u6, v6, bx6, by6, bw6, bh6, ba6 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
        
        if n >= 7:
            u7, v7, bx7, by7, bw7, bh7, br7, ba7 = generate_random_vortex(l)
            while np.hypot((bx2-bx7), (by2-by7)) < (br2 + br7 + tol) or np.hypot((bx1-bx7), (by1-by7)) < (br1 + br7 + tol) or np.hypot((bx3-bx7), (by3-by7)) < (br3 + br7 + tol) or np.hypot((bx4-bx7), (by4-by7)) < (br4 + br7 + tol) or np.hypot((bx5-bx7), (by5-by7)) < (br5 + br7 + tol) or np.hypot((bx6-bx7), (by6-by7)) < (br6 + br7 + tol):
                u7, v7, bx7, by7, bw7, bh7, br7, ba7 = generate_random_vortex(l)
        else: 
            u7, v7, bx7, by7, bw7, bh7, ba7 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            
        if n >= 8:
            u8, v8, bx8, by8, bw8, bh8, br8, ba8 = generate_random_vortex(l) 
            while np.hypot((bx2-bx8), (by2-by8)) < (br2 + br8 + tol) or np.hypot((bx1-bx8), (by1-by8)) < (br1 + br8 + tol) or np.hypot((bx3-bx8), (by3-by8)) < (br3 + br8 + tol) or np.hypot((bx4-bx8), (by4-by8)) < (br4 + br8 + tol) or np.hypot((bx5-bx8), (by5-by8)) < (br5 + br8 + tol) or np.hypot((bx6-bx8), (by6-by8)) < (br6 + br8 + tol) or np.hypot((bx7-bx8), (by7-by8)) < (br7 + br8 + tol):
                u8, v8, bx8, by8, bw8, bh8, br8, ba8 = generate_random_vortex(l)
        else: 
            u8, v8, bx8, by8, bw8, bh8, ba8 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
        
        if n >= 9:
            u9, v9, bx9, by9, bw9, bh9, br9, ba9 = generate_random_vortex(l) 
            while np.hypot((bx2-bx9), (by2-by9)) < (br2 + br9 + tol) or np.hypot((bx1-bx9), (by1-by9)) < (br1 + br9 + tol) or np.hypot((bx3-bx9), (by3-by9)) < (br3 + br9 + tol) or np.hypot((bx4-bx9), (by4-by9)) < (br4 + br9 + tol) or np.hypot((bx5-bx9), (by5-by9)) < (br5 + br9 + tol) or np.hypot((bx6-bx9), (by6-by9)) < (br6 + br9 + tol) or np.hypot((bx7-bx9), (by7-by9)) < (br7 + br9 + tol) or np.hypot((bx8-bx9), (by8-by9)) < (br8 + br9 + tol):
                u9, v9, bx9, by9, bw9, bh9, br9, ba9 = generate_random_vortex(l)
        else: 
            u9, v9, bx9, by9, bw9, bh9, ba9 = np.zeros((l,l)), np.zeros((l,l)), 0, 0, 0, 0, 0
            
        u = u1 + u2 + u3 + u4 + u5 + u6 + u7 + u8 + u9
        v = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9
    
        # Save image sample
        image.append((u,v))
        
        # Process and save neural network input data sample
        zeros = np.zeros((u.shape[0], u.shape[1], 2))
        for ii in range(u.shape[0]):
            for jj in range(u.shape[1]):
                zeros[ii,jj] = (u[ii,jj], v[ii,jj])
        data.append((zeros))
        
        # Save label data for each sample
        label.append(((bx1,by1,bw1,bh1,ba1), (bx2,by2,bw2,bh2,ba2), (bx3,by3,bw3,bh3,ba3), (bx4, by4, bw4, bh4,ba4), (bx5, by5, bw5, bh5,ba5), (bx6, by6, bw6, bh6,ba6), (bx7, by7, bw7, bh7,ba7), (bx8, by8, bw8, bh8,ba8), (bx9, by9, bw9, bh9,ba9)))
        
        # print number of sample
        print(i)
    
    # Convert lists of lists to Numpy array
    data = np.array(data)
    label = np.array(label)
    label = label.reshape(num, 9, 5)
    image = np.array(image)

    return image, data, label

def create_yolo_label(image, data, label, save_file_num, yolo_grid):
    """Convert labels to YOLO compatible form

    Parameters
    ----------
    image : numpy array of size (num, 2, l, l)
        image contains the u and v array separately which can be plotted easily
        
    data : numpy array of size (num, l, l, 2)
        data is saved in the form that can be input directly into neural network
        
    label : numpy array of size (num, 9, 5)
        label contains data about 5 parameters of the 9 vortices (max): bx, by, bw, bh, ba
        where bx, by : location of center of vortex,
              bw, bh : width and height of vortex,
              ba : angle at which the vortex is rotated by
              
    save_file_num : int
        Numbered label for the save file name. For eg, "'yolo_random_vortex_label_test3.npy' is the saved label file name where 3 is the save_file_num

    yolo_grid : Tuple of size (1,2)
        Determine the size of YOLO grid to be used in labelling

    Returns
    -------

    yolo_label : numpy array of size (num, yolo_grid_size, yolo_grid_size, 6)
        yolo_grid_size refers to the matrix size that the image is divided into for labelling (Refer to external documentation on how YOLO labelling works)
        Each element in the matrix contains 6 parameters: cp, bx, by, bw, bh, ba
        where cp : confidence probability (1 or 0) of whether a vortex is present in the grid cell
              bx, by : location of center of vortex,
              bw, bh : width and height of vortex,
              ba : angle at which the vortex is rotated by
    """  
    
    # Find the number of samples and the size of the velocity vector field
    num = image.shape[0]
    img_size = image.shape[2]
    
    # Set the grid size for YOLO labelling
    yolo_label = np.zeros((num, yolo_grid[0], yolo_grid[1], label.shape[2]+1))
    grid_size = img_size/yolo_grid[0], img_size/yolo_grid[1]
    
    # Convert every label into YOLO label form
    for img_num ,individual_label in enumerate(label):
        # Keep count of the number of label that has been processed
        print(img_num)
        for n,i in enumerate(individual_label):
            # Extract all parameters from original label file
            bx = i[0]
            by = i[1]
            bw = i[2]
            bh = i[3]
            ba = i[4]
            
            # Normalise parameters for every grid cell if a vortex is present in the grid cell and set parameters to zero if vortex is absent in the grid ell
            for ii in range(yolo_grid[0]):
                for jj in range(yolo_grid[1]):
                    # Check if vortex is present within a grid cell
                    if (ii*grid_size[0] < bx < (ii+1)*grid_size[0]) and (jj*grid_size[1] < by < (jj+1)*grid_size[1]):
                        # Normalise parameters
                        bx = (bx - ii*grid_size[0])/grid_size[0]
                        by = (by - jj*grid_size[0])/grid_size[0]
                        bw = bw/grid_size[0]
                        bh = bh/grid_size[0]
                        yolo_label[img_num, jj, ii] = (1, bx, by, bw, bh, ba)
                    elif yolo_label[img_num, jj, ii, 0] == 0:                    
                        # Set parameters to zero
                        yolo_label[img_num, jj, ii] = (0, 0, 0, 0, 0, 0)
                        
    # Save image, data and processed YOLO label data to .npy files                    
    np.save('Artificial data/yolo_random_vortex_image_test' + str(save_file_num) + '.npy', image)
    np.save('Artificial data/yolo_random_vortex_data_test' + str(save_file_num) + '.npy', data)
    np.save('Artificial data/yolo_random_vortex_label_test' + str(save_file_num) + '.npy', yolo_label)

    return yolo_label


def plot_samples(image, data, label, l, slice_interval):
    """Plot a Quiver plot with arrows indicating the magnitude and direction of velocity vector and plot u and v velocity vector field.

    Parameters
    ----------
    image : numpy array of size (num, 2, l, l)
        image contains the u and v array separately which can be plotted easily
        
    data : numpy array of size (num, l, l, 2)
        data is saved in the form that can be input directly into neural network
        
    label : numpy array of size (num, 9, 5)
        label contains data about 5 parameters of the 9 vortices (max): bx, by, bw, bh, ba
        where bx, by : location of center of vortex,
              bw, bh : width and height of vortex,
              ba : angle at which the vortex is rotated by
        
    l : float
        Size of the velocity vector field array. Eg, if l = 80 then shape of velocity vector field array is (80,80)
            
    slice_interval : int
        Slice interval for quiver plot
        Slicer index for smoother quiver plot
        General note: Adjust the slice interval and scale accordingly to get the required arrow size.
        Also, the units, and angles units are also responsible.

    Returns
    -------
    A quiver plot and two scalar plot
        
    """  
    
    # Use slice_interval to determine the number of data to be skipped from being plotted in an interval
    skip = (slice(None, None, slice_interval), slice(None, None, slice_interval))
    
    for n,i in enumerate(image):
        plt.figure()
        U = i[0]
        V = i[1]
        
        # Create a mesh field for Quiver plot
        x = np.linspace(0, l-1, l)
        y = np.linspace(0, l-1, l)
        X, Y = np.meshgrid(x, y)
    
        # Use list function to create a deep copy of label_angle
        label_copy = list(label[n])
        label_copy = np.array(label_copy)
        
        # Convert angle in label from radian to degree
        label_copy[:,-1] *= 180/np.pi
        print(label_copy)

        # Compute Velocity scalar field from U and V    
        vels = np.hypot(U, V)
        
        # Set up quiver plot
        Quiver = plt.quiver(X[skip], Y[skip],
                            U[skip], V[skip],
                            vels[skip],
                            units='height',
                            angles='uv',
                            scale=10,
                            pivot='mid',
                            color='blue',
                            )
        plt.title("Vortex")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.quiverkey(Quiver, 1.01, 1.01, 10, label='10m/s', labelcolor='blue', labelpos='N',
                           coordinates='axes')
        plt.colorbar(Quiver)
        
        # Add Bounding Boxes to the plot
        for box in label[n]:
            thickness = 2
            ax = plt.gca()
            r2 = patches.Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3], fc='none', color = 'r', lw = thickness)
            t2 = mpl.transforms.Affine2D().rotate_around(box[0], box[1], -box[4])
            r2.set_transform(t2 + ax.transData)
            ax.add_patch(r2)    
        
        plt.xticks()
        plt.yticks()
        plt.axis([0, l, 0, l])
        plt.grid()
        plt.show()
        
        # Set up U scalar plot
        # Add Bounding Boxes to the plot
        for box in label[n]:
            thickness = 2
            ax = plt.gca()
            r2 = patches.Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3], fc='none', color = 'r', lw = thickness)
            t2 = mpl.transforms.Affine2D().rotate_around(box[0], box[1], -box[4])
            r2.set_transform(t2 + ax.transData)
            ax.add_patch(r2)
        
        plt.imshow(U)
        plt.title("U")
        plt.colorbar()
        plt.axis([0, l, 0, l])
        plt.show()
        
        # Set up V scalar plot
        # Add Bounding Boxes to the plot       
        for box in label[n]:
            thickness = 2
            ax = plt.gca()
            r2 = patches.Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3], fc='none', color = 'r', lw = thickness)
            t2 = mpl.transforms.Affine2D().rotate_around(box[0], box[1], -box[4])
            r2.set_transform(t2 + ax.transData)
            ax.add_patch(r2)
        
        plt.imshow(V)
        plt.title("V")
        plt.colorbar()
        plt.axis([0, l, 0, l])
        plt.show()
        
    
def main(l, num, save_file_num, yolo_grid):      
    """Generate and plot random vortex samples and output image, data, label that are compatible with neural network requirement.
    
    Parameters
    ----------        
    l : float
        Size of the velocity vector field array. Eg, if l = 80 then shape of velocity vector field array is (80,80)
            
    num : int
        Number of samples to be generated
                
    save_file_num : int
        Numbered label for the save file name. For eg, "'yolo_random_vortex_label3.npy' is the saved label file name where 3 is the save_file_num

    yolo_grid : tuple
        Size of yolo grid used in labelling. Eg, (10,10)
    """  
    
    # Generate and plot vortex samples
    image, data, label = generate_samples(num, l)
    plot_samples(image, data, label, l, slice_interval = 3)
    
    # Preprocess label into YOLO compatible form to be fed into neural network
    create_yolo_label(image, data, label, save_file_num, yolo_grid)



main(l = 80, num = 1, save_file_num = 5, yolo_grid = (10,10))