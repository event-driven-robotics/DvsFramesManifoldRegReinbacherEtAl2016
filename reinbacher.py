# -*- coding: utf-8 -*-
"""
Copyright (C) 2020 - Event-driven Perception for Robotics
Authors: Sim Bamford
		Code is modified from https://github.com/VLOGroup/dvs-reconstruction 
This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>."""

import numpy as np

def TVL1_dual_kernel(p, tex_u, lambdaaa, sigma, width, height):
    gradX = np.diff(tex_u, n=1, axis=0, append=tex_u[np.newaxis, -1, :, :])
    gradY = np.diff(tex_u, n=1, axis=1, append=tex_u[:, np.newaxis, -1, :])
    grad = np.concatenate((gradX, gradY), axis=2)
    p = p + sigma*grad
    p = p / np.maximum(1, lambdaaa * np.linalg.norm(p, axis=2)[:,:,np.newaxis]) 
    return p

# No need to pass in g_u_
def TVL1_primal_kernel(g_u, g_u_, g_f, tex_p, tau, width, height):
    g_u_ = g_u.copy()
    ktp = np.diff(tex_p[:, :, 0, np.newaxis], n=1, axis=0, 
                                  prepend=tex_p[np.newaxis, 0, :, 0, np.newaxis]) + \
            np.diff(tex_p[:, :, 1, np.newaxis], n=1, axis=1, 
                                    prepend=tex_p[:, np.newaxis, 0, 1, np.newaxis])
    g_u = g_u + tau * ktp
    g_u = g_f + np.maximum(0, np.abs(g_u-g_f)-tau)*np.sign(g_u-g_f)
    g_u_ = 2*g_u - g_u_
    return g_u, g_u_

# No need to pass in g_m
def Prepare_manifold_kernel(g_m, t_t, lambda_time, width, height):
    tx = np.diff(t_t, n=1, axis=0, append=t_t[np.newaxis, -1, :])
    ty = np.diff(t_t, n=1, axis=1, append=t_t[:, np.newaxis, -1])
    tx = np.clip(tx[:, :, 0],-0.5,0.5)*lambda_time
    ty = np.clip(ty[:, :, 0],-0.5,0.5)*lambda_time
    g_m = np.zeros((width, height, 4), np.float64)
    g_m[:, :, 3] = 1+tx*tx+ty*ty
    g_m[:, :, 0] = tx
    g_m[:, :, 1] = ty
    g_m[:, :, 2] = t_t[:, :, 0] # maybe we need it
    return g_m

def getCoefficients(m):
    c = np.zeros((m.shape[0], m.shape[1], 3), np.float64)
    c[:, :, 0] = (1+m[:, :, 1]**2)/m[:, :, 3]
    c[:, :, 2] = (1+m[:, :, 0]**2)/m[:, :, 3] # Yes order is swapped
    c[:, :, 1] = m[:, :, 0]*m[:, :, 1]/m[:, :, 3]
    return c

def TV_manifold_dual_kernel(p, m, tex_u, lambdaaa, sigma, width, height):
    grad = np.zeros((width, height, 2), np.float64)
    grad[:, :, 0] = np.diff(tex_u[:, :, 0], n=1, axis=0, append=tex_u[np.newaxis, -1, :, 0])
    grad[:, :, 1] = np.diff(tex_u[:, :, 0], n=1, axis=1, append=tex_u[:, np.newaxis, -1, 0])
    c = getCoefficients(m)
    det = np.sqrt(m[:, :, 3])
    p_upd = np.zeros((width, height, 4), np.float64)
    p_upd[:, :, 0] = grad[:, :, 0]*c[:, :, 0] - grad[:, :, 1]*c[:, :, 1]
    p_upd[:, :, 1] = grad[:, :, 1]*c[:, :, 2] - grad[:, :, 0]*c[:, :, 1]
    p_upd[:, :, 2] = (m[:, :, 0]*grad[:, :, 0] + m[:, :, 1]*grad[:, :, 1])/det    
    p = p + sigma*p_upd
    p = p / (1 + sigma*0.3) # Huberize
    p = p / np.maximum(1, np.sqrt(p[:, :, 0]**2+p[:, :, 1]**2+p[:, :, 2]**2)/det*lambdaaa)[:,:,np.newaxis]
    return p

def edgedShiftXBack(array):
    return np.concatenate((array[0, np.newaxis,:,:], array[:-1,:,:]), axis=0)

def edgedShiftYBack(array):
    return np.concatenate((array[:,0, np.newaxis,:], array[:,:-1,:]), axis=1)

def computeKTP(tex_m, tex_p):
    m_xm1y = edgedShiftXBack(tex_m)
    m_xym1 = edgedShiftYBack(tex_m)
    c_xy = getCoefficients(tex_m)
    c_xm1y = getCoefficients(m_xm1y)
    c_xym1 = getCoefficients(m_xym1)
    pXBack = edgedShiftXBack(tex_p)
    pYBack = edgedShiftYBack(tex_p)    
    ktp = (tex_p[:,:,0] * (-c_xy[:,:,0] + c_xy[:,:,1]) + \
             pXBack[:,:,0] * c_xm1y[:,:,0] - \
             pYBack[:,:,0] * c_xym1[:,:,1]) + \
            (tex_p[:,:,1] * (-c_xy[:,:,2] + c_xy[:,:,1]) + \
             pYBack[:,:,1] * c_xym1[:,:,2] - \
             pXBack[:,:,1] * c_xm1y[:,:,1]) + \
            (tex_p[:,:,2] * (-tex_m[:,:,0]/tex_m[:,:,3] -tex_m[:,:,1]/tex_m[:,:,3]) + \
             pXBack[:,:,2] * m_xm1y[:,:,0]/m_xm1y[:,:,3] + \
             pYBack[:,:,2] * m_xym1[:,:,0]/m_xym1[:,:,3])
    weight = tex_m[:,:,3]
    return ktp, weight

def TVKLD_manifold_primal_kernel(g_u, g_u_, tex_p, g_f, tex_m, tau, u_min, u_max, width, height):
    g_u_ = g_u.copy()
    ktp,weight = computeKTP(tex_m,tex_p)
    g_u = g_u - tau*ktp[:, :, np.newaxis]    # primal update
    tau = tau * np.sqrt(weight[:,:,np.newaxis])
    g_u = np.clip((g_u-tau+np.sqrt((g_u-tau)**2+4*tau*g_f))/2, u_min, u_max) # KLD
    g_u_ = 2*g_u-g_u_
    return g_u, g_u_
    
# TODO: no need to pass in u - it gets copied from f
def solveTVIncrementalManifold(u, f, t, 
                               pt, manifold_t, p_manifold,
                                            lambdaaa, lambda_t, iterations, 
                                            iterations_t, u_min, u_max,
                                            width, height, tauManifold):
    L = np.sqrt(8)
    tau = 1/L
    sigma = 1/L
    # first denoise the time volume
    ft = t.copy()
    ut_ = t.copy()
    for k in range(iterations_t):
        pt = TVL1_dual_kernel(pt, ut_, 2, sigma, width, height)
        t, ut_ = TVL1_primal_kernel(t, ut_, ft, pt, tau, width, height)
    # prepare manifold
    manifold_t = Prepare_manifold_kernel(manifold_t, t, lambda_t, width, height)
    L = np.sqrt(8+2*np.sqrt(2))
    tau = tauManifold
    sigma = 1/tau/L/L
    # And now the intensity image
    u = f.copy()
    u_ = u.copy()
    for k in range(iterations):
        p_manifold = TV_manifold_dual_kernel(p_manifold, manifold_t, u_, lambdaaa, sigma, width, height)
        # TODO: Here you could implement an alternative method ...
        u, u_ = TVKLD_manifold_primal_kernel(u, u_, p_manifold, f, manifold_t, tau, u_min, u_max, width, height)
    return u

def reinbacher(dvs, **kwargs):
    stopTime = kwargs.get('stopTime', dvs['ts'][-1])
    eventsPerImage = kwargs.get('eventsPerImage', 1000)
    lambdaaa = kwargs.get('lambda', 180.0) # 'lambda' is reserved word
    lambda_t = kwargs.get('lambda_t', 2.0)
    u_min = kwargs.get('u_min', 1.0)
    u_max = kwargs.get('u_max', 2.0)
    c1 = kwargs.get('c1', 1.15)
    c2 = kwargs.get('c2', 1.25)
    tau = kwargs.get('tau', 0.01)
    iterations = kwargs.get('iterations', 50)
    iterations_t = kwargs.get('iterationsTime', iterations)
    #method = 'TV_KLD' # options are: TV_L1;TV_L2; TV_LogL2; - only implemented the default
    # Dimensions come from the DVS container - they can be overridden with kwargs
    width = kwargs.get('width', dvs.get('dimX', 304))
    height = kwargs.get('height', dvs.get('dimY', 240))
    
    #containers for results
    frames = []
    frameTs = []        
    # The following are top-level arrays
    # 'input' is built-in function
    inputtt = np.ones((width, height, 1), dtype=np.float64) * (u_min + u_max) / 2 
    manifold = np.zeros((width, height, 1), dtype=np.float64)
    output = np.ones((width, height, 1), dtype=np.float64) * (u_min + u_max) / 2
    # TODO: there's probably some speed wins here if these arrays and others could be recycled
    pt = np.zeros((width, height, 2), dtype=np.float64)
    manifold_t = np.zeros((width, height, 4), dtype=np.float64)
    p_manifold = np.zeros((width, height, 4), dtype=np.float64)
    
    # Iterate over event packets
    latestTs = 0
    latestIdx = 0
    while latestTs < stopTime:
        inputtt = output 
        #Set events
        try:
            for eventIdx in range(latestIdx, latestIdx + eventsPerImage):
                # latest timestamps to the manifold
                manifold[dvs['x'][eventIdx], dvs['y'][eventIdx]] = dvs['ts'][eventIdx]
                # skew the inputtt with the polarity and C1/2 threshold intensity factors
                if dvs['pol'][eventIdx]: # True
                    inputtt[dvs['x'][eventIdx], dvs['y'][eventIdx]] = \
                        inputtt[dvs['x'][eventIdx], dvs['y'][eventIdx]] * c1
                else:
                    inputtt[dvs['x'][eventIdx], dvs['y'][eventIdx]] = \
                        inputtt[dvs['x'][eventIdx], dvs['y'][eventIdx]] / c2
            
            latestIdx += eventsPerImage
            latestTs = dvs['ts'][eventIdx]
        except IndexError: # End of data
            break
        print('Latest timestamp: ' + str(latestTs))
        output = solveTVIncrementalManifold(output, inputtt, manifold,
                                            pt, manifold_t, p_manifold,
                                            lambdaaa, lambda_t, iterations, 
                                            iterations_t, u_min, u_max, 
                                            width, height, tau)
        # reshape and scale reconstructed frame, converting to 8 bit greyscale
        outputFrame = output.copy()[:, :, 0].T
        outputFrame = (outputFrame - u_min) / (u_max-u_min) * 255
        frames.append(outputFrame.astype(np.uint8))
    
        frameTs.append(latestTs)
    framesDict = {'frames': frames,
             'ts': np.array(frameTs)}
    return framesDict


