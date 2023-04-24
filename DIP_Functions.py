import cv2, imutils
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.util import random_noise as imnoise
from scipy import signal,ndimage
from skimage import measure

def HistEq(pin):
        hsv = cv2.cvtColor(pin,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        hist,bins = np.histogram(v.flatten(),256,[0,255])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        v = cdf[v]
        final_hsv = cv2.merge((h,s,v))
        pout = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
        return pout

def MAVG(pin, wsize):
        mask = np.ones((wsize,wsize))/(wsize*wsize)
        #pout = filter2(pin,mask)
        pout = cv2.filter2D(pin,-1,mask)
        return pout

def SobelX(pin, wsize):
        mask = np.ndarray([[-1,0,1],[-2,0,2],[-1,0,1]])
        #pout = filter2(pin,mask)
        pout = cv2.filter2D(pin,-1,mask)
        return pout

def GaussianFilter(pin, sigma):
        if (sigma>0):
                gmask = GaussianMask(sigma,math.ceil(2*sigma)+1, math.ceil(2*sigma)+1)
                #pout = filter2(pin,gmask)
                pout = cv2.filter2D(pin,-1,gmask)
        else:
                pout = pin   
        return pout

def GaussianMask(sigma,nrow,ncol):
        g = np.zeros((nrow, ncol))
        rcenter = math.floor(nrow/2)+1
        ccenter = math.floor(ncol/2)+1
        pi = np.arctan(1)*4
        s = 0
        for i in range(1,nrow):
                for j in range(1,ncol):
                        g[i][j] = math.exp(-(pow(i-rcenter,2)+pow(j-ccenter,2))/(2*pow(sigma,2)))/(2*pi*pow(sigma,2))
                        s = s + g[i][j]
        g = g/s
        return g

def filter2(img1, mask):
        r,g,b=cv2.split(img1)
        r = signal.convolve2d(r, mask,mode='same')
        g = signal.convolve2d(g, mask,mode='same')
        b = signal.convolve2d(b, mask,mode='same')
        pout = cv2.merge((r,g,b))
        pout = pout.astype(np.uint8)
        return pout

def imgradient(pin,xmask,ymask):
        r,g,b=cv2.split(pin)
        rx = signal.convolve2d(r, xmask,mode='same')
        gx = signal.convolve2d(g, xmask,mode='same')
        bx = signal.convolve2d(b, xmask,mode='same')
        px = cv2.merge((rx,gx,bx))
        ry = signal.convolve2d(r, ymask,mode='same')
        gy = signal.convolve2d(g, ymask,mode='same')
        by = signal.convolve2d(b, ymask,mode='same')
        py = cv2.merge((ry,gy,by))
        pout = np.sqrt(px**2+py**2)
        pout = mat2gray(pout)
        pout = pout*255
        pout = pout.astype(np.uint8)
        return pout

def LaplacianSharpening(img1, k):
        mask = np.array([[1,1,1],[1,-8,1],[1,1,1]],dtype=np.float64)
        mask = mask
        r,g,b=cv2.split(img1)
        r = r-k*signal.convolve2d(r, mask,mode='same')
        g = g-k*signal.convolve2d(g, mask,mode='same')
        b = b-k*signal.convolve2d(b, mask,mode='same')
        pout = cv2.merge((r,g,b))
        pout = matclip(pout,0,255)
        pout = pout.astype(np.uint8)
        return pout

def PowerLaw(pin, gamma):
        hsv = cv2.cvtColor(pin,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        v = v/255.0
        v = v**gamma
        v = v*255.0
        v = v.astype(np.uint8)
        final_hsv = cv2.merge((h,s,v))
        pout = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
        return pout

def LinearContrast(pin,lt,rt,bt,tp):
        hsv = cv2.cvtColor(pin,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        v = v.astype(np.double)
        vout = bt + (v-lt)/(rt-lt)*(tp-bt)
        vout[v<lt] = bt
        vout[v>=rt] = tp
        v = vout.astype(np.uint8)
        final_hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
        return img

def im2double(img):
    return cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    return cv2.normalize(A, out, 0.0, 1.0, cv2.NORM_MINMAX)

def matclip(img,low,high):
    v = np.double(img)
    v[v<low]=low
    v[v>high]=high
    return v
   
def AddGaussianNoise(pin, sd, mn, vr):
    hsv = cv2.cvtColor(pin,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    vout = imnoise(v, mode='gaussian', seed=sd, mean = mn, var=vr)
    v = vout*255
    v = v.astype(np.uint8)
    final_hsv = cv2.merge((h,s,v))
    pout = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return pout

def AddSaltandPepperNoise(pin, sd, amnt):
    hsv = cv2.cvtColor(pin,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    vout = imnoise(v, mode='s&p', seed = sd, amount=amnt)
    v = vout*255
    v = v.astype(np.uint8)
    final_hsv = cv2.merge((h,s,v))
    pout = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return pout

def FreqFiltering(pin,filter):
    r,g,b=cv2.split(pin)
    rf = np.fft.fftshift(np.fft.fft2(r))
    r = np.real(np.fft.ifft2(np.fft.ifftshift(rf*filter)))
    gf = np.fft.fftshift(np.fft.fft2(g))
    g = np.real(np.fft.ifft2(np.fft.ifftshift(gf*filter)))
    bf = np.fft.fftshift(np.fft.fft2(b))
    b = np.real(np.fft.ifft2(np.fft.ifftshift(bf*filter)))
    pout = cv2.merge((r,g,b))
    pout = mat2gray(pout)
    pout = pout*255
    pout = pout.astype(np.uint8)
    return pout

def ILPF(d0,nrow,ncol): #Ideal LPF
    LPF = np.zeros((nrow, ncol))
    rcenter = math.floor(nrow/2)+1; #N/2
    ccenter = math.floor(ncol/2)+1; #M/2
    for i in range(1,nrow):
        for j in range(1,ncol):
            Duv = np.sqrt(pow(i-rcenter,2)+pow(j-ccenter,2))
            if Duv <= d0:
                LPF[i][j] = 1.0
            else:
                LPF[i][j] = 0.0
    return LPF


def IHPF(d0,nrow,ncol): #Ideal HPF
    LPF = np.zeros((nrow, ncol))
    rcenter = math.floor(nrow/2)+1; #N/2
    ccenter = math.floor(ncol/2)+1; #M/2
    for i in range(1,nrow):
        for j in range(1,ncol):
            Duv = np.sqrt(pow(i-rcenter,2)+pow(j-ccenter,2))
            if Duv <= d0:
                LPF[i][j] = 0.0
            else:
                LPF[i][j] = 1.0
    return LPF

def BLPF(d0, N, nrow, ncol):  #Butterworth LPF
    b = np.zeros((nrow, ncol))
    rcenter = math.floor(nrow/2)+1; #N/2
    ccenter = math.floor(ncol/2)+1; #M/2
    for i in range(1,nrow):
        for j in range(1,ncol):
                d = (i-rcenter)**2+(j-ccenter)**2
                b[i][j] = 1/(1+(d/(d0**2))**N)
    return b

def BHPF(d0, N, nrow, ncol):  #Butterworth HPF
    b = np.zeros((nrow, ncol))
    rcenter = math.floor(nrow/2)+1; #N/2
    ccenter = math.floor(ncol/2)+1; #M/2
    for i in range(1,nrow):
        for j in range(1,ncol):
                d = (i-rcenter)**2+(j-ccenter)**2
                if d>0:
                    b[i][j] = 1/(1+((d0**2)/d)**N)
                else:
                    b[i][j] = 0
    return b

def GLPF(sigma,nrow,ncol): #Gaussian LPF
    g = np.zeros((nrow, ncol))
    rcenter = math.floor(nrow/2)+1; #N/2
    ccenter = math.floor(ncol/2)+1; #M/2
    for i in range(1,nrow):
        for j in range(1,ncol):
            g[i][j] = math.exp(-(pow(i-rcenter,2)+pow(j-ccenter,2))/(2*pow(sigma,2)))
    return g

def GHPF(sigma,nrow,ncol): #Gaussian HPF
    g = np.zeros((nrow, ncol))
    rcenter = math.floor(nrow/2)+1; #N/2
    ccenter = math.floor(ncol/2)+1; #M/2
    for i in range(1,nrow):
        for j in range(1,ncol):
            g[i][j] = 1-math.exp(-(pow(i-rcenter,2)+pow(j-ccenter,2))/(2*pow(sigma,2)))
    return g

def HBF(A,sigma,nrow,ncol): #High Boost Filter
    H = np.zeros((nrow, ncol))
    rcenter = math.floor(nrow/2)+1; #N/2
    ccenter = math.floor(ncol/2)+1; #M/2
    for i in range(1,nrow):
        for j in range(1,ncol):
            H[i][j] = A-math.exp(-(pow(i-rcenter,2)+pow(j-ccenter,2))/(2*pow(sigma,2)))
    return H


def findEdge(pin,nrow,ncol):
    g = np.ones((nrow, ncol))
    for i in range(0,nrow-2):
        for j in range(0,ncol-2):
            if ((np.abs(pin[i][j]-pin[i][j+1])>=1).any()):
                g[i][j] = 0
            
    for i in range(0,nrow-2):
        for j in range(0,ncol-2):
            if ((np.abs(pin[i][j]-pin[i+1][j])>=1).any()):
                g[i][j] = 0

    return g
    
def findMask(pin):
    hsv = cv2.cvtColor(pin,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    pout = (h<0.9)*(s>0.5)
    return pout


def lapZ(pin):
    for i, v1 in enumerate(pin):
        for j, v2 in enumerate(v1):
            if v2 == 255:
                pin[i, j] = 1
    return pin

def binaryImg(pin):
    img = pin
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v1=cv2.split(hsv)
    # v = v.astype(np.double)
    mask = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=np.float64)
    v = signal.convolve2d(v1, mask,mode='same')
    # print(v) //showmatrix
    O = (v >= 0) * 1
    I = v
    O = O*255
    return O,I

def binaryImgGaussian(pin):
    img = pin
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v1=cv2.split(hsv)
    # v = v.astype(np.double)
    mask = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],dtype=np.float64)
    v = signal.convolve2d(v1, mask,mode='same')
    # print(v) //showmatrix
    O = (v >= 0) * 1
    I = v
    O = O*255
    return O,I

def binaryImgGaussian2(pin):
    img = pin
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v1=cv2.split(hsv)
    # v = v.astype(np.double)
    mask = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=np.float64)
    v = signal.convolve2d(v1, mask,mode='same')
    # print(v) //showmatrix
    O = (v >= 0) * 1
    I = v
    O = O*255
    return O,I

def binaryImgNoMask(pin):
    img = pin
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v1=cv2.split(hsv)
    O = (v1 >= 0) * 1
    I = v1
    O = O*255
    return I,O

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def countM(pin):
    propsa = measure.regionprops(pin)
    length = len(propsa)
    return length

def particleCount(pin):
    labelarray, particle_count = ndimage.measurements.label(pin)
    return  particle_count