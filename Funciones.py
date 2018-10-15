# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:42:07 2018

@author: Nico
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

def Senoidal_Generator(fs, f0, N, a0=1, p0=0, plot=0): 
    """
    Quiero un gen de señales senoidales. Necesito:
    f0: frecuencia
    a0: amplitud
    p0: fase
    N: n de muestras
    fs: f de muestreo
    Devuelve:
    x: Tiempo
    y: Amplitud
    """
    ts=1/fs
    
    x = np.linspace(0,(N-1)*ts,N).flatten() #linspace(start,stop,total)
    x = x.reshape(N,1)

    y = a0 * np.sin(x*2*np.pi*f0 + p0).flatten()
    y = y.reshape(N,1)

    if plot!=0:
        #plt.ylim([-2, 1])
        plt.figure(plot)
        plt.grid(True)
        plt.plot(x,y)
        plt.title('Señal Senoidal')
        plt.xlabel('Tiempo [segundos]')
        plt.ylabel('Amplitud [V]')  
    return x,y

def Noise_Generator(mu, varianza ,N ,Ts=1 ,plot=0):
    noise = np.random.normal(mu, np.sqrt(varianza), N)
    n = np.linspace(0,(N-1)*Ts,N)
    if plot!=0:
        plt.figure(plot)
        plt.grid(True)
        plt.plot(n,noise)
        plt.title('Ruido gaussiano')
        plt.xlabel('Tiempo [segundos]')
        plt.ylabel('Amplitud [V]')
    return noise,n

def Cuadrada_Generator(fs,f0,N,D,a0=1,plot=0):
    Ts=1/fs
    T0=1/f0
    N1=D*T0*N
    N2=T0*N-N1
    Np=int(N1+N2)
    on = (np.zeros(int(N1)))+a0
    off = np.zeros(int(N2))
    #sq = np.concatenate(on,off)
    sq = np.hstack((on,off))
    a=N/Np
    if a>1:
        señal = sq[0:N-1]
    else:
        señal=sq
        for a in range(int(a)):
            señal = np.hstack((señal,sq))
    n = np.linspace(0,(N-1)*Ts,N)
    if plot!=0: 
        plt.figure(plot)
        plt.grid(True)
        plt.plot(n,señal)
        plt.title('Señal Cuadrada')
        plt.xlabel('Tiempo [segundos]')
        plt.ylabel('Amplitud [V]')
    return sq,n

def Analizador_de_Espectro(señal, N, Ts, plot=0):
    freq = np.fft.fftfreq(N, Ts)
    asd = np.array(señal)
    señal=asd.reshape(1,N)
#    espectro = np.fft.fft(señal)/(N/2)
    espectro = fft(señal)#/(N/2)
    if plot!=0:
        modulo=np.abs(espectro/(N/2))
        modulo=modulo.reshape(N,1)
        fase=np.angle(espectro)
        fase=fase.reshape(N,1)
        fin=(N/2)-1
        plt.figure(plot)
        plt.subplot(211)
        plt.stem(freq[0:int(fin)], modulo[0:int(fin)])#'ro'
        plt.title('Módulo del Espectro')
        plt.xlabel('Bins-$\Delta$f [Hz]')
        plt.ylabel('Amplitud [V]')
        plt.grid(True)
        plt.subplot(212)
        plt.stem(freq[0:int(fin)], fase[0:int(fin)])
        plt.title('Fase del Espectro')
        plt.xlabel('Bins-$\Delta$f [Hz]')
        plt.ylabel('Radianes [rd]')
        plt.grid(True)
    return espectro,freq

def PrintModule(freq,espectro,beg,fin,x,y,label,tipo="stem",scale="lin"):
    N=len(espectro.T)
    modulo=np.abs(espectro/(N/2))
    modulo=modulo.reshape(N,1)
    plt.figure(figsize=(x,y))
    if scale == "log":
        modulo = 20*np.log10(modulo)
        plt.ylabel('Amplitud [dB]')
    else:
        plt.ylabel('Amplitud [V]')
    if tipo == "plot":
        plt.plot(freq[int(beg):int(fin)], modulo[int(beg):int(fin)])
    else:
        plt.stem(freq[int(beg):int(fin)], modulo[int(beg):int(fin)])
    plt.title('Módulo del Espectro - '+label)
    plt.xlabel('Frecuencia [Hz]')
    
    
def PrintPhase(freq,espectro,beg,fin,x,y,label,tipo="stem"):
    N=len(espectro.T)
    fase=np.angle(espectro/(N/2))
    fase=fase.reshape(N,1)
    plt.figure(figsize=(x,y))
    if tipo == "plot":
        plt.plot(freq[int(beg):int(fin)], fase[int(beg):int(fin)])
    else:
        plt.stem(freq[int(beg):int(fin)], fase[int(beg):int(fin)])
    plt.title('Fase del Espectro - '+label)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')

def quant(señal,n):
    señal *= 0.9
    dq = (2**(n-1))-1
    Q = señal * dq
    Q = np.around(Q)

    return Q

def Energiaf(E,N):
    R = int(N/2)
    E = E.reshape(N,1)
    A = 0
    for i in range(R):
        A += (np.abs(E[i]))**2
    return A

def Ef0(x,y,N):
    R = int(N/2)
    A = 0
    M = 0
    x = x.reshape(N,1)
    y = y.reshape(N,1)
    for i in range(R):
        A += x[i] * np.abs(y[i])
        M += np.abs(y[i])
    return (A/M)

def ArgMax(x,y,N):
    R = int(N/2)
    y = abs( y.reshape(N,1) )
    E = []
    for i in range(R):
        E.append( (y[i])**2 )
    pos = np.argmax(E)
    return (x[pos])

def PrintArb(x,y,xlabel,ylabel,Title):
    plt.figure()
    plt.grid(True)
    plt.plot(x,y)
    plt.title(Title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def cleanS(E,N):
    E = E.reshape(N,1)
    for i in range(N):
        if abs(E[i]) < 1e-12:
            E[i] = 0
    return E.reshape(1,N)

def Mean(data,N):
    avg=0
    for i in range(N):
        avg += data[i]
    avg /= N
    return avg

def Varianza(data,Mean,N):
    V=0
    for i in range(N):
        V += ((data[i]-Mean)**2)
    V = V/(N-1)
    return V

def Var_est(data,Mean,N):
    V=0
    for i in range(N):
        V += ((data[i]-Mean)**2)
    V = V/(N)
    return V

def RMS_est(data,N):
    data = data.reshape(N,1)
    Om0 = (np.pi)/2
    Fac= (1000/4) / ((np.pi)/2)
    a = int( round( (Om0 - (2 * 2 * np.pi)/N)*Fac ) )
    b = int( round( (Om0 + (2 * 2 * np.pi)/N)*Fac ) )
    aux = 0
    for i in range(a,b+1):
        aux += (data[i]/(N/2))**2
    aux = (aux/5)**(0.5)
    return aux

def LogAvoidZero(E,N):
    for i in range(N):
        if E[0][i] == 0:
            E[0][i] = 1e-12
    return E
