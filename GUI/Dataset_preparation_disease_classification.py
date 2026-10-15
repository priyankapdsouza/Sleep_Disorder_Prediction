# Dataset preparation for healthy-unhealthy classification & disease classification

import csv
import pandas as pd
import scipy.io
import numpy as np
import h5py
import os

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load data from local dataa/Unbalanced Data/ folder (loading .csv files)
def load_csv_data(filename, n_samples=1024):
    """
    Load data from a CSV file.
    Assumes the CSV has 1025 columns: 1024 features + phase.
    Returns array with shape (n_rows, 1025).
    """
    file_path = os.path.join(script_dir, 'dataa', 'Unbalanced Data', filename)
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist.")
    data = np.loadtxt(file_path, delimiter=',')
    return data

healthy = load_csv_data('healthy.csv')
ins = load_csv_data('ins.csv')
nfle = load_csv_data('nfle.csv')
narco = load_csv_data('narco.csv')
rbd = load_csv_data('rbd.csv')
plm = load_csv_data('plm.csv')
sdb = load_csv_data('sdb.csv')

# Healthy-unhealthy dataset
# Both phases

n1=np.shape(healthy)[0]
n = (n1//10)*10   # 2 phases of 5 disorders each = 10
n11=int(n/2)
n12=int(n11/3)
print('samples of healthy subjects:', n)
n2=int(n/5)
n3=int(n2/2)
n4=int(n3/3)
print('samples of each disorders:', n2)
print('samples of unhealthy subjects:', n)
total=int(2*n)
print('Total samples:', total)

data=np.zeros([total, 1026])

# healthy
i=0
rows=np.where(healthy[:,1024]==0)
data[i:i+n11,0:1025]=healthy[rows][0:n11,:]
i+=n11
rows=np.where(healthy[:,1024]==1)
data[i:i+n12,0:1025]=healthy[rows][0:n12,:]
i+=n12
rows=np.where(healthy[:,1024]==2)
data[i:i+n12,0:1025]=healthy[rows][0:n12,:]
i+=n12
rows=np.where(healthy[:,1024]==3)
data[i:i+n12,0:1025]=healthy[rows][0:n12,:]
i+=n12

# insomnia
rows=np.where(ins[:,1024]==0)
data[i:i+n3,0:1025]=ins[rows][0:n3,:]
i+=n3
rows=np.where(ins[:,1024]==1)
data[i:i+n4,0:1025]=ins[rows][0:n4,:]
i+=n4
rows=np.where(ins[:,1024]==2)
data[i:i+n4,0:1025]=ins[rows][0:n4,:]
i+=n4
rows=np.where(ins[:,1024]==3)
data[i:i+n4,0:1025]=ins[rows][0:n4,:]
i+=n4

# nfle
rows=np.where(nfle[:,1024]==0)
data[i:i+n3,0:1025]=nfle[rows][0:n3,:]
i+=n3
rows=np.where(nfle[:,1024]==1)
data[i:i+n4,0:1025]=nfle[rows][0:n4,:]
i+=n4
rows=np.where(nfle[:,1024]==2)
data[i:i+n4,0:1025]=nfle[rows][0:n4,:]
i+=n4
rows=np.where(nfle[:,1024]==3)
data[i:i+n4,0:1025]=nfle[rows][0:n4,:]
i+=n4

# narcolepsy
rows=np.where(narco[:,1024]==0)
data[i:i+n3,0:1025]=narco[rows][0:n3,:]
i+=n3
rows=np.where(narco[:,1024]==1)
data[i:i+n4,0:1025]=narco[rows][0:n4,:]
i+=n4
rows=np.where(narco[:,1024]==2)
data[i:i+n4,0:1025]=narco[rows][0:n4,:]
i+=n4
rows=np.where(narco[:,1024]==3)
data[i:i+n4,0:1025]=narco[rows][0:n4,:]
i+=n4

# rbd
rows=np.where(rbd[:,1024]==0)
data[i:i+n3,0:1025]=rbd[rows][0:n3,:]
i+=n3
rows=np.where(rbd[:,1024]==1)
data[i:i+n4,0:1025]=rbd[rows][0:n4,:]
i+=n4
rows=np.where(rbd[:,1024]==2)
data[i:i+n4,0:1025]=rbd[rows][0:n4,:]
i+=n4
rows=np.where(rbd[:,1024]==3)
data[i:i+n4,0:1025]=rbd[rows][0:n4,:]
i+=n4

# plm
rows=np.where(plm[:,1024]==0)
data[i:i+n3,0:1025]=plm[rows][0:n3,:]
i+=n3
rows=np.where(plm[:,1024]==1)
data[i:i+n4,0:1025]=plm[rows][0:n4,:]
i+=n4
rows=np.where(plm[:,1024]==2)
data[i:i+n4,0:1025]=plm[rows][0:n4,:]
i+=n4
rows=np.where(plm[:,1024]==3)
data[i:i+n4,0:1025]=plm[rows][0:n4,:]
i+=n4

# labelling unhealthy
data[n:total,1025]=1
np.savetxt('dataa/healthy_unhealthy1.csv', data, delimiter=',')

rows=np.where(data[:,1024]==0)
print(np.shape(rows))
rows=np.where(data[:,1024]==1)
print(np.shape(rows))
rows=np.where(data[:,1024]==2)
print(np.shape(rows))
rows=np.where(data[:,1024]==3)
print(np.shape(rows))

# B phase

data = np.loadtxt('dataa/healthy_unhealthy1.csv', delimiter=',')
rows = np.where(data[:,1024]==0)
dataB=np.empty([int(np.shape(data)[0]/2), np.shape(data)[1]])
dataB=data[rows]
np.savetxt('dataa/healthy_unhealthyB.csv', dataB, delimiter=',')

# A phase

data = np.loadtxt('dataa/healthy_unhealthy1.csv', delimiter=',')
n1=int(np.shape(data)[0]/2)
n2=int(n1/3)
dataA=np.empty([n1, np.shape(data)[1]])
rows = np.where(data[:,1024]==1)
dataA[0:n2]=data[rows]
rows = np.where(data[:,1024]==2)
dataA[n2:2*n2]=data[rows]
rows = np.where(data[:,1024]==3)
dataA[2*n2:3*n2]=data[rows]
np.savetxt('dataa/healthy_unhealthyA.csv', dataA, delimiter=',')

# Disease classification dataset

n1=np.shape(ins)[0]
n2=np.shape(nfle)[0]
n3=np.shape(narco)[0]
n4=np.shape(rbd)[0]
n5=np.shape(plm)[0]
n_min=min(n1, n2, n3, n4, n5)
print('Available Samples of ins:', n1)
print('Available Samples of nfle:', n2)
print('Available Samples of narco:', n3)
print('Available Samples of rbd:', n4)
print('Available Samples of plm:', n5)
print('Samples considered of each disease:', n_min)
n3=int(n_min/2)
n4=int(n3/3)
total=int(n_min*5)
print('Total samples:', total)

data=np.zeros([total, 1026])

i=0
i1=0
# insomnia
rows=np.where(ins[:,1024]==0)
data[i:i+n3,0:1025]=ins[rows][0:n3,:]
i+=n3
rows=np.where(ins[:,1024]==1)
data[i:i+n4,0:1025]=ins[rows][0:n4,:]
i+=n4
rows=np.where(ins[:,1024]==2)
data[i:i+n4,0:1025]=ins[rows][0:n4,:]
i+=n4
rows=np.where(ins[:,1024]==3)
data[i:i+n4,0:1025]=ins[rows][0:n4,:]
i+=n4
data[i1:i,-1]=0
i1=i

# nfle
rows=np.where(nfle[:,1024]==0)
data[i:i+n3,0:1025]=nfle[rows][0:n3,:]
i+=n3
rows=np.where(nfle[:,1024]==1)
data[i:i+n4,0:1025]=nfle[rows][0:n4,:]
i+=n4
rows=np.where(nfle[:,1024]==2)
data[i:i+n4,0:1025]=nfle[rows][0:n4,:]
i+=n4
rows=np.where(nfle[:,1024]==3)
data[i:i+n4,0:1025]=nfle[rows][0:n4,:]
i+=n4
data[i1:i,-1]=1
i1=i

# narcolepsy
rows=np.where(narco[:,1024]==0)
data[i:i+n3,0:1025]=narco[rows][0:n3,:]
i+=n3
rows=np.where(narco[:,1024]==1)
data[i:i+n4,0:1025]=narco[rows][0:n4,:]
i+=n4
rows=np.where(narco[:,1024]==2)
data[i:i+n4,0:1025]=narco[rows][0:n4,:]
i+=n4
rows=np.where(narco[:,1024]==3)
data[i:i+n4,0:1025]=narco[rows][0:n4,:]
i+=n4
data[i1:i,-1]=2
i1=i

# rbd
rows=np.where(rbd[:,1024]==0)
data[i:i+n3,0:1025]=rbd[rows][0:n3,:]
i+=n3
rows=np.where(rbd[:,1024]==1)
data[i:i+n4,0:1025]=rbd[rows][0:n4,:]
i+=n4
rows=np.where(rbd[:,1024]==2)
data[i:i+n4,0:1025]=rbd[rows][0:n4,:]
i+=n4
rows=np.where(rbd[:,1024]==3)
data[i:i+n4,0:1025]=rbd[rows][0:n4,:]
i+=n4
data[i1:i,-1]=3
i1=i

# plm
rows=np.where(plm[:,1024]==0)
data[i:i+n3,0:1025]=plm[rows][0:n3,:]
i+=n3
rows=np.where(plm[:,1024]==1)
data[i:i+n4,0:1025]=plm[rows][0:n4,:]
i+=n4
rows=np.where(plm[:,1024]==2)
data[i:i+n4,0:1025]=plm[rows][0:n4,:]
i+=n4
rows=np.where(plm[:,1024]==3)
data[i:i+n4,0:1025]=plm[rows][0:n4,:]
i+=n4
data[i1:i,-1]=4
i1=i

np.savetxt('dataa/both_phases.csv', data, delimiter=',')

# B phase

data = np.loadtxt('dataa/both_phases.csv', delimiter=',')
rows = np.where(data[:,1024]==0)
dataB=np.empty([int(np.shape(data)[0]/2), np.shape(data)[1]])
dataB=data[rows]
np.savetxt('dataa/B_phases.csv', dataB, delimiter=',')

# A phase

data = np.loadtxt('dataa/both_phases.csv', delimiter=',')
n1=int(np.shape(data)[0]/2)
n2=int(n1/3)
dataA=np.empty([n1, np.shape(data)[1]])
rows = np.where(data[:,1024]==1)
dataA[0:n2]=data[rows]
rows = np.where(data[:,1024]==2)
dataA[n2:2*n2]=data[rows]
rows = np.where(data[:,1024]==3)
dataA[2*n2:3*n2]=data[rows]
np.savetxt('dataa/A_phases.csv', dataA, delimiter=',')
