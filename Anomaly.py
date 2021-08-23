from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture
from numpy import quantile, where
from sklearn.metrics import f1_score


cols="""duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

columns=[]
for c in cols.split(','):
    if(c.strip()):
        columns.append(c.strip())

columns.append('target')

attacks_types = {
    'normal': 1,
'back': 0,
'buffer_overflow': 0,
'ftp_write': 0,
'guess_passwd': 0,
'imap': 0,
'ipsweep': 0,
'land': 0,
'loadmodule': 0,
'multihop': 0,
'neptune': 0,
'nmap': 0,
'perl': 0,
'phf': 0,
'pod': 0,
'portsweep': 0,
'rootkit': 0,
'satan': 0,
'smurf': 0,
'spy': 0,
'teardrop': 0,
'warezclient': 0,
'warezmaster': 0,
}


df=pd.read_csv("data/kddcup.data_10_percent.gz",names=columns)

df['Attack Type']= df.target.apply(lambda r:attacks_types[r[:-1]])
df.drop('target', axis = 1, inplace = True)
df.isnull().sum()


#protocol_type feature mapping
pmap = {'icmp':0,'tcp':1,'udp':2}
df['protocol_type'] = df['protocol_type'].map(pmap)

#flag feature mapping
fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
df['flag'] = df['flag'].map(fmap)
df.drop('service', axis=1 , inplace= True)


X = df.drop(['Attack Type',], axis=1)
Y= df[['Attack Type']]
Y = Y.values.ravel()

#PCA
pca = PCA(n_components= 'mle', whiten = False )
X = pca.fit_transform(X)
a,b=X.shape


X = X + np.absolute(X).max()
X = np.sqrt(X)


#GaussianMixture
gm = GaussianMixture(n_components = b).fit(X)

Z = gm.predict_proba(X)
cluster = gm.predict(X)

#p(x) 
Z = 1 - Z
array = np.cumprod(Z, axis = 1)[:,-1]
array = 1-array

#bestEpsilon
bestEpsilon = 0
F1=0
bestF1=0

for epsilon in np.arange(0, 1, 0.01):
    not_normal = where(array <= epsilon)
    cluster[not_normal] = 1

    normal = where(array > epsilon)
    cluster[normal] = 0
    F1 = f1_score(y_true=Y, y_pred= cluster, average='binary')
    if F1 > bestF1:
        bestEpsilon = epsilon
        bestF1 = F1
        
print("Best F1", bestF1)
print("Best epsilon", bestEpsilon)

#elective epsilon
# epsilon = ???
# not_normal = where(array <= epsilon)
# cluster[not_normal] = 1

# normal = where(array > epsilon)
# cluster[normal] = 0

# print("F1 score: ", f1_score(y_true=Y, y_pred= cluster, average='binary'))