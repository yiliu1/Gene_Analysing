
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
get_ipython().magic('matplotlib inline')
import math
from numpy import linalg as LA
import numpy as np
from scipy import stats


# # Wasserstein  distance bound   for  linear  model

# In[28]:


import random
random.seed(2019)
from scipy.sparse import random


# In[29]:


n=10# training data size
p=1000 # dimension
n_new=5#test data
sigma_matrix=1# variation for design matrix
sigma_noise=0.01 # variation for additive noise 
delta=0.01 # probability parameter


# # Synthesize design matrix 

# In[30]:


design_matrix=np.random.normal(0,sigma_matrix,size=(n,p))
design_matrix=(design_matrix-np.mean(design_matrix))/(np.max(design_matrix-np.min(design_matrix)))
design_matrix.shape


# # Synthesize true parameter and target

# In[31]:


beta_0=random(p,1,density=0.01)
beta_true=np.reshape(beta_0.A,(1000))#create parameter
Y=np.matmul(design_matrix,beta_0.A)+np.random.normal(0,sigma_noise,size=(n,1))
Y.shape #creat  target


# In[32]:


design_matrix.shape


# In[33]:


Y.shape


# In[34]:


beta_true[np.nonzero(beta_true)]# true sparse parameters


# # Part I :Lasso  to find parameter

# In[237]:


from sklearn import linear_model
## according to the paper ,alpha=lambda/n=sigma sqrt(2*log(ep/delta)/n)
alpha=1*sigma_noise*np.sqrt(2*np.log(np.e*p/delta)/n)
print("tuning parameters",alpha)
Lasso_model=linear_model.Lasso(alpha=alpha)
Lasso_model.fit(design_matrix,Y)


# # Parameter consistency

# In[286]:


print("L1 norm of beta_estimation-beta_true:",LA.norm((Lasso_model.coef_-beta_true),ord=1))
print(LA.norm(Lasso_model.coef_,ord=1)+LA.norm(beta_true,ord=1))
print("L1 norm of true parameters:",LA.norm(beta_true,ord=1))
print("L1 norm of lasso parameters:",LA.norm(Lasso_model.coef_,ord=1))


# # In sample error (theoretical risk fix design)

# In[243]:


Y_pred_in_sample=Lasso_model.predict(design_matrix)
Y_0=np.reshape(np.matmul(design_matrix,beta_0.A),(n,))# synthesized by beta_0
In_sample_error=LA.norm(Y_pred_in_sample-Y_0,ord=2)**2/n
print("In_sample_error:",In_sample_error)


# In[244]:


Y_pred_in_sample.shape


# In[245]:


Y_0.shape


# # Test on new design matrix 

# In[246]:


W_distance=[]
Error_matrix=[]
N=100000
a= np.ones((n_new,))/n_new
b= np.ones((n,))/ n     # uniform distribution on samples

#Normal distribution
for iteration in range(0,N):
    #Normalized design matrix 
    new_design=np.random.normal(0,sigma_matrix,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    #New predictions
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(n_new))
    Y_pred_new=Lasso_model.predict(new_design)
    # L2 error for out of sample 
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    # Wasserstein distance  between sampled design and new design
    # a, b are uniform distribution
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    # add to list 
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Poisson distribution
for iteration in range(0,N):
    new_design=np.random.poisson(5,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#hypergeometric distribution
for iteration in range(0,N):
    new_design=np.random.hypergeometric(15, 15, 15, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Von Mises distribution
for iteration in range(0,N):
    new_design=np.random.vonmises(0,4,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#rayleigh distribution
for iteration in range(0,N):
    new_design=np.random.rayleigh(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#power distribution
for iteration in range(0,N):
    new_design=np.random.power(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Chisquare 
for iteration in range(0,N):
    new_design=np.random.chisquare(2, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
#beta distribution
for iteration in range(0,N):
    new_design=np.random.beta(0.5,0.5, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Zipf distribution
for iteration in range(0,N):
    new_design=np.random.zipf(2,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#Uniform distributin
for iteration in range(0,N):
    new_design=np.random.uniform(-1,1,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    Y_pred_new=Lasso_model.predict(new_design)
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)


# In[247]:


W_distance=0.02*np.array(W_distance)
len(Error_matrix)


# In[248]:


import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0,1000000)

# red dashes, blue squares and green triangles
fig=plt.figure(figsize=(20,10))
plt.plot(t, W_distance, 'g^', t, Error_matrix, 'r--')#, t, t**3, 'g^')
#plt.plot(t, W_distance, 'r--', t, Error_matrix, 'bs')
plt.legend(('Scaled Wasserstein distance', 'Out of Sample Error'),
           loc='upper left', prop={'size': 25})

plt.show()
fig.savefig('./Downloads/Error_lasso.png')


# # Part II: Ridge regression to find parameters

# In[249]:


alpha_ridge=2*sigma_noise*np.sqrt(2*n*np.log(np.e*p/delta))
ridge_model=linear_model.Ridge(alpha=alpha_ridge)
ridge_model.fit(design_matrix,Y)


# # Parameter consistency 

# In[267]:


ridge_model.coef_=np.reshape(ridge_model.coef_,(p))


# In[268]:


beta_true.shape


# In[269]:


ridge_model.coef_.shape


# In[285]:


print("L1 norm of beta_estimation-beta_true:", LA.norm(ridge_model.coef_-beta_true,ord=1))
print("L1 norm of ridge regression parameters:",LA.norm(ridge_model.coef_,ord=1))
print("L1 norm of true parameters:",LA.norm(beta_true,ord=1))


# # In Sample Error 

# In[287]:


Y_ridge_pred=np.reshape(ridge_model.predict(design_matrix),(n,))
Y_0=np.reshape(np.matmul(design_matrix,beta_0.A),(n,))
Y_ridge_true=Y_0
# In sample error
In_sample_error_ridge=LA.norm(Y_ridge_pred-Y_ridge_true,ord=2)**2/n
print("In_sample_error for ridge regression",In_sample_error_ridge)


# # Test on out of sample error 

# In[275]:


W_distance=[]
Error_matrix=[]
N=100000
a= np.ones((n_new,))/n_new
b= np.ones((n,))/n     # uniform distribution on samples

#Normal distribution
for iteration in range(0,N):
    #Normalized design matrix 
    new_design=np.random.normal(0,sigma_matrix,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    #New predictions
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(n_new))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    # L2 error for out of sample 
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    # Wasserstein distance  between sampled design and new design
    # a, b are uniform distribution
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    # add to list 
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

    
#Poisson distribution
for iteration in range(0,N):
    new_design=np.random.poisson(5,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

    
#hypergeometric distribution
for iteration in range(0,N):
    new_design=np.random.hypergeometric(15, 15, 15, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Von Mises distribution

for iteration in range(0,N):
    new_design=np.random.vonmises(0,4,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#rayleigh distribution
for iteration in range(0,N):
    new_design=np.random.rayleigh(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#power distribution
for iteration in range(0,N):
    new_design=np.random.power(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Chisquare 
for iteration in range(0,N):
    new_design=np.random.chisquare(2, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
#beta distribution
for iteration in range(0,N):
    new_design=np.random.beta(0.5,0.5, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Zipf distribution
for iteration in range(0,N):
    new_design=np.random.zipf(2,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#Uniform distributin
for iteration in range(0,N):
    new_design=np.random.uniform(-1,1,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=ridge_model.predict(new_design)
    Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)


# In[276]:


W_distance=0.02*np.array(W_distance)


# In[278]:


import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0,1000000)

# red dashes, blue squares and green triangles
fig=plt.figure(figsize=(20,10))
plt.plot(t, W_distance, 'g^', t, Error_matrix, 'r--')#, t, t**3, 'g^')
#plt.plot(t, W_distance, 'r--', t, Error_matrix, 'bs')
plt.legend(('Scaled Wasserstein distance', 'Out of Sample Error'),
           loc='upper left', prop={'size': 25})

plt.show()
fig.savefig('./Downloads/Error_ridge.png')


# # Part III： ElasticNet  to find parameter

# In[288]:


ratio=0.7
alpha_enet=sigma_noise*np.sqrt(2*np.log(np.e*p/delta)/n)/ratio
enet=linear_model.ElasticNet(alpha=alpha_enet,l1_ratio=ratio)


# In[289]:


enet.fit(design_matrix,Y)


# # Parameter consistency 

# In[292]:


print("L1 norm of beta_estimation-beta_true:",LA.norm(enet.coef_-beta_true,ord=1))
print("L1 norm of enet parameters:",LA.norm(enet.coef_,ord=1))
print("L1 norm of true parameters:",LA.norm(beta_true,ord=1))


# # In sample Error 

# In[293]:


enet.predict(design_matrix).shape


# In[294]:


Y_enet_pred=enet.predict(design_matrix) #shape(n,1)
Y_0=np.reshape(np.matmul(design_matrix,beta_0.A),(n,)) #shape(n,1)
Y_enet_true=Y_0

# In sample error
In_sample_error_enet=LA.norm(Y_enet_pred-Y_enet_true,ord=2)**2/n
print("In_sample_error for Elasticnet regression",In_sample_error_enet)


# # Out of sample error 

# In[295]:


W_distance=[]
Error_matrix=[]
N=100000
a= np.ones((n_new,))/n_new
b= np.ones((n,))/n     # uniform distribution on samples

#Normal distribution
for iteration in range(0,N):
    #Normalized design matrix 
    new_design=np.random.normal(0,sigma_matrix,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    #New predictions
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(n_new))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    # L2 error for out of sample 
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    # Wasserstein distance  between sampled design and new design
    # a, b are uniform distribution
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    # add to list 
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Poisson distribution
for iteration in range(0,N):
    new_design=np.random.poisson(5,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#hypergeometric distribution

for iteration in range(0,N):
    new_design=np.random.hypergeometric(15, 15, 15, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Von Mises distribution

for iteration in range(0,N):
    new_design=np.random.vonmises(0,4,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#rayleigh distribution
for iteration in range(0,N):
    new_design=np.random.rayleigh(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#power distribution
for iteration in range(0,N):
    new_design=np.random.power(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
   # Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Chisquare 
for iteration in range(0,N):
    new_design=np.random.chisquare(2, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
#beta distribution
for iteration in range(0,N):
    new_design=np.random.beta(0.5,0.5, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Zipf distribution
for iteration in range(0,N):
    new_design=np.random.zipf(2,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
   # Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#Uniform distributin
for iteration in range(0,N):
    new_design=np.random.uniform(-1,1,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=enet.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)


# In[296]:


W_distance=0.02*np.array(W_distance)


# In[298]:


import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0,1000000)

# red dashes, blue squares and green triangles
fig=plt.figure(figsize=(20,10))
plt.plot(t, W_distance, 'g^', t, Error_matrix, 'r--')#, t, t**3, 'g^')
#plt.plot(t, W_distance, 'r--', t, Error_matrix, 'bs')
plt.legend(('Scaled Wasserstein distance', 'Out of Sample Error'),
           loc='upper left', prop={'size': 25})

plt.show()
fig.savefig('./Downloads/Error_Elasticnet.png')


# # Part IV ：Least square estimator

# In[301]:


reg=linear_model.LinearRegression()
reg.fit(design_matrix,Y)


# # Parameter Consistency

# In[303]:


reg.coef_=np.reshape(reg.coef_,(p))


# In[307]:


beta_true.shape


# In[309]:


print("L1 norm of difference of parameters:",LA.norm(reg.coef_-beta_true,ord=1))
print("L1 norm of estimation parameters:",LA.norm(reg.coef_,ord=1))
print("L1 norm of true parameters:",LA.norm(beta_true,ord=1))


# # In Sample Error 

# In[314]:


reg.predict(design_matrix).shape


# In[315]:


Y_reg_pred=reg.predict(design_matrix) #shape(n)
Y_0=np.reshape(np.matmul(design_matrix,beta_0.A),(n,)) #shape(n,)
Y_reg_true=Y_0
# In sample error
# Remarkable
In_sample_error_reg=LA.norm(Y_reg_pred-Y_reg_true,ord=2)**2/n
print("In_sample_error for Least square regression",In_sample_error_reg)


# # Out of Sample Error 

# In[316]:


W_distance=[]
Error_matrix=[]
N=100000
a= np.ones((n_new,))/n_new## empirical distribution on samples 
b= np.ones((n,))/n  ##uniform distribution on samples


#Normal distribution
for iteration in range(0,N):
    #Normalized design matrix 
    new_design=np.random.normal(0,sigma_matrix,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    #New predictions
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(n_new))
    
    Y_pred_new=reg.predict(new_design)# shape(n_new,)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    # L2 error for out of sample 
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    # Wasserstein distance  between sampled design and new design
    # a, b are uniform distribution
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    # add to list 
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Poisson distribution
for iteration in range(0,N):
    new_design=np.random.poisson(5,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#hypergeometric distribution

for iteration in range(0,N):
    new_design=np.random.hypergeometric(15, 15, 15, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Von Mises distribution

for iteration in range(0,N):
    new_design=np.random.vonmises(0,4,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#rayleigh distribution
for iteration in range(0,N):
    new_design=np.random.rayleigh(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#power distribution
for iteration in range(0,N):
    new_design=np.random.power(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
   # Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Chisquare 
for iteration in range(0,N):
    new_design=np.random.chisquare(2, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
#beta distribution
for iteration in range(0,N):
    new_design=np.random.beta(0.5,0.5, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Zipf distribution
for iteration in range(0,N):
    new_design=np.random.zipf(2,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
   # Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#Uniform distributin
for iteration in range(0,N):
    new_design=np.random.uniform(-1,1,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=reg.predict(new_design)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)


# In[317]:


W_distance=0.02*np.array(W_distance)


# In[319]:


import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0,1000000)

# red dashes, blue squares and green triangles
fig=plt.figure(figsize=(20,10))
plt.plot(t, W_distance, 'g^', t, Error_matrix, 'r--')#, t, t**3, 'g^')
#plt.plot(t, W_distance, 'r--', t, Error_matrix, 'bs')
plt.legend(('Scaled Wasserstein distance', 'Out of Sample Error'),
           loc='upper left', prop={'size': 25})

plt.show()
fig.savefig('./Downloads/Error_Least_square.png')


# # Part V :Random Parameter 

# In[10]:


theta=np.random.rand(1000)


# In[35]:


theta.shape


# # Parameter consistency 

# In[39]:


print("L1 norm of difference of parameters:",LA.norm(theta-beta_true,ord=1))
print("L1 norm of estimation parameters:",LA.norm(theta,ord=1))
print("L1 norm of true parameters:",LA.norm(beta_true,ord=1))


# # In sample Error

# In[46]:


np.matmul(design_matrix,beta_0.A).shape


# In[47]:


design_matrix.shape


# In[48]:


Y_pred=np.matmul(design_matrix,theta)
Y_0=np.reshape(np.matmul(design_matrix,beta_0.A),(n,)) #shape(n,)
# In sample error
# Remarkable
In_sample_error_reg=LA.norm(Y_pred-Y_0,ord=2)**2/n
print("In_sample_error for random parameter",In_sample_error_reg)


# # Out of sample error 

# In[ ]:


W_distance=[]
Error_matrix=[]
N=100000
a= np.ones((n_new,))/n_new ## empirical distribution on samples 
b= np.ones((n,))/n         ##uniform distribution on samples


#Normal distribution
for iteration in range(0,N):
    #Normalized design matrix 
    new_design=np.random.normal(0,sigma_matrix,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    #New predictions
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(n_new))
    
    Y_pred_new=np.matmul(new_design,theta)# shape(n_new,)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    # L2 error for out of sample 
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    # Wasserstein distance  between sampled design and new design
    # a, b are uniform distribution
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    # add to list 
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Poisson distribution
for iteration in range(0,N):
    new_design=np.random.poisson(5,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#hypergeometric distribution

for iteration in range(0,N):
    new_design=np.random.hypergeometric(15, 15, 15, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
    
#Von Mises distribution

for iteration in range(0,N):
    new_design=np.random.vonmises(0,4,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#rayleigh distribution
for iteration in range(0,N):
    new_design=np.random.rayleigh(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#power distribution
for iteration in range(0,N):
    new_design=np.random.power(3, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
   # Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Chisquare 
for iteration in range(0,N):
    new_design=np.random.chisquare(2, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)
#beta distribution
for iteration in range(0,N):
    new_design=np.random.beta(0.5,0.5, size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

# Zipf distribution
for iteration in range(0,N):
    new_design=np.random.zipf(2,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
   # Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)

#Uniform distributin
for iteration in range(0,N):
    new_design=np.random.uniform(-1,1,size=(n_new,1000))
    new_design=(new_design-np.mean(new_design))/(np.max(new_design)-np.min(new_design))
    Y_new=np.matmul(new_design,beta_0.A)
    Y_new=np.reshape(Y_new,(5))
    
    Y_pred_new=np.matmul(new_design,theta)
    #Y_pred_new=np.reshape(Y_pred_new,(n_new))
    
    Error=LA.norm(Y_new-Y_pred_new,ord=2)**2/n_new
    M=ot.dist(new_design,design_matrix)
    Wasserstein_distance=ot.emd2(a,b,M)
    
    W_distance.append(Wasserstein_distance)
    Error_matrix.append(Error)


# In[ ]:


W_distance=0.02*np.array(W_distance)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0,1000000)

# red dashes, blue squares and green triangles
fig=plt.figure(figsize=(20,10))
plt.plot(t, W_distance, 'g^', t, Error_matrix, 'r--')#, t, t**3, 'g^')
#plt.plot(t, W_distance, 'r--', t, Error_matrix, 'bs')
plt.legend(('Scaled Wasserstein distance', 'Out of Sample Error'),
           loc='upper left', prop={'size': 25})

plt.show()
#fig.savefig('./Downloads/Error_Least_square.png')

