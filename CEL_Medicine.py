
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,10
import random
random.seed('liuyi')


# In[83]:





# In[84]:


#ls


# In[58]:


gene_Data=pd.read_csv("./gene_data.txt")
gene_Data=gene_Data.set_index("Unnamed: 0")
gene_Data=gene_Data.T


# In[59]:


gene_Data


# In[60]:


Symbol=pd.read_csv("./Symbol.txt")
Symbol=Symbol.set_index('Unnamed: 0').T.to_dict('list')


# In[61]:


Symbol.get('1053_at')


# In[62]:


x=gene_Data.values[[0,1,4,5],:]


# In[63]:


x.shape


# # Design Matrix is X

# In[64]:


y=np.array([1,1,0,0])


# In[65]:


y.shape


# In[66]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


# In[67]:


clf = LogisticRegression(penalty='l1'
                       ,max_iter=2000,C=100).fit(x, y)


# In[68]:


coefs=clf.coef_.reshape(clf.coef_.shape[1])


# In[69]:


nonzero_index=np.nonzero(coefs)


# In[70]:


len(nonzero_index[0])


# In[71]:


np.nonzero(coefs)[0]


# In[38]:


coefs[np.nonzero(coefs)[0]]


# In[72]:


relevant_genes=gene_Data.columns[nonzero_index]
len(relevant_genes)


# In[73]:


len(relevant_genes)


# In[74]:


relevant_genes


# In[75]:


from bioservices.kegg import KEGG
k = KEGG()


# In[76]:


A=list(relevant_genes)


# In[77]:


import sys


# In[78]:


len(A)


# In[79]:


#pwd


# In[81]:


with open('./pathway_baicalin_386_genes.txt','w') as f:
    for i in A:
        try:
            listname=Symbol.get(i)
            n=str(listname[0])
            if (n!='nan'):
                j=k.get_pathway_by_gene(n,"hsa")
                if (j!=None):
                    for keys,values in j.items():
                        print(n,keys,values,sep="\t")
                    #f.write("%s \n" %n)
                    #for keys,values in j.items():
                    #    f.write("%s %s %s\n" %(n,keys,values))         
        except AttributeError as err:
            print('error')


# In[134]:


#Print algorithms
#for i in A:
#    try:
#        listname=Symbol.get(i)
#        n=str(listname[0])
#        if (n!='nan'):
#            #print("hello")
#            j=k.get_pathway_by_gene(n,"hsa")
#            if (j!=None):
#                for keys,values in j.items():
#                     print(n,keys,values,sep="\t")
#    except AttributeError as err:
#        print('error')


# # Plot

# In[82]:


plt.rcdefaults()
fig, ax = plt.subplots()


# set x,y axis
genes =A
y_pos = np.arange(len(genes))
performance = coefs[np.nonzero(coefs)[0]]
#error = np.random.rand(len(genes))


ax.barh(y_pos, performance, align='center',
        color='purple', ecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(genes)
ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Significant Coefficients')
ax.set_ylabel('Genes')
ax.set_title('Relavant genes')
#plot figues or save figures
#plt.show()
#plt.savefig("gene.png")

