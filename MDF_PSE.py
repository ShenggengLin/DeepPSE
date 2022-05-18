
# coding: utf-8

# In[57]:


from __future__ import division
from __future__ import print_function
from itertools import combinations
from IPython.display import display
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
import random
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
import gc


# In[58]:
start=time.time()

seed=0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# In[59]:
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

len_after_AE=400
n_layers=4
drop_out_rating=0.5
learn_rate=0.00002
batch_size=2048
epo_num=50



n_heads=4
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
weight_decay_rate=0.0001

# Returns dictionary from combination ID to pair of stitch IDs, 
# dictionary from combination ID to list of polypharmacy side effects, 
# and dictionary from side effects to their names.
def load_combo_se(fname='./Datasets/combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id1, stitch_id2, se, se_name = line.strip().split(',')
        combo = stitch_id1 + '_' + stitch_id2
        combo2stitch[combo] = [stitch_id1, stitch_id2]
        combo2se[combo].add(se)
        se2name[se] = se_name
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    print ('Drug combinations: %d Side effects: %d' % (len(combo2stitch), len(se2name)))
    print ('Drug-drug interactions: %d' % (n_interactions))
    return combo2stitch, combo2se, se2name

# Returns dictionary from Stitch ID to list of individual side effects, 
# and dictionary from side effects to their names.
def load_mono_se(fname='./Datasets/mono.csv'):
    stitch2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        contents = line.strip().split(',')
        stitch_id, se, = contents[:2]
        se_name = ','.join(contents[2:])
        stitch2se[stitch_id].add(se)
        se2name[se] = se_name
    return stitch2se, se2name

# Returns dictionary from Stitch ID to list of drug targets
def load_targets(fname='./Datasets/targets-all.csv'):
    stitch2proteins_all = defaultdict(set)
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id, gene = line.strip().split(',')
        stitch2proteins_all[stitch_id].add(gene)
    return stitch2proteins_all


# In[60]:


def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')


# In[61]:


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)


# In[62]:


#Load data
combo2stitch, combo2se, se2name = load_combo_se()
stitch2se, se2name_mono = load_mono_se()
stitch2proteins_all = load_targets()


# In[63]:


#Most common side effects in drug combinations
def get_se_counter(se_map):
    side_effects = []
    for drug in se_map:
        side_effects += list(set(se_map[drug]))
    return Counter(side_effects)
combo_counter = get_se_counter(combo2se)
print("Most common side effects in drug combinations:")
common_se = []
common_se_counts = []
common_se_names = []
for se, count in combo_counter.most_common(964):
    common_se += [se]
    common_se_counts += [count]
    common_se_names += [se2name[se]]
df = pd.DataFrame(data={"Side Effect": common_se, "Frequency in Drug Combos": common_se_counts, "Name": common_se_names})  
#display(df)
print(len(df))


# In[64]:


#list of drugs
lst=[]
for key , value in combo2stitch.items():
    first_name, second_name = map(lambda x: x.strip(), key.split('_'))
    if first_name not in lst:
        lst.append(first_name)
    if second_name not in lst:
        lst.append(second_name)
print(len(lst))


# In[65]:


#list of proteins
p=[]
for k,v in stitch2proteins_all.items():
    for i in v:
        if i not in p:
            p.append(i)
print(len(p))


# In[66]:


val_test_size=0.05
n_drugdrug_rel_types=964
n_drugs=645
n_proteins=7795


# In[67]:


#construct drug-protein-adj matrix
drug_protein_adj=np.zeros((n_drugs,n_proteins))
for i in range(n_drugs):
    for j in stitch2proteins_all[lst[i]]:
        k=p.index(j)
        drug_protein_adj[i,k]=1
#print(drug_protein_adj)

del stitch2proteins_all
gc.collect()
del p
gc.collect()
# In[68]:


#construct drug-drug-adj matrices for all side effects
drug_drug_adj_list=[]
l=[]
for i in range(n_drugdrug_rel_types):
    
    mat = np.zeros((n_drugs, n_drugs))
    l.append(df.at[i,'Side Effect'])
    for se in l:
        
        for d1, d2 in combinations(list(range(n_drugs)), 2):
            if lst[d1]+"_"+lst[d2]  in combo2se:
                if se in combo2se[lst[d1]+"_"+lst[d2]]:
                    mat[d1,d2]=mat[d2,d1]=1
    l=[]
    drug_drug_adj_list.append(mat)
#drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]


# In[69]:
del combo2stitch
gc.collect()
del combo2se
gc.collect()
del se2name
gc.collect()
del df
gc.collect()


#select drug pairs for training & validation & testing
edges=[]
for k in range(n_drugdrug_rel_types):
    l=[]
    for i in range(n_drugs):
        for j in range(n_drugs):
            if drug_drug_adj_list[k][i,j]==1:
                l.append([i,j])
    edges.append(l)
edges_false=[]
for k in range(n_drugdrug_rel_types):
    l=[]
    for i in range(n_drugs):
        for j in range(n_drugs):
            if drug_drug_adj_list[k][i,j]==0:
                l.append([i,j])
    edges_false.append(l)
for k in range(n_drugdrug_rel_types):
    np.random.shuffle(edges[k])
    np.random.shuffle(edges_false[k])
for k in range(n_drugdrug_rel_types):
    a=len(edges[k])
    edges_false[k]=edges_false[k][:a]
edges_all=[]
for k in range(n_drugdrug_rel_types):
    edges_all.append(edges[k]+edges_false[k])
for k in range(n_drugdrug_rel_types):
    np.random.shuffle(edges_all[k])
for k in range(n_drugdrug_rel_types):
    a=len(edges[k])
    edges_all[k]=edges_all[k][:a]

del edges_false
gc.collect()
del edges
gc.collect()




# In[70]:



#construct drug features
se_mono=[]
for k in se2name_mono:
    se_mono.append(k)
drug_label=np.zeros((n_drugs,len(se_mono)))

del se2name_mono
gc.collect()

for key,value in stitch2se.items():
    j=lst.index(key)
    for v in value:
        i=se_mono.index(v)
        drug_label[j,i]=1
del se_mono
gc.collect()
del lst
gc.collect()
del stitch2se
gc.collect()

pca = PCA(.95)
pca.fit(drug_label)
pca_= PCA(.95)
pca_.fit(drug_protein_adj)
drug_feat = np.concatenate((pca.transform(drug_label),pca_.transform(drug_protein_adj)),axis=1)


# In[71]:
print("will success!")



del drug_protein_adj
gc.collect()
del drug_label
gc.collect()

print("will success")
'''val=[]

for k in range(n_drugdrug_rel_types):
    a=int(np.floor(len(edges_all[k])*val_test_size))
    val.append(edges_all[k][:a])'''
#construct train & validation & test sets
val_sets=[]
val_labels=[]
for k in range(n_drugdrug_rel_types):
    huafen_len=int(np.floor(len(edges_all[k])*val_test_size))
    v=[]
    a=[]
    for  i in edges_all[k][:huafen_len]:
        v.append(np.concatenate((drug_feat[i[0]],drug_feat[i[1]])))
        a.append(drug_drug_adj_list[k][i[0],i[1]])
    random.seed(k)
    random.shuffle(v)
    random.seed(k)
    random.shuffle(a)
    val_sets.append(np.array(v))
    val_labels.append(np.array(a))


print("will success")
'''test=[]
for k in range(n_drugdrug_rel_types):
    a=int(np.floor(len(edges_all[k])*val_test_size))
    test.append(edges_all[k][a:a+a])'''
test_sets=[]
test_labels=[]
for k in range(n_drugdrug_rel_types):
    huafen_len = int(np.floor(len(edges_all[k]) * val_test_size))
    te=[]
    a=[]
    for  i in edges_all[k][huafen_len:huafen_len+huafen_len]:
        te.append(np.concatenate((drug_feat[i[0]],drug_feat[i[1]])))
        a.append(drug_drug_adj_list[k][i[0],i[1]])
    random.seed(k)
    random.shuffle(te)
    random.seed(k)
    random.shuffle(a)
    test_sets.append(np.array(te))
    test_labels.append(np.array(a))


print("will success")
'''train=[]
for k in range(n_drugdrug_rel_types):
    a=int(np.floor(len(edges_all[k])*val_test_size))
    train.append(edges_all[k][a+a:])'''
train_sets=[]
train_labels=[]
for k in range(n_drugdrug_rel_types):
    huafen_len = int(np.floor(len(edges_all[k]) * val_test_size))
    tr=[]
    a=[]
    for i in edges_all[k][huafen_len+huafen_len:]:
        tr.append(np.concatenate((drug_feat[i[0]],drug_feat[i[1]])))
        a.append(drug_drug_adj_list[k][i[0],i[1]])
    random.seed(k)
    random.shuffle(tr)
    random.seed(k)
    random.shuffle(a)
    train_sets.append(np.array(tr))
    train_labels.append(np.array(a))
drug_fea_len=drug_feat.shape[1]
del drug_drug_adj_list
gc.collect()
del drug_feat
gc.collect()
del edges_all
gc.collect()

for k in range(n_drugdrug_rel_types):
    train_sets[k]=np.vstack((train_sets[k],np.hstack((train_sets[k][:,len(train_sets[0])//2:],train_sets[k][:,:len(train_sets[0])//2]))))
    train_labels[k] = np.hstack((train_labels[k], train_labels[k]))
    np.random.seed(k)
    np.random.shuffle(train_sets[k])
    np.random.seed(k)
    np.random.shuffle(train_labels[k])





'''print("will success!")
val_org=[]
val_label_org=[]
test_org=[]
test_label_org=[]
train_org=[]
train_label_org=[]

for k in range(n_drugdrug_rel_types):
    val_org.append(np.array(val_sets[k]))
    val_label_org.append(np.array(val_labels[k]))
del val_sets
gc.collect()
del val_labels
gc.collect()
for k in range(n_drugdrug_rel_types):
    test_org.append(np.array(test_sets[k]))
    test_label_org.append(np.array(test_labels[k]))
del test_sets
gc.collect()
del test_labels
gc.collect()
for k in range(n_drugdrug_rel_types):
    train_org.append(np.array(train_sets[k]))
    train_label_org.append(np.array(train_labels[k]))
del train_sets
gc.collect()
del train_labels
gc.collect()'''

#construct model
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,input_dim,n_heads,ouput_dim=None):
        
        super(MultiHeadAttention, self).__init__()
        self.d_k=self.d_v=input_dim//n_heads
        self.n_heads = n_heads
        if ouput_dim==None:
            self.ouput_dim=input_dim
        else:
            self.ouput_dim=ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
    def forward(self,X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q=self.W_Q(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        K=self.W_K(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        V=self.W_V(X).view( -1, self.n_heads, self.d_v).transpose(0,1)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# In[107]:


class EncoderLayer(torch.nn.Module):
    def __init__(self,input_dim,n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim,n_heads)
        self.AN1=torch.nn.LayerNorm(input_dim)
        
        self.l1=torch.nn.Linear(input_dim, input_dim)
        self.AN2=torch.nn.LayerNorm(input_dim)
    def forward (self,X):
        
        output=self.attn(X)
        X=self.AN1(output+X)
        
        output=self.l1(X)
        X=self.AN2(output+X)
        
        return X
class AE1(torch.nn.Module): #Joining together
    def __init__(self,vector_size):
        super(AE1,self).__init__()
        
        self.vector_size=vector_size
        
        self.l1 = torch.nn.Linear(self.vector_size,(self.vector_size+len_after_AE)//2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE)//2)
        
        self.att2=EncoderLayer((self.vector_size+len_after_AE)//2,n_heads)
        self.l2 = torch.nn.Linear((self.vector_size+len_after_AE)//2,len_after_AE)
        
        self.l3 = torch.nn.Linear(len_after_AE,(self.vector_size+len_after_AE)//2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE)//2)
        
        self.l4 = torch.nn.Linear((self.vector_size+len_after_AE)//2,self.vector_size)
        
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac=torch.nn.GELU()
        
    def forward(self,X):
        
        X=self.dr(self.bn1(self.ac(self.l1(X))))
        
        X=self.att2(X)
        X=self.l2(X)
        
        X_AE=self.dr(self.bn3(self.ac(self.l3(X))))
        
        X_AE=self.l4(X_AE)
        
        return X,X_AE


# In[110]:


class AE2(torch.nn.Module):# twin network
    def __init__(self,vector_size):
        super(AE2,self).__init__()
        
        self.vector_size=vector_size//2
        
        self.l1 = torch.nn.Linear(self.vector_size,(self.vector_size+len_after_AE//2)//2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE//2)//2)
        
        self.att2=EncoderLayer((self.vector_size+len_after_AE//2)//2,n_heads)
        self.l2 = torch.nn.Linear((self.vector_size+len_after_AE//2)//2,len_after_AE//2)
        
        self.l3 = torch.nn.Linear(len_after_AE//2,(self.vector_size+len_after_AE//2)//2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE//2)//2)
        
        self.l4 = torch.nn.Linear((self.vector_size+len_after_AE//2)//2,self.vector_size)
        
        self.dr = torch.nn.Dropout(drop_out_rating)
        
        self.ac=torch.nn.GELU()
        
    def forward(self,X):
        
        X1=X[:,0:self.vector_size]
        X2=X[:,self.vector_size:]
        
        X1=self.dr(self.bn1(self.ac(self.l1(X1))))
        X1=self.att2(X1)
        X1=self.l2(X1)
        X_AE1=self.dr(self.bn3(self.ac(self.l3(X1))))
        X_AE1=self.l4(X_AE1)
        
        X2=self.dr(self.bn1(self.ac(self.l1(X2))))
        X2=self.att2(X2)
        X2=self.l2(X2)
        X_AE2=self.dr(self.bn3(self.ac(self.l3(X2))))
        X_AE2=self.l4(X_AE2)
        
        X=torch.cat((X1,X2), 1)
        X_AE=torch.cat((X_AE1,X_AE2), 1)
        
        return X,X_AE


# In[111]:


class cov(torch.nn.Module):
    def __init__(self,vector_size):
        super(cov,self).__init__()
        
        self.vector_size=vector_size
        
        self.co2_1=torch.nn.Conv2d(1, 1, kernel_size=(2,cov2KerSize))
        self.co1_1=torch.nn.Conv1d(1, 1, kernel_size=cov1KerSize)
        self.pool1=torch.nn.AdaptiveAvgPool1d(len_after_AE)
        
        self.ac=torch.nn.GELU()
        
        
    def forward(self,X):
        
        X1=X[:,0:self.vector_size//2]
        X2=X[:,self.vector_size//2:]
        
        X=torch.cat((X1,X2), 0)
        
        X=X.view(-1,1,2,self.vector_size//2)
        
        X=self.ac(self.co2_1(X))
        
        X=X.view(-1,self.vector_size//2-cov2KerSize+1, 1)  
        X=X.permute(0,2,1)
        X=self.ac(self.co1_1(X))
        
        X=self.pool1(X)
        
        X=X.contiguous().view(-1,len_after_AE)
        
        return X


# In[112]:


class ADDAE(torch.nn.Module):
    def __init__(self,vector_size):
        super(ADDAE,self).__init__()
        
        self.vector_size=vector_size//2
        
        self.l1 = torch.nn.Linear(self.vector_size,(self.vector_size+len_after_AE)//2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE)//2)
        
        self.att1=EncoderLayer((self.vector_size+len_after_AE)//2,n_heads)
        self.l2 = torch.nn.Linear((self.vector_size+len_after_AE)//2,len_after_AE)
        #self.att2=EncoderLayer(len_after_AE//2,bert_n_heads)
        
        self.l3 = torch.nn.Linear(len_after_AE,(self.vector_size+len_after_AE)//2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size+len_after_AE)//2)
        
        self.l4 = torch.nn.Linear((self.vector_size+len_after_AE)//2,self.vector_size)
        
        self.dr = torch.nn.Dropout(drop_out_rating)
        
        self.ac=torch.nn.GELU()
        
    def forward(self,X):
        
        X1=X[:,0:self.vector_size]
        X2=X[:,self.vector_size:]
        X=X1+X2
        
        X=self.dr(self.bn1(self.ac(self.l1(X))))
        
        X=self.att1(X)
        X=self.l2(X)
        
        X_AE=self.dr(self.bn3(self.ac(self.l3(X))))
        
        X_AE=self.l4(X_AE)
        X_AE=torch.cat((X_AE,X_AE), 1)
        
        return X,X_AE
class Model(torch.nn.Module):
    def __init__(self,input_dim,n_heads,n_layers):
        super(Model, self).__init__()
        
        self.ae1=AE1(input_dim)  #Joining together
        self.ae2=AE2(input_dim)#twin loss
        self.cov=cov(input_dim)#cov 
        self.ADDAE=ADDAE(input_dim)
        
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.input_dim=input_dim
        
        self.layers = torch.nn.ModuleList([EncoderLayer(len_after_AE*5,n_heads) for _ in range(n_layers)])
        
        
        self.l1=torch.nn.Linear(len_after_AE*5,len_after_AE*5//4)
        self.bn1=torch.nn.BatchNorm1d(len_after_AE*5//4)
        
        self.l2=torch.nn.Linear(len_after_AE*5//4,1)
        
        self.ac=torch.nn.GELU()
        
        self.ac_sig=torch.nn.Sigmoid()
        
    def forward(self, X):
        X1,X_AE1=self.ae1(X)
        X2,X_AE2=self.ae2(X)
        
        X3=self.cov(X)
        
        X4,X_AE4=self.ADDAE(X)
        
        X5=X1+X2+X3+X4
        
        X=torch.cat((X1,X2,X3,X4,X5), 1)
        
        for layer in self.layers:
            X = layer(X)
        
        
        X=self.dr(self.bn1(self.ac(self.l1(X))))
        
        X=self.ac_sig(self.l2(X))
        
        return X,X_AE1,X_AE2,X_AE4
        


# In[73]:


class My_loss(torch.nn.Module):
    def __init__(self):
        
        super(My_loss,self).__init__()
        
        self.criteria1 = torch.nn.BCELoss()
        self.criteria2=torch.nn.MSELoss()

    def forward(self, X, target,inputs,X_AE1,X_AE2,X_AE4):


        
        loss=calssific_loss_weight*self.criteria1(X,target)+             self.criteria2(inputs.float(),X_AE1)+             self.criteria2(inputs.float(),X_AE2)+             self.criteria2(inputs.float(),X_AE4)
        return loss


# In[74]:




class DDIDataset(Dataset):
    def __init__(self,x,y):
        self.len=len(x)
        self.x_data=torch.from_numpy(x)
        self.y_data=torch.from_numpy(y)
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

#get criteria
roc_score=[]
aupr_score=[]
f_score=[]
thr=[]
precision=[]
recall=[]
tpos=[]
fpos=[]
tneg=[]
fneg=[]
acc=[]
mcc=[]

for k in range(n_drugdrug_rel_types):
    print(k)
    model=Model(drug_fea_len*2,n_heads,n_layers)
    model_optimizer=torch.optim.Adam(model.parameters(),lr=learn_rate,weight_decay=weight_decay_rate)
    model=torch.nn.DataParallel(model)
    model=model.to(device)
    
    train_dataset = DDIDataset(train_sets[k],train_labels[k])
    val_dataset = DDIDataset(val_sets[k],val_labels[k])
    test_dataset = DDIDataset(test_sets[k],test_labels[k])
    
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    
    my_loss=My_loss()
    
    for epoch in range(epo_num):

        train_loss = []

        model.train()
        for batch_idx,data in enumerate(train_loader,0):
            inputs, targets = data

            inputs=inputs.to(device)
            targets=targets.reshape(-1, 1).to(device).float()

            model_optimizer.zero_grad()

            X,X_AE1,X_AE2,X_AE4=model(inputs.float())

            loss= my_loss(X, targets,inputs,X_AE1,X_AE2,X_AE4)

            loss.backward()
            model_optimizer.step()   
            train_loss.append(loss.item())

        model.eval()
        val_loss=[]
        with torch.no_grad():
            for batch_idx,data in enumerate(val_loader,0):
                inputs,targets=data

                inputs = inputs.to(device)

                targets=targets.reshape(-1, 1).to(device).float()

                X,X_AE1,X_AE2,X_AE4=model(inputs.float())

                loss= my_loss(X, targets,inputs,X_AE1,X_AE2,X_AE4)
                val_loss.append(loss.item())
        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)
        print('epoch [%d] train_loss: %.6f val_loss: %.6f ' % (epoch+1,mean_train_loss,mean_val_loss))

    
    pre_score=np.zeros((0, 1), dtype=float)
    model.eval()        
    with torch.no_grad():
        for batch_idx,data in enumerate(test_loader,0):
            inputs,_=data
            inputs=inputs.to(device)
            X,_,_,_= model(inputs.float())
            pre_score =np.vstack((pre_score,X.cpu().numpy()))
            
    roc=metrics.roc_auc_score(test_labels[k],pre_score)
    roc_score.append(roc)
    aupr=metrics.average_precision_score(test_labels[k],pre_score)
    aupr_score.append(aupr)
    fpr, tpr, thresholds=metrics.roc_curve(test_labels[k],pre_score)
    scores=[metrics.f1_score(test_labels[k], to_labels(pre_score, t)) for t in thresholds]
    ma= max(scores)
    f_score.append(ma)
    idx=np.argmax(scores)
    bt=thresholds[idx]
    thr.append(bt)
    p=metrics.precision_score(test_labels[k], to_labels(pre_score, bt))
    precision.append(p)
    r=metrics.recall_score(test_labels[k], to_labels(pre_score, bt))
    recall.append(r)
    TP, FP, TN, FN=perf_measure(test_labels[k],to_labels(pre_score, bt))
    tpos.append(TP)
    fpos.append(FP)
    tneg.append(TN)
    fneg.append(FN)
    ac = float(TP + TN)/(TP+FP+FN+TN)
    acc.append(ac)
    mc=metrics.matthews_corrcoef(test_labels[k],to_labels(pre_score, bt))
    mcc.append(mc)
mean_F_score = np.mean(f_score)
mean_AUPR = np.mean(aupr_score)
mean_roc_score=np.mean(roc_score)
mean_acc_score=np.mean(acc)
mean_mcc=np.mean(mcc)
mean_precision=np.mean(precision)
mean_recall=np.mean(precision)
print('F_score:[%.6f]' % mean_F_score)
print('AUPR:[%.6f]' % mean_AUPR)
print('roc_score:[%.6f]' % mean_roc_score)
print('acc_score:[%.6f]' % mean_acc_score)
print('mcc:[%.6f]' % mean_mcc)
print('precision:[%.6f]' % mean_precision)
print('recall:[%.6f]' % mean_recall)

print("time used:", (time.time() - start) / 3600)