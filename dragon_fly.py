# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:33:16 2020

@author: Soumyajit Saha
"""
import cv2
import numpy as np
import random
#from sklearn.metrics import mean_squared_error as mse
from PIL import ImageChops, Image
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import color
from statsmodels.stats.outliers_influence import variance_inflation_factor
from math import log10, sqrt 
import skimage 
from skimage import filters
import math 
from skimage.metrics import structural_similarity as ssim
from PIL import ImageEnhance, ImageOps
from sewar.full_ref import vifp
from skimage import io

def objective(sol,greyscale_img,uniq,region):
        
    out=np.zeros(shape=([len(greyscale_img),len(greyscale_img[0])]))
           
        
    for it in range(0,len(uniq)): # creation of new image
        for j in range(len(region[it][0])):
            out[region[it][0][j]][region[it][1][j]]=sol[it]
                    
    entropy = skimage.measure.shannon_entropy(out)  # calculation of entropy
    pv=len(out) # height of the image
    ph=len(out[0]) # weight of the image
    edge_sobel = filters.sobel(out) # egdes obtained from sobel filter
    # edge_sobel = filters.laplace(out)
    edge_sobel_fl=edge_sobel.flatten()
        
    E=0
    ne=np.count_nonzero(edge_sobel_fl)
    for i in range(len(edge_sobel_fl)):
        if edge_sobel_fl[i]>0:
            E+=edge_sobel_fl[i] # calculation of number of edge pixels
                    
        
    freq, x =  np.histogram(out, bins=256) # calculation of frequencies
    h=freq/sum(freq) # calculation of distribution of unique values
        
    Nt=0;
        
    for i in range(0,len(freq)):
        if h[i]>=h.mean(axis=0).mean(): # calulation the number of unique values having higher distribution than mean
            Nt+=1
        
    del_h=np.var(h)
            
    #math.log(entropy * Nt / del_h)    
    Fz=math.log(Nt/del_h)/(pv*ph)  # objective function        
        
        
    return Fz
        
######### Fitness function ##########
    
def fitness(sol,greyscale_img,uniq,region):
    value=objective(sol,greyscale_img,uniq,region)
    # value=evaluator(sol)
    # if (value >= 0):
    #     z = 1 / (1 + value)
    # else:
    #     z = 1 + abs(value)
    return value

def distance(a,b,dim):
    o = np.zeros(dim)
    for i in range(0,len(a)):
        o[i] = abs(a[i] - b[i])
    return o

def Levy(d):
    beta=3/2
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u=np.random.randn(d)*sigma
    v=np.random.randn(d)
    step=u/abs(v)**(1/beta)
    o=0.01*step
    return o        


def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                      # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr
psnr=np.zeros(24)
vif=np.zeros(24)
ssim_val=np.zeros(24)
img_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
for img_no in img_list:
    
    print("Image No. " + str(img_no))
    
    sr='D:/Project/Image Enhancement/Kodak/kodim' + str(img_no).zfill(2) + '.png'  # original image from kodak dataset
    
    source=Image.open(sr)
    source_gray=ImageOps.grayscale(source)

    
    source_gray.save('D:/Project/Image Enhancement/Kodak/kodim' + str(img_no).zfill(2) + '_grey.png')
    
    source=cv2.imread(sr)
    source=cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    
    dest=ImageEnhance.Contrast(source_gray)
    dest=dest.enhance(0.3)
    ds='D:/Project/Image Enhancement/kodak_30%/' + str(img_no).zfill(2) + '.png'   # lesser contrast image to be saved here
    dest.save(ds)
    
    
    
    img=cv2.imread(ds)
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gr=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    uniq=np.unique(greyscale_img)
    uniq=np.sort(uniq)
    
    dim=len(uniq)
    
    region=[]
    
    for i in uniq:
        reg=np.where(greyscale_img==i)
        #reg_list= list(zip(reg[0], reg[1]))
        region.append(reg)
    
    ub=255
    lb=0
    
    r=(ub-lb)/10
    Delta_max=(ub-lb)/8.5
    
    Food_fitness=0
    Food_pos=np.zeros(dim)
    
    Enemy_fitness=math.inf
    Enemy_pos=np.zeros(dim)
    
    population_size=50
    
    fitness_of_X = np.zeros(population_size)
    All_fitness = np.zeros(population_size)
    
    X = np.zeros(shape=(population_size,dim))
    DeltaX = np.zeros(shape=(population_size,dim))
    
    for i in range(0,population_size):
        for j in range(0,dim):
            X[i][j]=int(lb + random.uniform(0,1)*(ub-lb))
        
        X[i] = np.sort(X[i])
    
    i1=random.randint(0,population_size-1)
    i2=random.randint(0,population_size-1)
    
    # while i2==i1:
    #     i2=random.randint(0,49)
        
    
    # ub_del=max(distance(X[i1],X[i2],dim))
    
    ub_del=25
    
    
    
    for i in range(0,population_size):
        for j in range(0,dim):
            DeltaX[i][j]=int(lb + random.uniform(0,1)*(ub_del-lb))
        
        #DeltaX[i] = np.sort(DeltaX[i])
        
    Max_iteration=50
    
    
    for itr in range(1,Max_iteration+1):
        
        r=(ub_del-lb)/4+((ub_del-lb)*(itr/Max_iteration)*2)
        w=0.9-itr*((0.9-0.4)/Max_iteration)
        my_c=0.1-itr*((0.1-0)/(Max_iteration/2))
        
        if my_c<0:
            my_c=0
        
        s=2*random.random()*my_c
        a=2*random.random()*my_c
        c=2*random.random()*my_c
        f=2*random.random()*my_c
        e=my_c
        
        for i in range(0,population_size):
            fitness_of_X[i] = fitness(X[i],greyscale_img,uniq,region)
            All_fitness[i] = fitness_of_X[i]
            
            if fitness_of_X[i] > Food_fitness:
                Food_fitness = fitness_of_X[i]
                Food_pos=X[i]
            
            if fitness_of_X[i] < Enemy_fitness:
                if all((X[i] <= ub)) and all((X[i] >= lb)):
                    Enemy_fitness = fitness_of_X[i]
                    Enemy_pos = X[i]
                    
        for i in range(0,population_size):
            index=0
            neighbours_no=0
            
            Neighbours_X = np.zeros(shape=(population_size,dim))
            Neighbours_DeltaX = np.zeros(shape=(population_size,dim))
            
            for j in range(0,population_size):
                Dist2Enemy = distance(X[i],X[j],dim)
                if (all(Dist2Enemy<=r) and all(Dist2Enemy!=0)):
                    index=index+1
                    neighbours_no=neighbours_no+1
                    Neighbours_DeltaX[index]=DeltaX[j]
                    Neighbours_X[index]=X[j]
                    
            S=np.zeros(dim)           
            if neighbours_no>1:
                for k in range(0,neighbours_no):
                    S=S+(Neighbours_X[k]-X[i])
                S=-S
            else:
                S=np.zeros(dim)
                
            
            
            if neighbours_no>1:
                A=(sum(Neighbours_DeltaX))/neighbours_no
            else:
                A = DeltaX[i]
            
            
            
            if neighbours_no>1:
                C_temp=(sum(Neighbours_X))/neighbours_no
            else:
                C_temp=X[i]
        
            C=C_temp-X[i]
            
            
            
            Dist2Food=distance(X[i],Food_pos,dim)
                               
            if all(Dist2Food<=r):
                F=Food_pos-X[i]
            else:
                F=np.zeros(dim)
            
            
            
            Dist2Enemy=distance(X[i],Enemy_pos,dim)
                               
            if all(Dist2Enemy<=r):
                Enemy=Enemy_pos-X[i]
            else:
                Enemy=np.zeros(dim)
            
            
            
            for tt in range(0,dim):
                if X[i][tt]>ub:
                    X[i][tt]=ub
                    DeltaX[i][tt]=random.uniform(0,1)*(50-lb)
                    
                if X[i][tt]<lb:
                    X[i][tt]=lb
                    DeltaX[i][tt]=random.uniform(0,1)*(50-lb)
            
            temp=np.zeros(dim)
            Delta_temp=np.zeros(dim)
            
            if any(Dist2Food>r):
                if neighbours_no>1:
                    for j in range(0,dim):
                        Delta_temp[j] = int(w*DeltaX[i][j] + random.random()*A[j] + random.random()*C[j] + random.random()*S[j])
                        if Delta_temp[j]>Delta_max:
                            Delta_temp[j]=Delta_max
                        if Delta_temp[j]<-Delta_max:
                            Delta_temp[j]=-Delta_max
                        temp[j]=X[i][j]+(Delta_temp[j])
                else:
                    temp=(X[i] + (Levy(dim))*X[i]).astype(int)
                    Delta_temp=np.zeros(dim)
            
            else:
                for j in range(0,dim):
                    Delta_temp[j] = int((a*A[j] + c*C[j] + s*S[j] + f*F[j] + e*Enemy[j]) + w*DeltaX[i][j])
                    if Delta_temp[j]>Delta_max:
                        Delta_temp[j]=Delta_max
                    if Delta_temp[j]<-Delta_max:
                        Delta_temp[j]=-Delta_max
                    temp[j]=X[i][j]+Delta_temp[j]
                    
            for j in range(0,dim):
                if temp[j]<lb: # Bringinging back to search space
                        temp[j]=lb
                    
                if temp[j]>ub: # Bringinging back to search space
                    temp[j]=ub
            temp=np.sort(temp)
            Delta_temp=np.sort(Delta_temp)
            if(fitness(temp,greyscale_img,uniq,region)) > fitness_of_X[i]:
                X[i]=temp
                DeltaX[i]=Delta_temp
            
            
                    
        Best_score=Food_fitness
        Best_pos=Food_pos
        
        print("Iteration = " + str(itr))
    
    best_sol=Best_pos
     
    for it in range(0,len(uniq)): # creation of new image
        for j in range(len(region[it][0])):
            gr[region[it][0][j]][region[it][1][j]]=best_sol[it]
                   
    res='D:/Project/Image Enhancement/results/DA_Nt_del_h_obj_30/' + str(img_no).zfill(2)  # folder containing the output
    #res=''
    cv2.imwrite(res + '/before.jpg', greyscale_img)
    ax = plt.hist(greyscale_img.ravel(), 256,[0,256])
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(res + '/plot_before.png')
    
    cv2.imwrite(res + '/after.jpg', gr)
    ax = plt.hist(gr.ravel(), 256, [0,256])
    fig2 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig(res + '/plot_after.png')
    
    
    # b=np.ones(len(uniq))
    # ck = np.column_stack([best_sol, b])
    # vif = [variance_inflation_factor(ck, i) for i in range(ck.shape[1])]
    
    vif[img_no -1]=vifp(source, gr)
    psnr[img_no -1]=PSNR(source,gr)
    ssim_val[img_no -1]=ssim(source, gr)
    
    ######### the parameter values ##########
    print("psnr = "+str(psnr[img_no -1]))
    print("vif = "+str(vif[img_no -1]))
    print("ssim = "+str(ssim_val[img_no -1]))
    
    
    file1 = open("D:/Project/Image Enhancement/results/DA_Nt_del_h_obj_30/results.txt", "a")  
    file1.write(str(img_no).zfill(2) + ". \n\n")
    file1.write("psnr = "+ str(psnr[img_no -1]) + "\n")
    file1.write("vif = "+str(vif[img_no -1]) + "\n")
    file1.write("ssim = "+str(ssim_val[img_no -1]) + "\n\n\n")
    file1.close()
    
file1 = open("D:/Project/Image Enhancement/results/DA_H_del_h_obj_30/results.txt", "a")
file1.write("avg psnr = "+ str(sum(psnr)/len(psnr)) + "\n")
file1.write("avg vif = "+str(sum(vif)/len(vif)) + "\n")
file1.write("avg ssim = "+str(sum(ssim_val)/len(ssim_val)) + "\n\n\n")
file1.close() 
    
    