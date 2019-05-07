
####........import statement.........#########

from copy import deepcopy
import numpy as np
import pandas as pd
import math
from numpy import array
import random
from math import exp
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')



###.........data import ...........##

df = pd.read_csv('/Users/saurabhghosh/Downloads/xclara.csv',index_col=False)
cols = [1,2]
data= df[df.columns[cols]]


Z=data.shape[0]               #total no. of data poins
print("number of data points :",Z)


###...............UB LB................
UB=[]
LB=[]

a=data['V1'].max()
b=data['V1'].min()
c=data['V2'].max()
d=data['V2'].min()
UB.append(a)
UB.append(c)

LB.append(b)
LB.append(d)
print(a,b,c,d)


#####............data plotting..........###
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))

plt.scatter(f1, f2, c='black', s=7)
plt.show()


##.......max gen..........##
maxgen=10

##.......no of cluster...... ###
k=3


###.......firefly creation...........##
NF=5
F=np.zeros((NF,k,2), dtype=np.float64)


for i in range(NF):
    C_x = np.random.randint(0, 200, size=k)
    C_y = np.random.randint(0, 200, size=k)
    Y = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    F[i]=Y

print("Initial firefly")
print(F)




####.......plotting firefly..........#####


for i in range(NF):
    plt.scatter(F[i][:,0],F[i][:,1], label='True Position',s=150)
    plt.show()



#####......objective function calculation...........####

J=np.zeros((NF), dtype=np.float64)

def cal_objective(num):
    arr = np.ones((Z, k), dtype=np.float64)
    for j in range(Z):
        for i in range(k):
            try:
              dis = math.sqrt(((float(X[j][0] )- F[num][i][0]) ** 2) + ((float(X[j][1]) - F[num][i][1]) ** 2))
              arr[j][i] = dis
            except ValueError:
                continue


    min_dist_clus0 = np.ones((Z), dtype=np.float64)  # stores the nearest cluster distance

    for j in range(Z):
        min = 0
        val=0
        for i in range(k):
            if arr[j][i] < arr[j][min]:
                    val = arr[j][i]
                    min = i

        min_dist_clus0[j] = val



    sum=0
    for i in range(Z):
        sum=sum+min_dist_clus0[i]

    J[num]=math.sqrt(sum)
    return J[num]

for i in range(NF):
    J[i]=cal_objective(i)

print(J)

####.........parameter for firefly algorithm............#####


beta = 0
beta_0 = 1
gamma = 1
alpha = 0.5


####...........Best Firefly Calculation...........#######

obj=[]
for gen in range(maxgen):
    for i in range(5):
        scale = abs((UB[0] - LB[0])+(UB[1]-UB[1]))
        for j in range(5):
            if(J[i]>J[j]):
                r=0
                for l in range(k):
                    r=r+(F[i][l][0] - F[j][l][0]) + (F[i][l][1] - F[j][l][1])
                r = abs(r)
                r = math.sqrt(r)
                betamin=0.2
                gamma=1.0
                beta0 = 1.0
                beta = (beta0 - betamin) * \
                       math.exp(-gamma * math.pow(r, 2.0)) + betamin
                beta = beta_0 * (exp(-gamma) * (r** 2))
               # print(beta)
                for l in range(k):
                    r = random.uniform(0, 1)
                    tmpf = alpha * (r - 0.5) * scale
                    F[i][l][0] = F[i][l][0] * (1.0 - beta) +F[j][l][0] * beta + tmpf
                    F[i][l][1] = F[i][l][1] * (1.0 - beta) + F[j][l][1] * beta + tmpf

            temp = np.zeros((k, 2), dtype=np.float64)
            temp = F[i]
            for l in range(k):
                for s in range(2):
                    if temp[l][s] < LB[s]:
                        temp[l][s] = random.uniform(LB[s],UB[s])

                    if temp[l][s] > UB[s]:
                        temp[l][s] = random.uniform(LB[s],UB[s])


            F[i] = temp
            arr = np.zeros((Z, k), dtype=np.float64)
            for n in range(Z):
                for m in range(k):
                    try:

                        dis = math.sqrt(((float(X[n][0]) - F[i][m][0]) ** 2) + ((float(X[n][1]) - F[i][m][1]) ** 2))
                        arr[n][m] = dis
                    except ValueError:
                        continue
            min_dist_clus = np.zeros((Z), dtype=np.float64)
            for n in range(Z):
                min = 0
                val=0
                for m in range(k):
                    if arr[n][m] < arr[n][min]:
                            val = arr[n][m]
                            min= m

                min_dist_clus[n] = val

            sum = 0
            for n in range(Z):
                sum = sum + min_dist_clus[n]
            J[i]= math.sqrt(sum)

    obj.append(np.amin(J))
    for i in range(1,5):
        if(J[i-1]<J[i]):
            best=i


print("best firefly ")
print(F[best])



######...........ploting best fF(firefly)........#######

plt.scatter(F[best][:,0],F[best][:,1])
plt.show()


#####......... Euclidean Distance Calculator..............#####
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


############################################################
######..............K Means.......................########
###########################################################


#####..............Centroid initialization...........#######

C = F[best]
print("intitial centroid")
print(C)
C_x=[]
C_y=[]

for i in range(k):
    C_x.append(C[i][0])
    C_y.append(C[i][1])



# #########..............Plotting along with the Centroids..................################
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.show()


#####....... To store the value of centroids when it updates...........########
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1)
clusters = np.zeros(len(X))



#######......... Error func. - Distance between new centroids and old centroids.........#######
error = dist(C, C_old, None)

# Loop will run untill the error becomes zero
k_obj=[]
x_axis=[]
x_val=[]
val=200
count=0
while error != 0:
    x_axis.append(count)
    x_val.append(val)
    count += 1
    val+=1

    arr = np.ones((Z, k), dtype=np.float64)
    for j in range(Z):
        for i in range(k):
            try:
                dis = math.sqrt(((float(X[j][0]) - C[i][0]) ** 2) + ((float(X[j][1]) - C[i][1]) ** 2))
                arr[j][i] = dis
            except ValueError:
                continue

    min_dist_clus0 = np.ones((Z), dtype=np.float64)  # stores the nearest cluster distance

    for j in range(Z):
        min = 0
        val = 0
        for i in range(k):
            if arr[j][i] < arr[j][min]:
                val = arr[j][i]
                min = i

        min_dist_clus0[j] = val

    sum = 0
    for i in range(Z):
        sum = sum + min_dist_clus0[i]

    J= math.sqrt(sum)
    obj.append(J)
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    # Storing the old centroid values
    C_old = deepcopy(C)
    #print(C_old)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
print (C)



############............k-means plot.............##############


colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')


plt.show()




#############.............ploting objective function............########
'''
print(obj)
print(k_obj)

obj_0=[]
X=[]
for i in range(maxgen):
    X.append(i)
plt.plot(X,obj)
plt.title('objective_function')
plt.xlabel('no of generation')
plt.ylabel('objective func value')
plt.show()


###########.............ploting K-means error.......#############

plt.plot(x_axis,k_obj)
plt.title('error_function')
plt.xlabel('no of generation')
plt.ylabel('error func value')
plt.show()



for i in range(len(obj)):
		if(i< maxgen):
			plt.plot(i,obj[i])
		else:
			plt.plot(i,obj[i],"r")
plt.show()

'''

###########............ ploting k-means error with firefly......##########
'''x1 = [1, 2, 3]
y1 = [2, 4, 1]
# plotting the line 1 points
plt.plot(x1, y1, label="line 1")

# line 2 points
x2 = [1, 2, 3]
y2 = [4, 1, 3]
# plotting the line 2 points
plt.plot(x2, y2, label="line 2")

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')
# giving a title to my graph
plt.title('Two lines on same graph!')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()'''

for i in range(len(obj)):
		if(i< maxgen):
			plt.plot(i,obj[i],"g *")
		else:
			plt.plot(i,obj[i],"r o")
plt.show()