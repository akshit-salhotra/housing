import numpy as np
import csv
import matplotlib.pyplot as plt

house="Housing.csv"
w=np.zeros(shape=(1,12),dtype="uint64")
predict=np.zeros(shape=(545))
area=np.zeros(shape=(545))
b=0
price=np.zeros(shape=(545))

data=np.zeros(shape=(545, 12), dtype="uint64")
with open(house, 'r', newline='') as csv_file:

    csv_reader = csv.reader(csv_file)
    i=0
    for row in csv_reader:
        if i == 0:
            i+=1
            continue
        price[i-1]=int(row[0])
        for f in range(1,13):
            data[i-1][f-1]=int(row[f])
        i+=1

#for i in range(545):
   # print(data[i]);
#for i in range(545):
    #print(price[i]);





def compute_gradient(w,b,x,alpha):
    tw=np.zeros(shape=(12),dtype="int64")
    tb=0
    for i in range(545):
        for j in range(12):
            tw[j] += (1 / 545) * (np.dot(w, x[i]) + b - price[i]) * x[i][j]
        tb += (1 / 545) * (np.dot(w, x[i]) + b - price[i])
    for i in range(12):
        w[i]=w[i]-alpha*tw[i]
    b=b-alpha*tb

    return w,b


def compute_cost(w,b,x):
    cost=0

    for i in range(545):
        cost+=(1/545)*((np.dot(w,x[i])+b)-price[i])**2
    print(cost)
    return cost


def descent_gradient():
    iterations=1000
    alpha=0.000000001
    w = [8000,3,2,2,0,1,0,1,1,0,1,2]
    b=1000
    for i in range(iterations):
        c=compute_cost(w,b,data)
        w,b=compute_gradient(w,b,data,alpha)
        #print(i);
    for i in range(545):
        predict[i]=np.dot(w,data[i])+b
        print(f"price:{predict[i]}  and cost :{price[i]}")
    area=data[:,0];
    plt.plot(area,predict,c='b',label="prediction")
    plt.scatter(area,price,c='r',label='actual')
    plt.title("Housing Prices")
    plt.ylabel('Price (in 1000s of dollars)')
    plt.xlabel('Size')
    plt.legend()
    plt.show()



descent_gradient()






















       


