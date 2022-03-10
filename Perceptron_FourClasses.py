import numpy as np
import matplotlib.pyplot as plt
class Perceptron(object):
    def __init__(self,total_inputs,epoach=100,learning_rate=0.5):
        self.num_inputs=total_inputs
        self.epoach=epoach
        self.learning_rate=learning_rate
        self.weights=np.zeros(total_inputs+1)
    def predict(self,inputs):
        y=np.dot(inputs,self.weights[1:])+self.weights[0]
        return 1.0 if y>=0.0 else 0.0
    def train(self, training_inputs, labels):
        for i in range(self.epoach):
            for inputs,label in zip(training_inputs,labels):
                prediction=self.predict(inputs)
                self.weights[1:]+=self.learning_rate*(label-prediction)*inputs
                self.weights[0]+=self.learning_rate*(label-prediction)
        return self.weights

x_plot=[5,5,6,6,7]
y_plot=[5,7,3,4,-2.5]
s_plot=[6,6,7,3,2,1]
t_plot=[3,4,-2.5,3,2,-1.2]
g_plot=[3,2,2.5,3,2,1]
h_plot=[6,5,5,3,2,-1.2]

label1=np.array([0,0,1,1,1])
label2=np.array([0,0,0,1,1,1])
label3=np.array([0,0,0,1,1,1])

train_input1=[]
train_input2=[]
train_input3=[]

for i in range(5):
    train_input1.append(np.array([x_plot[i],y_plot[i]]))
for j in range(6):
    train_input2.append(np.array([s_plot[j],t_plot[j]]))
    train_input3.append(np.array([g_plot[j],h_plot[j]]))

perceptron=Perceptron(2)
i=np.linspace(0,8)
j=np.linspace(0,8)
k=np.linspace(0,8)

a=perceptron.train(train_input1,label1)

line1=((-a[1]/a[2])*i-a[0]/a[2])
fig=plt.figure()
plt.plot(x_plot,y_plot,color='k',linewidth=0,marker='.')
for xx,yy in zip(x_plot,y_plot):
    plt.text(xx-0.3,yy,str(xx))
    plt.text(xx,yy,str(yy))
plt.plot(j,line1,color="r",linewidth=2)


b=perceptron.train(train_input2,label2)
line2=((-b[1]/b[2])*i-b[0]/b[2])
plt.plot(s_plot,t_plot,color='k',linewidth=0,marker='.')
for ss,tt in zip(s_plot,t_plot):
    plt.text(ss-0.3,tt,str(ss))
    plt.text(ss,tt,str(tt))
plt.plot(j,line2,color="y",linewidth=2)

c=perceptron.train(train_input3,label3)
line3=((-c[1]/c[2])*i-c[0]/c[2])
plt.plot(k,line3,color="b",linewidth=2)
plt.plot(g_plot,h_plot,color='k',linewidth=0,marker='.')
for gg,hh in zip(g_plot,h_plot):
    plt.text(gg-0.3,hh,str(gg))
    plt.text(gg,hh,str(hh))
    
plt.title('Four Classes')
plt.xlabel('p1')
plt.ylabel('p2')
plt.show()