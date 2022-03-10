import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self,epoch=50,learning_rate=0.5): #初始化,跑 50 次,learning rate設 0.5
        self.num_test=epoch
        self.learning_rate=learning_rate
        self.weights=[0.0,0.0,0.0]
    def predict(self,inputs):
        y = np.dot(inputs,self.weights[1:])+self.weights[0]
        return 1.0 if y>= 0.0 else 0.0
    def train(self,training_inputs,labels):
        for i in range(self.num_test):
            for inputs, label in zip(training_inputs,labels):
                prediction=self.predict(inputs)
                #renew weight
                self.weights[1:]+=self.learning_rate*(label-prediction)*inputs
                self.weights[0]+=self.learning_rate*(label-prediction)
        return self.weights
        
train_input=[]
x_plot=[2.1,2.3,4.8,3.7,1.5,1.2,-1.2,-3]
y_plot=[1.5,3,2.8,1.7,-0.6,-2.9,0.5,-3.5]
label=np.array([0,0,0,0,1,1,1,1])

for i in range(len(x_plot)):
    train_input.append(np.array([x_plot[i],y_plot[i]]))
perceptron=Perceptron(2)
a=perceptron.train(train_input,label)

i=np.linspace(-5,5)
boundary_line=((-a[1]/a[2])*i-a[0]/a[2])
fig=plt.figure()


plt.title('Two Classes')
plt.xlabel('p1')
plt.ylabel('p2')
plt.plot(x_plot,y_plot,linewidth=0,marker='^')
for x_axis,y_axis in zip(x_plot,y_plot):
    plt.text(x_axis-0.6,y_axis,str(x_axis))
    plt.text(x_axis+0.1,y_axis,str(y_axis))
plt.plot(i,boundary_line,linewidth=3)
plt.show()