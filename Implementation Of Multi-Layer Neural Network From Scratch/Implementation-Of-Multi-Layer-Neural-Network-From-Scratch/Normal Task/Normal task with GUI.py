#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tkinter import *
from tkinter import ttk as ttk
import pandas as pd
import numpy as np
from scipy.special import expit
import ast


# In[10]:


class task_3():
    
    def preprocess_data(self):
        df = pd.read_csv("penguins.csv")
        
        df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
        df['gender'] = df['gender'].replace({'male': 0, 'female': 1})
        df['gender'] = df['gender'].astype(int)
        
        df['species'] = df['species'].replace({'Adelie': 0, 'Chinstrap': 1 ,'Gentoo':2})
        df['species'] = df['species'].astype(int)
        
        df_Adelie=df[df.iloc[:, 0] == 0]
        df_Gentoo=df[df.iloc[:, 0] == 2]
        df_Chinstrap=df[df.iloc[:, 0] == 1]
        
        df_Adelie_train=df_Adelie.sample(frac=0.6, random_state=10)
        df_Adelie_test=df_Adelie.drop(df_Adelie_train.index)
            
        df_Gentoo_train=df_Gentoo.sample(frac=0.6, random_state=10)
        df_Gentoo_test=df_Gentoo.drop(df_Gentoo_train.index)
        
        
        df_Chinstrap_train = df_Chinstrap.sample(frac=0.6, random_state=10)
        df_Chinstrap_test=df_Chinstrap.drop(df_Chinstrap_train.index)
        
        df_train = pd.concat([df_Adelie_train, df_Gentoo_train, df_Chinstrap_train], axis=0, ignore_index=True)
        df_test = pd.concat([df_Adelie_test, df_Gentoo_test, df_Chinstrap_test], axis=0, ignore_index=True)
        
        X_train= df_train.iloc[:, 1:]
        
        Y_train=df_train.iloc[:, 0]
        
        X_test =df_test.iloc[:, 1:]
        
        Y_test = df_test.iloc[:, 0]
        
        X_train=X_train.to_numpy()
        X_test=X_test.to_numpy()
        Y_train = Y_train.to_numpy()
        Y_test = Y_test.to_numpy()
        
#         train_num = X_train.shape[0]
#         train_indices = np.random.permutation(train_num)
#         X_train=X_train[train_indices]
#         Y_train=Y_train[train_indices]
        
#         test_num=X_test.shape[0]
#         test_indices = np.random.permutation(test_num)
#         X_test=X_test[test_indices]
#         Y_test=Y_test[test_indices]
        
        X_train=X_train.T
        X_test=X_test.T
        
        
        Y_train = np.resize(Y_train , (Y_train.shape[0],1))
        ecoding_train_y = np.zeros((Y_train.shape[0], 3))
        ecoding_train_y[np.arange(Y_train.shape[0]), Y_train.flatten()] = 1
        Y_train = ecoding_train_y.astype(int)
        Y_train=Y_train.T
        
        
        Y_test = np.resize(Y_test , (Y_test.shape[0],1))
        ecoding_test_y = np.zeros((Y_test.shape[0], 3))
        ecoding_test_y[np.arange(Y_test.shape[0]), Y_test.flatten()] = 1
        Y_test = ecoding_test_y.astype(int)
        Y_test =Y_test.T
        
        
        
        return X_train ,Y_train ,X_test,Y_test
        
    
    
    def activation(self,x,actv):
        if actv == "sigmoid":
            return expit(x)

        elif actv=="tanh":
            return np.tanh(x)
            
        
    
    def prepare_layers(self ,hidden_layers , X_train ):
        np.random.seed(1)
        layers_dimentions = []
        layers_dimentions.append(X_train.shape[0])
        for i in range(0,len(hidden_layers)):
             layers_dimentions.append(hidden_layers[i])
        layers_dimentions.append(3)
        parameters={}
        for i in range(1,len(layers_dimentions)):
            parameters['W' + str(i)] = np.random.randn(layers_dimentions[i],layers_dimentions[i-1] ) *0.01
            parameters['b' + str(i)] = np.zeros((layers_dimentions[i],1))
        
        return parameters
    
    def forward(self ,X_train,parameters,activ,bias_or_not):
        A = X_train
        L=len(parameters) // 2
        caching={'A0':A}
        for i in range(1,L+1):
            W = parameters['W' + str(i)]
            b = parameters['b' + str(i)]
            if bias_or_not == True:
               
                
                z=np.dot(W,A)+b
            else:
                z=np.dot(W,A)
                
            caching['Z' + str(i)]=z
            if activ == "sigmoid":
                A= self.activation(z,"sigmoid")
            else:
                A= self.activation(z,"tanh")
                
            caching['A' + str(i)]=A
        
        return caching ,A
    

        
    def transform_output(self,Y):
        modified_lists = []
        for lst in Y:
            max_prob = max(lst)
            modified_lst = [1 if prob == max_prob else 0 for prob in lst]
            modified_lists.append(modified_lst)
        modified_lists = np.array(modified_lists)
        return modified_lists

    def backward(self, parameters,Y_train ,caching ):
        gradiants={}
        L = len(parameters)//2
       
        dZ=caching["A"+str(L)] - Y_train
        gradiants['dW' + str(L)] = np.dot(dZ,caching['A' + str(L-1)].T) / Y_train.shape[1]
        gradiants['db' + str(L)] = np.sum(dZ, axis=1, keepdims=True) / Y_train.shape[1]
        
        for i in range(L-1,0,-1):
            dA = np.dot( parameters['W' + str(i+1)].T,dZ)
            dZ = dA * caching['A' + str(i)] * (1 - caching['A' + str(i)])
            gradiants['dW' + str(i)] = np.dot(dZ,caching['A' + str(i-1)].T ) / Y_train.shape[1]
            gradiants['dW' + str(i)] += (0.001 / Y_train.shape[1])*gradiants['dW' + str(i)]
            gradiants['db' + str(i)] = np.sum(dZ, axis=1, keepdims=True) / Y_train.shape[1]
        return gradiants
         
    def update(self,parameters, gradiants, learn_rate):
        L = len(parameters) // 2
        for i in range(1, L+1):
            parameters['W' + str(i)] -= learn_rate * gradiants['dW' + str(i)]
            parameters['b' + str(i)] -= learn_rate * gradiants['db' + str(i)]
        return parameters
    
    def train(self ,X_train,Y_train,activ,bias_or_not,learn_rate,epoches,hidden_layers):
        parameters = self.prepare_layers(hidden_layers,X_train)
        for i in range(0,epoches):
            caching,_=self.forward(X_train,parameters,activ,bias_or_not)
            gradiants=self.backward(parameters,Y_train,caching)
            parameters=self.update(parameters, gradiants, learn_rate)
        
        return parameters
    
    def test(self,X_test,parameters,activ,bias_or_not):
        caching,A = self.forward(X_test,parameters,activ,bias_or_not)
        
        result=self.transform_output(A.T)
        return result
    
    def accuracy(self,result,labels):
        #for class 0
        tp=0
        fp=0
        tn=0
        fn=0
        labels=self.transform_output(labels.T)
        
        for i in range(0,60):
            if np.argmax(result[i]) == np.argmax(labels[i]):
                tp+=1
            elif np.argmax(result[i]) == 0 and np.argmax(labels[i])!=0:
                fp+=1
            
            elif np.argmax(result[i]) != 0 and np.argmax(labels[i])!=0:
                tn+=1     
                
            elif np.argmax(result[i]) != 0 and np.argmax(labels[i])==0:
                fn+=1
                
        accuracy = (tp / 60 ) * 100
        
        return tp,fp,tn,fn,accuracy
                
               


# In[13]:


class GUI():
        
    def load_gui(self):
        self.root = Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.screen_x = (screen_width / 2) - (500 / 2)
        self.screen_y= (screen_height / 2) - (800 / 2)
        self.root.geometry("%dx%d+%d+%d" % (500, 800, 0, 0))



        activation_label = Label(self.root, text="Choose Activate Function :")
        activation_label.place(x=10 , y=50)
        activations = ["Sigmoid", "Tanh"]
        activation_combo = ttk.Combobox(self.root , values = activations)
        activation_combo.place (x=200 ,y=50)
        
        learning_rate_label = Label(self.root, text="Enter Learning Rate :")
        learning_rate_label.place(x=10 , y=90)
        learning_rate_input = Entry(self.root)
        learning_rate_input.place(x=200 , y=90)
        
        epoches_label = Label(self.root, text="Enter Epoches :")
        epoches_label.place(x=10 , y=130)
        epoches_input = Entry(self.root)
        epoches_input.place(x=200 , y=130)
        
        layer_label = Label(self.root, text="Enter Hidden Layers :")
        layer_label.place(x=10 , y=170)
        layer_input = Entry(self.root)
        layer_input.place(x=200 , y=170)
        
        bias_choice = BooleanVar()
        bias_input = Checkbutton(self.root, text="Bias or Not ?" ,variable=bias_choice)
        bias_input.place(x=200 , y=210)
        
        run_btn = Button(self.root , text="Run" ,width=50 ,fg="green" ,command=lambda:self.run_button(activation_combo.get(),learning_rate_input.get(),epoches_input.get(),layer_input.get(),bias_choice.get()))
        run_btn.place(x=50 , y=250)
        self.root.mainloop()
        
    def run_button(self,activation,learning_rate,epoches,layer_input,bias):
      
        activ = activation
        bias_or_not = bias
        learn_rate=float(learning_rate)
        epoches=int(epoches)
        hidden_layers = ast.literal_eval(layer_input)
        
        
        obj = task_3()
        X_train ,Y_train ,X_test,Y_test = obj.preprocess_data()
        

        parameters=obj.train(X_train,Y_train,activ,bias_or_not,learn_rate,epoches,hidden_layers)
    
            
        result=obj.test(X_test,parameters,activ,bias_or_not)
        tp,fp,tn,fn,accuracy = obj.accuracy(result,Y_test)
        
        self.second_window(tp,fp,tn,fn,accuracy)
        
        
    def second_window(self,tp,fp,tn,fn,accuracy):
        self.new_window_1 = Tk()
        self.new_window_1.geometry("%dx%d+%d+%d" % (500, 800, 0, 0))
        
        #--------------------
        tp_label=Label(self.new_window_1, text="True Positive")
        tp_label.place(x=10 , y=60)
        
        tp_label_v=Label(self.new_window_1, text=str(tp))
        tp_label_v.place(x=100 , y=60)    
        #--------------------
        
        tn_label=Label(self.new_window_1, text="True Negative")
        tn_label.place(x=300 , y=60)
        
        tn_label_v=Label(self.new_window_1, text=str(tn))
        tn_label_v.place(x=400 , y=60)
        #--------------------
        
        fp_label=Label(self.new_window_1, text="False Positive")
        fp_label.place(x=10 , y=90)
        
        fp_label_v=Label(self.new_window_1, text=str(fp))
        fp_label_v.place(x=100 , y=90)
        #--------------------
                
        fn_label=Label(self.new_window_1, text="False Negative")
        fn_label.place(x=300 , y=90)
        
        fn_label_v=Label(self.new_window_1, text=str(fn))
        fn_label_v.place(x=400 , y=90)
        #------------------------------------------------------------------------------------------
 
        
        err = Label(self.new_window_1, text="Accuracy is ")
        err.place(x=10 ,y=150)
        err_lbl = Label(self.new_window_1, text=str(accuracy))
        err_lbl.place(x=110 ,y=150)


# In[14]:


gui = GUI()
gui.load_gui()


# In[ ]:


# result (tanh) |  epoches 5000| learn rate 0.01  | layers [10,20,50] | bias
#max_accuracy = 57 |    tp=   34   |     fp=  3  |    tn=   3    |         fn= 20

# result (tanh) |  epoches 5000| learn rate 0.01  | layers [10,20,50] | no bias
#max_accuracy = 34|    tp=  20    |     tn= 20   |    fp=   0   |         fn=20

# result (sigmoid)|   epoches 5000 | learn rate 0.01  | layers [10,20,50] | bias
#max_accuracy = 57|    tp= 34     |     tn=  3  |    fp=   3   |         fn= 20

# result (sigmoid) |  epoches 5000| learn rate 0.01 | layers [10,20,50] | no bias
#max_accuracy = 34 |    tp=   20   |     tn=  20  |    fp=  0    |         fn= 20

