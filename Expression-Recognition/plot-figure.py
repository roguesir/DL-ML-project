import matplotlib.pyplot as plt 
import numpy as np

x = np.linspace(1,501,500)
item_loss1 = []
item_acc1 = []
with open('value-basic.txt','r') as f:
	data = f.readlines()
	for line in data:
		#print(line[1])
		item = line.split('\t')
		#item_loss = map(float, item[0])
		#item_acc = map(float, item[1])
		item_loss1.append(float(item[0]))
		item_acc1.append(float(item[1]))
	#print(item_loss)

item_loss2 = []
item_acc2 = []
with open('value-8.txt','r') as f:
	data = f.readlines()
	for line in data:
		#print(line[1])
		item = line.split('\t')
		#item_loss = map(float, item[0])
		#item_acc = map(float, item[1])
		item_loss2.append(float(item[0]))
		item_acc2.append(float(item[1]))
	#print(item_loss)

item_loss3 = []
item_acc3 = []
with open('value-11.txt','r') as f:
	data = f.readlines()
	for line in data:
		#print(line[1])
		item = line.split('\t')
		#item_loss = map(float, item[0])
		#item_acc = map(float, item[1])
		item_loss3.append(float(item[0]))
		item_acc3.append(float(item[1]))
	#print(item_loss)

item_loss4 = []
item_acc4 = []
with open('value-13-conv.txt','r') as f:
	data = f.readlines()
	for line in data:
		#print(line[1])
		item = line.split('\t')
		#item_loss = map(float, item[0])
		#item_acc = map(float, item[1])
		item_loss4.append(float(item[0]))
		item_acc4.append(float(item[1]))

#print(item_loss1[1])
plt.figure()
plt.plot(x, item_loss1, 'r',label='basic')
plt.plot(x, item_loss2, 'g',label='model-8')
plt.plot(x, item_loss3, 'b',label='model-13')
plt.plot(x, item_loss4, 'c',label='model-13-conv')
for i in range(500):
	if (i+1) % 50 == 0:
		#plt.scatter(x[i],item_loss1[i])
		#plt.scatter(x[i],item_loss2[i])
		#plt.scatter(x[i],item_loss3[i])
		plt.plot(x[i],item_loss1[i],'ro-',x[i],item_loss2[i],'g+-',x[i],item_loss3[i],'b^-',x[i],item_loss4[i], 'c*-')

plt.xlabel(r'$iteration$')
plt.ylabel(r'$loss$')
plt.legend()
plt.show()

plt.figure()
plt.plot(x, item_acc1, 'r', label='basic')
plt.plot(x, item_acc2, 'g', label='model-8')
plt.plot(x, item_acc3, 'b', label='model-13')
plt.plot(x, item_acc4, 'c',label='model-13-conv')
for i in range(500):
	if (i+1) % 50 == 0:
		#plt.scatter(x[i],item_loss1[i])
		#plt.scatter(x[i],item_loss2[i])
		#plt.scatter(x[i],item_loss3[i])
		plt.plot(x[i],item_acc1[i],'ro-',x[i],item_acc2[i],'g+-',x[i],item_acc3[i],'b^-',x[i],item_acc4[i], 'c*-')
plt.xlabel(r'$iteration$')
plt.ylabel(r'$acc$')
plt.legend()
plt.show()



