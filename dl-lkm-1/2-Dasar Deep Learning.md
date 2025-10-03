

| Panduan Praktikum Mata Kuliah Deep Learning Program Studi Informatika, Fakultas Ilmu Komputer Universitas Jember |
| :---: |

| Praktikum Ke- | : | 2 |
| :---- | :---- | :---- |
| Judul | : | Dasar Deep Learning |
| NIM | : |  |
| Nama | : |  |
| Kelas | : |  |

1. Neuron Tunggal

| Code: |
| :---- |
| import numpy as np \# Input x \= np.array(\[2, 3\])  \# dua fitur input w \= np.array(\[0.5, \-0.4\])  \# bobot b \= 0.1  \# bias  \# Fungsi aktivasi sigmoid def sigmoid(z):     return 1 / (1 \+ np.exp(-z)) \# Hitung output neuron z \= np.dot(x, w) \+ b a \= sigmoid(z) print("z (kombinasi linear):", z) print("Output setelah sigmoid:", a)  |
| **Output dan Analisis:** |
|  |

2. Membandingkan Aktivasi

| Code: |
| :---- |
| import matplotlib.pyplot as plt z \= np.linspace(\-5, 5, 100) \# Definisi fungsi aktivasi relu \= lambda x: np.maximum(0, x) sigmoid \= lambda x: 1 / (1 \+ np.exp(-x)) plt.plot(z, relu(z), label="ReLU") plt.plot(z, sigmoid(z), label="Sigmoid") plt.legend() plt.title("Perbandingan Fungsi Aktivasi") plt.show()  |
| **Output dan Analisis:** |
|  |

3. Membuat Neuron Sederhana

| Code: |
| :---- |
| import torch import torch.nn as nn import torch.optim as optim \# Data OR gate X \= torch.tensor(\[\[0.,0.\],\[0.,1.\],\[1.,0.\],\[1.,1.\]\]) Y \= torch.tensor(\[\[0.\],\[1.\],\[1.\],\[1.\]\]) \# Definisikan model sederhana: 2 input \-\> 1 output model \= nn.Sequential(     nn.Linear(2, 1),   \# neuron: 2 input \-\> 1 output     nn.Sigmoid()       \# fungsi aktivasi ) \# Loss function dan optimizer criterion \= nn.BCELoss() optimizer \= optim.SGD(model.parameters(), lr=0.1) \# Training loop for epoch in range(1000):     y\_pred \= model(X)     loss \= criterion(y\_pred, Y)     optimizer.zero\_grad()     loss.backward()     optimizer.step() \# Cek hasil print("Prediksi setelah training:") print(model(X).detach())    |
| **Output dan Analisis:** |
|  |

**Pertanyaan Untuk Soal 1-3**  
	1\. Apa yang terjadi pada output neuron jika bias dihilangkan?  
	2\. Mengapa fungsi aktivasi Sigmoid cocok untuk kasus probabilitas?  
	3\. Apa yang terjadi jika learning rate terlalu besar?  
	4\. Apakah model bisa digunakan untuk AND Gate atau XOR Gate?  
	

4. MLP

| Code: |
| :---- |
| import torch import torch.nn as nn import torch.optim as optim import torchvision import torchvision.transforms as transforms import matplotlib.pyplot as plt\# Transformasi: ubah gambar ke tensor & normalisasi transform \= transforms.Compose(\[transforms.ToTensor(),                                 transforms.Normalize((0.5,), (0.5,))\]) \# Download dataset trainset \= torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) testset \= torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) trainloader \= torch.utils.data.DataLoader(trainset, batch\_size=64, shuffle=True) testloader \= torch.utils.data.DataLoader(testset, batch\_size=64, shuffle=False) \# Cek contoh data images, labels \= next(iter(trainloader)) plt.imshow(images\[0\].squeeze(), cmap="gray") plt.title(f"Label: {labels\[0\]}") plt.show() \# implementasi neouron tunggalimport torch.nn as nn import torch.nn.functional as F class SingleNeuron(nn.Module): def \_\_init\_\_(self, activation\="sigmoid"): super(SingleNeuron, self).\_\_init\_\_() self.fc \= nn.Linear(28\*28, 1) \# 784 \-\> 1 self.activation \= activation def forward(self, x): x \= x.view(\-1, 28\*28) \# flatten z \= self.fc(x) if self.activation \== "sigmoid": return torch.sigmoid(z) elif self.activation \== "tanh": return torch.tanh(z) elif self.activation \== "relu": return F.relu(z) else: return z \# identitas \# contoh forwaard passmodel \= SingleNeuron(activation="sigmoid") \# Ambil batch data images, labels \= next(iter(trainloader)) \# Ubah label ke binary (misal: deteksi digit 0\) labels\_binary \= (labels \== 0).float().unsqueeze(1) outputs \= model(images) print("Output probabilitas (batch 1):", outputs\[:10\].detach().squeeze()) print("Label asli:", labels\[:10\]) print("Label binary:", labels\_binary\[:10\].squeeze()) \# los function & optimizercriterion \= nn.BCELoss() optimizer \= torch.optim.SGD(model.parameters(), lr=0.01) for epoch in range(1): \# 1 epoch cukup untuk demo running\_loss \= 0.0 for images, labels in trainloader: labels\_binary \= (labels \== 0).float().unsqueeze(1) optimizer.zero\_grad() outputs \= model(images) loss \= criterion(outputs, labels\_binary) loss.backward() optimizer.step() running\_loss \+= loss.item() print(f"Epoch {epoch+1}, Loss: {running\_loss/len(trainloader):.4f}") correct, total \= 0, 0 with torch.no\_grad(): for images, labels in testloader: labels\_binary \= (labels \== 0).float().unsqueeze(1) outputs \= model(images) predicted \= (outputs \>= 0.5).float() total \+= labels\_binary.size(0) correct \+= (predicted \== labels\_binary).sum().item() print(f"Akurasi deteksi digit '0' vs bukan '0': {100 \* correct / total:.2f}%")  |
| **Output dan Analisis:** |
|  |

**Pertanyaan Untuk Soal 4**  
 **\-** Ubah parameter activation pada model menjadi "tanh" atau "relu". Bandingkan hasil akurasi.  
	1\. Apa perbedaan bentuk output sigmoid vs tanh vs ReLU?  
	2\. Mengapa ReLU cenderung bekerja lebih baik pada jaringan dalam?  
	3\. Apa risiko menggunakan sigmoid pada data dengan banyak kelas?  
	

\*\*Dilarang keras plagiat dengan teman dan men-copy langsung jawaban dari ai apapun.