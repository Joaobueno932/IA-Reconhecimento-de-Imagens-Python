#GoogLeNet
modelo = "GoogLeNet"

import os
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sn            
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
root_dir = "Caminho de imagens de treino G1020"
annotation_dir = "Caminho de anotações de treino G1020"

class SegmentationLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SegmentationLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Converter coeficiente de dados em perda de dados (1 - dados)
        dice_loss = 1 - dice_coeff

        return dice_loss

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, annotation_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = "Caminho de imagens de treino G1020"
        self.annotation_dir = "Caminho de anotações de treino G1020"
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = int(self.data.iloc[idx, 1])

        annotation_name = os.path.join(self.annotation_dir, self.data.iloc[idx, 0])
        annotation = Image.open(annotation_name)
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)
            annotation = self.transform(annotation)

        return image, label, annotation

class CustomDataset2(Dataset):
    def __init__(self, csv_file, root_dir, annotation_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = "Caminho de imagens de treino ORIGA"
        self.annotation_dir = "Caminho de anotações de treino ORIGA"
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        image = Image.open(img_name)
        label = int(self.data.iloc[idx, 4])

        annotation_name = os.path.join(self.annotation_dir, self.data.iloc[idx, 1])
        annotation = Image.open(annotation_name)
        label = int(self.data.iloc[idx, 4])

        if self.transform:
            image = self.transform(image)
            annotation = self.transform(annotation)

        return image, label, annotation

# Exemplo de uso
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



def test_load_data(root_dir):
    image_dir_test = os.path.join(root_dir, 'test\\images')
    annotation_dir_test = os.path.join(root_dir, 'test\\anotacoes')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar todas as imagens para o tamanho 224x224
        transforms.ToTensor()
    ])
    test_data = torchvision.datasets.ImageFolder(
        root=image_dir_test,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=48,
        shuffle=False
    )

    return test_loader



def evaluate_model(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in data_loader:  # Ignorando a anotação com "_"
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def validation_load_data(root_dir):  # Corrigir o nome do parâmetro para root_dir
    image_dir_val = os.path.join(root_dir, 'valid\\images')
    annotation_dir_val = os.path.join(root_dir, 'valid\\anotacoes')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_data = torchvision.datasets.ImageFolder(
        root=image_dir_val,
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=48,
        shuffle=False
    )

    return val_loader

def train_model(train_loader, val_loader, num_epochs, MODELO, palavroto, MM):
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    dropout_rate = 0.5
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, 2)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    best_accuracy = 0.0  # Acompanha a melhor precisão do teste
    train_accuracies = []  # Lista para armazenar precisões do treinamento
    train_losses = []  # Lista para armazenar perdas do treinamento
    best_val_accuracy = 0.0  # Acompanha a melhor precisão da validação
    patience = 25  # Número de épocas sem melhoria para parar o treinamento
    early_stopping_counter = 0
    val_accuracies = []  # Lista para armazenar a acurácia da validação
    val_losses = []  # Lista para armazenar a perda da validação

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, annotations in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            annotations = annotations.to(device)  # Adicione esta linha
        
            optimizer.zero_grad()
        
            outputs = model(inputs)
        
            # Calcule a perda da segmentação (você precisará definir uma função de perda adequada)
            segmentation_loss_function = SegmentationLoss()  # Crie uma instância da classe SegmentationLoss
            segmentation_loss = segmentation_loss_function(inputs, annotations)  # Calcule a perda de segmentação
        
            # Calcule a perda de classificação (cross-entropy)
            classification_loss = criterion(outputs, labels)
        
            loss = segmentation_loss + classification_loss
            loss.backward()
        
            optimizer.step()
        
            running_loss += loss.item()
        

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        train_accuracy = evaluate_model(model, train_loader)
        train_accuracies.append(train_accuracy)

        # Validação após cada época
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2%}, Validation Accuracy: {val_accuracy:.2%}, Loss: {epoch_loss:.4f}')

        # Salva o melhor modelo baseado na validação com a maior acurácia 
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print("UEPA")
            early_stopping_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
           print(f'Early stopping at epoch {epoch+1}')
           break

        
    # Salva o melhor modelo
    torch.save(best_model_state, MM)

    # Mostra a acurácia do treino e validação
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Validation Accuracy')
    plt.savefig(modelo+'Acurácia por época'+palavroto+'.png')  # Salvar a matriz de confusão para ORIGA em um arquivo PNG
    plt.show()
    plt.close()  # Fechar o gráfico após salvar

    # Plot train and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.savefig(modelo+'Perda por época'+palavroto+'.png')  # Salvar a matriz de confusão para ORIGA em um arquivo PNG
    plt.show()
    plt.close()  # Fechar o gráfico após salvar

    return model


def test_model(model, test_loader,palavroto):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define o dispositivo de execução
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Obter as probabilidades de saída (scores)
            scores = F.softmax(outputs, dim=1)
            y_scores.extend(scores[:, 1].cpu().numpy())

    # Calcula a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Acurácia:", accuracy)
    print("Precisão:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    print("True Negative (TN):", tn)
    print("False Positive (FP):", fp)
    print("False Negative (FN):", fn)
    print("True Positive (TP):", tp)
    class_names = ['Classe 0 (Sem Glaucoma)', 'Classe 1 (Com Glaucoma)']

    # Plot da matriz de confusão
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    #plt.show()
    plt.savefig(modelo+'_matriz_confusao_'+palavroto+'.png')  # Salvar a matriz de confusão para ORIGA em um arquivo PNG
    plt.close()  # Fechar o gráfico após salvar

    # Relatório de classificação
    print("Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Calcular as estatísticas ROC
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Plot do gráfico ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Gráfico ROC')
    plt.legend(loc='lower right')
    #plt.show()
    plt.savefig(modelo+'_curvaROC_'+palavroto+'.png')  # Salvar a matriz de confusão para ORIGA em um arquivo PNG
    plt.close()  # Fechar o gráfico após salvar

    # Calcular as estatísticas RC
    precision, recall, thresholds_rc = precision_recall_curve(y_true, y_scores)

    # Plot do gráfico RC
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Curva RC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Gráfico RC')
    plt.legend(loc='lower left')
    #plt.show()
    plt.savefig(modelo+'_curvaRC_'+palavroto+'.png')  # Salvar a matriz de confusão para ORIGA em um arquivo PNG
    plt.close()  # Fechar o gráfico após salvar

    # Relatório de classificação
    print("Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))

num_epochs = 10

print('\n')
# Carregar os dados para G1020
print("G1020:")
TrainG1020_root = "Caminho de imagens de treino G1020"
annotation_dir = "Caminho de anotações de treino G1020"
Valid_root = Teste_root =  "Caminho G1020     Ex: C:\\Users\\Eu\\Desktop\\G1020\\"
#Train_loader_G1020 = dataloader(TrainG1020_root)
csv_file = "Criar caminho pro csv     C:\\Users\\Eu\\Desktop\\G1020\\G1020 (1).csv"  # Substitua pelo caminho para o seu arquivo CSV
root_dir = "Colocar o caminho para o diretório  C:\\Users\\Eu\\Desktop\\G1020"  # Substitua pelo diretório onde as imagens estão localizadas
dataset = CustomDataset(csv_file, root_dir, annotation_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=48, shuffle=True)
Val_loader = validation_load_data(Valid_root)
Test_loader = test_load_data(Teste_root)
G1020M = 'G1020best_modelGX.pth'
MG1020M = modelo+"_"+G1020M
modelo_G1020 = train_model(dataloader, Val_loader, num_epochs,modelo,G1020M,MG1020M)
print("Fim do treino.")
melhor_estado_modelo_G1020 = torch.load(MG1020M)
modelo_G1020.load_state_dict(melhor_estado_modelo_G1020)
test_model(modelo_G1020, Test_loader,G1020M)
print("Fim do teste.")
print('\n')



### Mesmo processo feito acima pro resto



print('\n')
# Carregar os dados para ORIGA
print("ORIGA:")
TrainORIGA_root = "C:\\Users\\Eu\\Desktop\\ORIGA\\train\\imageTA"
annotation_dir = "C:\\Users\\Eu\\Desktop\\ORIGA\\train\\anotacoeTA"
Valid_root = Teste_root = "C:\\Users\\Eu\\Desktop\\ORIGA\\"
#Train_loader_ORIGA = load_data(TrainORIGA_root)
csv_file = "C:\\Users\\Eu\\Desktop\\ORIGA\\train\\OrigaList (1).csv"  # Substitua pelo caminho para o seu arquivo CSV
root_dir = "C:\\Users\\Eu\\Desktop\\ORIGA"  # Substitua pelo diretório onde as imagens estão localizadas
Val_loader = validation_load_data(Valid_root)
Test_loader = test_load_data(Teste_root)
dataset = CustomDataset2(csv_file, root_dir, annotation_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=48, shuffle=True)
Val_loader = validation_load_data(Valid_root)
Test_loader = test_load_data(Teste_root)
ORIGM = 'ORIGAbest_modelGX.pth'
MORIGM = modelo+"_"+ORIGM
modelo_ORIGA = train_model(dataloader, Val_loader, num_epochs,modelo,ORIGM,MORIGM)
print("Fim do treino.")
melhor_estado_modelo_ORIGA = torch.load(MORIGM)
modelo_ORIGA.load_state_dict(melhor_estado_modelo_ORIGA)
test_model(modelo_ORIGA, Test_loader,ORIGM)
print("Fim do teste.")
print('\n')

print('\n')
# Carregar os dados para REFUGE
print("REFUGE:")
TrainREFUGE_root = Valid_root = Teste_root = "/home/Eu/Área de Trabalho/talho/REFUGE"
Train_loader_REFUGE = load_data(TrainREFUGE_root)
Val_loader = validation_load_data(Valid_root)
Test_loader = test_load_data(Teste_root)
REFUM = 'REFUGEbest_modelGX.pth'
MREFUM = modelo+REFUM
modelo_REFUGE = train_model(Train_loader_REFUGE, Val_loader, num_epochs,modelo,REFUM,MREFUM)
print("Fim do treino.")
melhor_estado_modelo_REFUGE = torch.load(MREFUM)
modelo_REFUGE.load_state_dict(melhor_estado_modelo_REFUGE) 
test_model(modelo_REFUGE, Test_loader,REFUM)
print("Fim do teste.")
print('\n')

print('\n')
# Carregar os dados para REFUGE
print("REFUG102ORIGA:")
TrainREFUG102ORIGA_root = Valid_root = Teste_root = "/home/Eu/Área de Trabalho/talho/REFUG102ORIGA"
Train_loader_REFUG102ORIGA = load_data(TrainREFUG102ORIGA_root)
Val_loader = validation_load_data(Valid_root)
Test_loader = test_load_data(Teste_root)
REFUG102ORIGAM = 'REFUG102ORIGAbest_model.pth'
MREFUG102ORIGAM = modelo+REFUG102ORIGAM
modelo_REFUG102ORIGA = train_model(Train_loader_REFUG102ORIGA, Val_loader, num_epochs,modelo,REFUG102ORIGAM,REFUG102ORIGAM)
print("Fim do treino.")
melhor_estado_modelo_REFUG102ORIGA = torch.load(MREFUG102ORIGAM)
modelo_REFUG102ORIGA.load_state_dict(melhor_estado_modelo_REFUG102ORIGA)
test_model(modelo_REFUG102ORIGA, Test_loader,REFUG102ORIGAM)
print("Fim do teste.")
print('\n')
