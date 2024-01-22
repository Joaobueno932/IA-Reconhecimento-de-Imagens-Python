#DenseNet-121
modelo = "DenseNet-121"

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

# Função para carregar as imagens e anotações
def load_data(root_dir):
    image_dir_train = os.path.join(root_dir, 'train/images')
    annotation_dir_train = os.path.join(root_dir, 'train/anotacoes')


    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar todas as imagens para o tamanho 224x224
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(
        root=image_dir_train,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=48,
        shuffle=True
    )
    return train_loader

def test_load_data(root_dir):
    image_dir_test = os.path.join(root_dir, 'test/images')
    annotation_dir_test = os.path.join(root_dir, 'test/anotacoes')

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
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def validation_load_data(root_dir):  # Corrigir o nome do parâmetro para root_dir
    image_dir_val = os.path.join(root_dir, 'valid/images')
    annotation_dir_val = os.path.join(root_dir, 'valid/anotacoes')

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

def train_model(train_loader, val_loader, num_epochs,MODELO):
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_accuracy = 0.0  # Acompanha a melhor precisão do teste
    train_accuracies = []  # Lista para armazenar precisões do treinamento
    train_losses = []  # Lista para armazenar perdas do treinamento
    best_val_accuracy = 0.0  # Acompanha a melhor precisão da validação

    val_accuracies = []  # Lista para armazenar a acurácia da validação



    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        train_accuracy = evaluate_model(model, train_loader)
        train_accuracies.append(train_accuracy)

        # Validação após cada época
        model.eval()
        val_accuracy = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2%}, Validation Accuracy: {val_accuracy:.2%}, Loss: {epoch_loss:.4f}')

        # Salva o melhor modelo baseado na validação com a maior acurácia 
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict()

    # Salva o melhor modelo
    torch.save(best_model_state, MODELO)

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
    plt.savefig(modelo+'curva_roc'+palavroto+'.png')  # Salvar a matriz de confusão para ORIGA em um arquivo PNG
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
    plt.savefig(modelo+'curva_rc'+palavroto+'.png')  # Salvar a matriz de confusão para ORIGA em um arquivo PNG
    plt.close()  # Fechar o gráfico após salvar

    # Relatório de classificação
    print("Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))

#"C:\\Users\\Kevin!\\Desktop\\talho\\G1020"
Valid_root = "Caminho ORIGA"
Teste_root = "Caminho G1020"
num_epochs = 10

Val_loader = validation_load_data(Valid_root)
Test_loader = test_load_data(Teste_root)

print('\n')
# Carregar os dados para G1020
print("G1020:")
TrainG1020_root = "Caminho G1020"
Train_loader_G1020 = load_data(TrainG1020_root)
G1020Nome = 'G1020'
G1020M = 'G1020best_model.pth'
MG1020M = modelo+G1020M
modelo_G1020 = train_model(Train_loader_G1020, Val_loader, num_epochs,MG1020M)
print("Fim do treino.")
melhor_estado_modelo_G1020 = torch.load(MG1020M)
modelo_G1020.load_state_dict(melhor_estado_modelo_G1020)
test_model(modelo_G1020, Test_loader,G1020Nome)
print("Fim do teste.")
print('\n')

print('\n')
# Carregar os dados para ORIGA
print("ORIGA:")
TrainORIGA_root = "Caminho ORIGA"
train_loader_ORIGA = load_data(TrainORIGA_root)
print("Fim do treino.")
ORIGANome = 'ORIGA'
ORIGM = 'ORIGAbest_model.pth'
MORIGM = modelo+ORIGM
modelo_ORIGA = train_model(train_loader_ORIGA, Val_loader, num_epochs,MORIGM)
melhor_estado_modelo_ORIGA = torch.load(MORIGM)
modelo_ORIGA.load_state_dict(melhor_estado_modelo_ORIGA)
test_model(modelo_ORIGA, Test_loader,ORIGANome)
print("Fim do teste.")
print('\n')

print('\n')
# Carregar os dados para REFUGE
print("REFUGE:")
TrainREFUGE_root = "Caminho REFUGE"
train_loader_REFUGE = load_data(TrainREFUGE_root)
print("Fim do treino.")
REFUGENome = 'REFUGE'
REFUM = 'REFUGEbest_model.pth'
MREFUM = modelo+REFUM
modelo_REFUGE = train_model(train_loader_REFUGE, Val_loader, num_epochs,MREFUM)
melhor_estado_modelo_REFUGE = torch.load(MREFUM)
modelo_REFUGE.load_state_dict(melhor_estado_modelo_REFUGE)
test_model(modelo_REFUGE, Test_loader,REFUGENome)
print("Fim do teste.")
print('\n')
