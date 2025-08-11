#Load
# -----------------------
# Bibliotecas PyTorch
# -----------------------
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Bibliotecas do torchvision (modelos pré-treinados e transformações)
# -----------------------

# -----------------------
# Bibliotecas científicas e de manipulação de dados
# -----------------------
import numpy as np
from tqdm import tqdm

# -----------------------
# Leitura de imagens georreferenciadas e processamento de imagens
# -----------------------
import rioxarray as rxr  # Leitura de imagens satélite ou com georreferência
import cv2               # OpenCV para operações de imagem
import time
import csv
from Utils.Evaluation import apply_metrics_amazon
# -----------------------
# Manipulação de arquivos e sistema
# -----------------------
import os
import glob

# -----------------------
# Bibliotecas adicionais
# -----------------------


# -----------------------
# Modelos
# -----------------------
from Models.Proposed_SegModels import Proposed

# -----------------------
# Loss
# -----------------------
from CombinedLoss import CombinedLoss


#--------------------------------------
# Carregar Amazon Dataset
#--------------------------------------
# Função para carregar e normalizar imagens
def load_images(path, channels=4):
    file_list = sorted(glob.glob(os.path.join(path, "*.tif")))
    images = []
    names = []

    for file in file_list:
        img = np.array(rxr.open_rasterio(file), dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Normalização [0,1
        images.append(img)
        names.append(os.path.basename(file))

    images = np.array(images)
    return images, names

# Função para carregar máscaras
def load_masks(path):
    file_list = sorted(glob.glob(os.path.join(path, "*.tif")))

    # Carregar e inverter máscaras
    masks = [np.array(rxr.open_rasterio(file), dtype=np.float32) for file in file_list]
    masks = np.array(masks)
    names = [file for file in file_list]
    names = np.array(names)

    # Verificar se os valores estão normalizados (0-1) ou (0-255)
    if masks.max() > 1:
        masks = 255 - masks  # Inverte (0↔255)
    else:
        masks = 1 - masks  # Inverte (0↔1)

    return masks, file_list



# Criar um Dataset personalizado para imagens e máscaras
class SatelliteDataset(Dataset):
    def __init__(self, images, masks, filenames=None, transform=None):
        self.images = images
        self.masks = masks
        self.filenames = filenames if filenames is not None else [None] * len(images)  # Garante compatibilidade
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        filename = self.filenames[idx]


        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)


        if self.transform:
            image = self.transform(image)

        if filename is not None:
            return image, mask, filename  # Retorna o nome se fornecido
        else:
            return image, mask


#--------------------------------------------------
# Traning Functions
#--------------------------------------------------

def train_combine(model, train_loader, val_loader, num_epochs, name, path_save):
    # Hiperparâmetros
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = CombinedLoss(alpha=0.1, beta=0.9, smooth=1.66, gamma=4.41)

    # Variáveis de controle
    best_val_loss = float("inf")
    patience = 10
    epochs_no_improve = 0
    early_stop = False

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    csv_path = os.path.join(path_save, f"{name}_training_log.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            masks = torch.squeeze(masks, dim=1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks.long())

            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss inválida - ignorando batch")
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.float() #/ (masks.max() + 1e-8)
                masks = torch.squeeze(masks, dim=1)

                outputs = model(images)
                loss = loss_fn(outputs, masks.long())

                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Val Loss contém NaN ou Inf!")
                    continue

                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Salvar as perdas no CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), os.path.join(path_save, name + ".pth"))
            torch.save(model, os.path.join(path_save, name + "model.pth"))
            print("✅ Modelo melhorado e salvo!")
        else:
            epochs_no_improve += 1
            print(f"⏳ Sem melhoria há {epochs_no_improve} época(s).")

            if epochs_no_improve >= patience:
                early_stop = True
                print(f"⛔ Early stopping ativado após {epoch} épocas sem melhoria.")
                break

    # Tempo total
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"⏱️ Tempo total de treinamento: {int(mins)}m {int(secs)}s")

    # Salvar tempo no .txt
    with open(os.path.join(path_save, "training_time.txt"), "w") as f:
        f.write(f"Tempo total de treinamento: {int(mins)}m {int(secs)}s\n")

    # Salvar tempo total no .csv
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Tempo total de treinamento", f"{int(mins)}m {int(secs)}s"])


#------------------------------------------------------
# Funções de Teste
#-------------------------------------------------------
def segment_and_time_combine(path_model, test_loader, path_save_segmented):
    # Limpar cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Carregar modelo
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device)
    model.eval()

    times = []

    # Aquecimento
    with torch.no_grad():
        for image, _, _ in test_loader:
            image = image.to(device)
            _ = model(image)
            torch.cuda.synchronize()
            break

    # Loop principal
    with torch.no_grad():
        for image, _, filename in tqdm(test_loader, desc="Segmentando imagens"):
            image = image.to(device)

            torch.cuda.synchronize()
            start_time = time.time()

            output = model(image)

            torch.cuda.synchronize()
            end_time = time.time()

            # Tempo
            execution_time = end_time - start_time
            times.append(execution_time)

            # Processar e salvar
            _, predicted = torch.max(output.data, 1)
            output_np = 255 * np.squeeze(predicted.cpu().numpy()).astype(np.uint8)

            original_name = filename[0].replace('.tif', '.png')
            save_path = os.path.join(path_save_segmented, original_name)
            cv2.imwrite(save_path, output_np)

    print(f"✅ Segmentação concluída! Imagens salvas em: {path_save_segmented}")
    print(f"⏱️ Tempo médio de execução: {np.mean(times):.6f} segundos")
    return times





if __name__ == "__main__":
    PATH_CLUSTER = ""
    PATH_ROOT = os.path.join(PATH_CLUSTER, "AMAZON_tests/")
    PATH_DATASET = os.path.join(PATH_CLUSTER, "Datasets/AMAZON/")
    PATH_GT = os.path.join(PATH_DATASET, "Test/mask_png/")

    PATH_DIR = os.path.join(PATH_ROOT, "50EP_SEED784_16B")

    NUM_EPOCHS = 50
    SEED = 784
    BATCH_SIZE = 16


    cache_dir = PATH_CLUSTER + 'newcache'
    os.makedirs(cache_dir, exist_ok=True)
    os.chmod(cache_dir, 0o777)  # Garantir permissões de escrita

    os.environ['TORCH_HOME'] = cache_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #------------------------------
    # Carrega Datasets
    #------------------------------
    print("\n Carregando Dados \n")
    # Carregar imagens e máscaras para treino, teste e validação
    training_images2, _ = load_images(os.path.join(PATH_DATASET, "Training/images"))
    training_masks2, _ = load_masks(os.path.join(PATH_DATASET, "Training/masks"))

    validation_images2, _ = load_images(os.path.join(PATH_DATASET, "Validation/images"))
    validation_masks2, _ = load_masks(os.path.join(PATH_DATASET, "Validation/masks"))

    test_images2, names = load_images(os.path.join(PATH_DATASET, "Test/images"))
    test_masks2, _ = load_masks(os.path.join(PATH_DATASET, "Test/masks"))

    # Verificar formatos
    print(f"📌 Treino - Imagens: {training_images2.shape}, Máscaras: {training_masks2.shape}")
    print(f"📌 Teste - Imagens: {test_images2.shape}, Máscaras: {test_masks2.shape}")
    print(f"📌 Validação - Imagens: {validation_images2.shape}, Máscaras: {validation_masks2.shape}")


    #----------------------------
    # Cria Datasets
    #----------------------------
    print("\n\n Criando Datasets \n")
    # Criando os datasets de treinamento e validação
    train_dataset = SatelliteDataset(training_images2, training_masks2)
    val_dataset = SatelliteDataset(validation_images2, validation_masks2)
    test_dataset = SatelliteDataset(test_images2, test_masks2, filenames=names)

    #----------------------------
    # Cria Seed e Dataloaders
    #----------------------------
    print("\n\n Criando Dataloaders \n")
    # Para um seed fixa
    g = torch.Generator()
    g.manual_seed(SEED)
    g.manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    #-----------------------------------
    # Treinar Redes
    #-----------------------------------

    print("\n\Proposed\n\n")
    path_save = os.path.join(PATH_DIR, "Proposed_AMAZON/")
    os.makedirs(path_save, exist_ok=True)

    # Cria instância do modelo
    model = Proposed(in_channels=4, out_channels=2).to(device)

    train_combine(model, train_loader, val_loader, NUM_EPOCHS, "proposed", path_save)


    #----------------------------------------
    # Fim dos treinamentos
    print("Fim dos treinamentos\n")

    print("\nInicío dos Testes\n\n")

    # -----------------------------------------
    print("\n\nProposed\n")
    path_model = os.path.join(PATH_DIR, "Proposed_AMAZON/proposed.pth")
    path_save_segmented = os.path.join(PATH_DIR, "Proposed_AMAZON/Results/")
    os.makedirs(path_save_segmented, exist_ok=True)

    # Cria instância do modelo
    model = Proposed(in_channels=4, out_channels=2).to(device)
    segment_and_time_combine(path_model, test_loader, path_save_segmented)
    apply_metrics_amazon(path_save_segmented, PATH_GT)



