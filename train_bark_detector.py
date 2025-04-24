import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor
from functools import partial

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from pathlib import Path
import os
import random
from tqdm import tqdm
from torch_audiomentations import Compose, AddColoredNoise, Gain, PitchShift # , TimeStretch
import matplotlib.pyplot as plt

from processing.dataset import AudioDataset
from processing.sampler import BalancedBatchSampler
from models.wav2vec import Wav2VecClassifier
from transformers import AutoModelForAudioClassification

# --- Configuration ---
DATA_DIR = Path("data/processed")
BARK_DIR = DATA_DIR / "bark"
NO_BARK_DIR = DATA_DIR / "no_bark"
CHECKPOINT_DIR = Path("checkpoints_bark_detector_wav2vec_v10")  # New dir for 2D model checkpoints
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Training params
BATCH_SIZE = 384
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50 # Adjust as needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42 # For deterministic split

# Augmentation probabilities
P_GAIN = 0.5         # Modify loudness
P_TIME_STRETCH = 0.5 # Stretch sound
P_OVERLAP = 0.0      # Overlap bark and no_bark (applied only on no_bark samples)
P_NOISE = 0.5        # Add noise
P_PITCH_SHIFT = 0.5  # Optional: Add pitch shift

# --- Set Seed for Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# --- Data Loading and Splitting ---
# (load_data_paths function remains the same as before)
def load_data_paths(bark_dir, no_bark_dir):
    bark_files = list(bark_dir.glob("*.wav"))
    no_bark_files = list(no_bark_dir.glob("*.wav"))

    if not bark_files or not no_bark_files:
         raise FileNotFoundError(f"Could not find WAV files in {bark_dir} or {no_bark_dir}. Please check paths.")

    files = bark_files + no_bark_files
    labels = [1] * len(bark_files) + [0] * len(no_bark_files)

    print(f"Found {len(bark_files)} bark samples and {len(no_bark_files)} no_bark samples.")

    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"Train set size: {len(train_files)}, Test set size: {len(test_files)}")
    return train_files, test_files, train_labels, test_labels, list(set(bark_files).intersection(set(train_files)))


# --- Training and Evaluation Functions ---
# (train_epoch and evaluate_epoch remain largely the same, just ensure output squeezing is correct)
def train_epoch(model, dataloader, criterion, optimizer, device, processor):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = processor(inputs).input_values.squeeze(0).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.logits.squeeze(1)  # Ensure shape [Batch] for BCEWithLogitsLoss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs).round().detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def evaluate_epoch(model, dataloader, criterion, device, processor):
    model.eval()
    total_loss = 0.0
    all_outputs = [] # Store raw outputs
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = processor(inputs).input_values.squeeze(0).to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.logits.squeeze(1) # Ensure shape [Batch]
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy()) # Store probabilities
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss, np.array(all_outputs), np.array(all_labels)


# --- Plotting Function ---
def plot_metrics(epochs, train_losses, test_losses, train_accs, test_accs, test_recalls, test_f1s, test_precisions, save_path=None):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # Loss Plot
    ax[0].plot(epochs, train_losses, label='Train Loss', marker='o')
    ax[0].plot(epochs, test_losses, label='Test Loss', marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Test Loss')
    ax[0].legend()
    ax[0].grid(True)

    # Accuracy Plot
    ax[1].plot(epochs, train_accs, label='Train Accuracy', marker='o')
    ax[1].plot(epochs, test_accs, label='Test Accuracy', marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Test Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    # Recall Plot
    ax[2].plot(epochs, test_recalls, label='Test Recall', marker='o', color='green')
    ax[2].plot(epochs, test_f1s, label='Test F1 Score', marker='o', color='orange')
    ax[2].plot(epochs, test_precisions, label='Test Precision', marker='o', color='red')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Recall')
    ax[2].set_title('Test Recall')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")
    plt.show()

def plot_precision_recall_curve(test_outputs, test_labels, save_path=None):
    """Plots the precision-recall curve and F1 score vs threshold."""
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    precisions, recalls, thresholds = precision_recall_curve(test_labels, test_outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10) # Calculate F1 scores

    optimal_threshold_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_index]
    optimal_f1_score = f1_scores[optimal_threshold_index]

    # Precision-Recall Curve
    ax1.plot(recalls, precisions, label='Precision-Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.axvline(recalls[optimal_threshold_index], color='gray', linestyle='--', label=f'Optimal Threshold Recall ({recalls[optimal_threshold_index]:.2f})')
    ax1.axhline(precisions[optimal_threshold_index], color='gray', linestyle='-.', label=f'Optimal Threshold Precision ({precisions[optimal_threshold_index]:.2f})')
    ax1.legend(loc="lower left")
    ax1.grid(True)


    ax2 = ax1.twinx() # Secondary y-axis for F1 score
    ax2.plot(thresholds, f1_scores[:-1], color='orange', label='F1 Score vs Threshold') # thresholds is one element shorter
    ax2.set_ylabel('F1 Score', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right')


    plt.title(f'Precision-Recall Curve and F1 Score vs Threshold (Optimal Threshold: {optimal_threshold:.3f}, Max F1: {optimal_f1_score:.3f})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-Recall curve plot saved to {save_path}")
    plt.show()

    return optimal_threshold


def evaluate_with_threshold(test_outputs, test_labels, threshold):
    """Evaluates the model with a given threshold and returns metrics."""
    preds = (test_outputs >= threshold).astype(int)
    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    return accuracy, precision, recall, f1

# --- Main Training Script ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Data Paths and Split
    train_files, test_files, train_labels, test_labels, all_bark_files = load_data_paths(BARK_DIR, NO_BARK_DIR)

    # 2. Create Datasets
    train_dataset = AudioDataset(
        file_paths=train_files,
        labels=train_labels,
        bark_file_paths=all_bark_files,
        p_overlap=0.0,
        is_train=True,
        preload=True,
    )
    test_dataset = AudioDataset(
        file_paths=test_files,
        labels=test_labels,
        transform=None, # No audio augmentation on test set
        p_overlap=0.0,
        is_train=False,
        preload=True,
    )
    processor_raw = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    processor = partial(processor_raw, sampling_rate=16000, return_tensors="pt")
    # processor = AudioProcessor(
    #     transform=augment,
    #     device=DEVICE
    # )
    sampler = BalancedBatchSampler(train_dataset, batch_size=BATCH_SIZE)

    # 3. Create DataLoaders
    # Consider reducing num_workers if you encounter memory issues or DataLoader hanging
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialize Model, Loss, Optimizer
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=1, problem_type="single_label_classification").to(DEVICE)  # Wav2VecClassifier(num_classes=1).to(DEVICE)
    model.wav2vec2.feature_extractor._freeze_parameters()  # Freeze feature extractor
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Training ---")
    best_test_f1 = 0.0

    # Lists to store metrics for plotting
    epoch_nums = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    test_recalls = []
    test_precisions = [] # Optional
    test_f1s = []       # Optional

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, processor)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluate
        test_loss, test_outputs, test_labels_epoch = evaluate_epoch(model, test_loader, criterion, DEVICE, processor)
        test_acc, test_prec, test_rec, test_f1 = evaluate_with_threshold(test_outputs, test_labels_epoch,
                                                                         0.5)  # Default threshold = 0.5 for epoch-wise metrics
        print(f"  Test Loss : {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"  Test Prec : {test_prec:.4f}, Test Rec: {test_rec:.4f}, Test F1: {test_f1:.4f}")

        # Store metrics
        epoch_nums.append(epoch + 1)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_recalls.append(test_rec)
        test_precisions.append(test_prec) # Optional
        test_f1s.append(test_f1)          # Optional


        # Save Checkpoint
        # checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch+1:03d}.pth"
        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': test_loss,
        #     'accuracy': test_acc,
        #     'precision': test_prec,
        #     'recall': test_rec,
        #     'f1': test_f1,
        # }, checkpoint_path)
        # print(f"  Checkpoint saved to {checkpoint_path}")

        # Optionally save the best model based on F1 score
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_model_path = CHECKPOINT_DIR / "best_model.pth"
            # Save only the model state dict for the best model for easier loading later
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved new best model to {best_model_path} (F1: {best_test_f1:.4f})")


    print("\n--- Training Finished ---")

    # --- Plotting ---
    plot_metrics(
        epoch_nums,
        train_losses,
        test_losses,
        train_accs,
        test_accs,
        test_recalls,
        test_f1s,
        test_precisions,
        save_path=CHECKPOINT_DIR / "training_metrics.png"  # Save the plot
    )

    print("\n--- Finding Optimal Threshold and Evaluating Best Model ---")
    best_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base", num_labels=1, problem_type="single_label_classification").to(DEVICE) #Wav2VecClassifier(num_classes=1).to(DEVICE)
    best_model.load_state_dict(torch.load(best_model_path))  # Load best model weights

    test_loss, test_outputs, test_labels_final = evaluate_epoch(best_model, test_loader, criterion, DEVICE,
                                                                processor)  # Evaluate best model

    optimal_threshold = plot_precision_recall_curve(test_outputs, test_labels_final,
                                                    save_path=CHECKPOINT_DIR / "precision_recall_curve.png")
    print(f"Optimal Threshold (Max F1): {optimal_threshold:.3f}")

    # Evaluate with optimal threshold
    final_accuracy, final_precision, final_recall, final_f1 = evaluate_with_threshold(test_outputs, test_labels_final,
                                                                                      optimal_threshold)
    print("\n--- Performance of Best Model on Test Set (Optimal Threshold) ---")
    print(f"  Accuracy : {final_accuracy:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall   : {final_recall:.4f}")
    print(f"  F1 Score : {final_f1:.4f}")
