import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.model2 import process_video, process_audio, VideoFeatureExtractor, AudioFeatureExtractor, DeepfakeClassifier

print("train.py started")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_name, label in [("real", 0), ("fake", 1)]:
            folder = os.path.join(root_dir, label_name)
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith(".mp4"):
                        self.samples.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video_tensor = process_video(video_path)
        audio_tensor = process_audio(video_path)
        return video_tensor.squeeze(0), audio_tensor.squeeze(0), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    videos, audios, labels = zip(*batch)
    videos = torch.stack(videos)
    audios = nn.utils.rnn.pad_sequence(audios, batch_first=True)
    labels = torch.stack(labels)
    return videos, audios, labels


print("Loading datasets...")
train_dataset = DeepfakeDataset("dataset/train")
val_dataset = DeepfakeDataset("dataset/val")

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

if len(train_dataset) == 0:
    print("No training videos found inside dataset/train/real and dataset/train/fake")
    exit()

if len(val_dataset) == 0:
    print("No validation videos found inside dataset/val/real and dataset/val/fake")
    exit()

print("Creating dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

print("Loading models...")
video_model = VideoFeatureExtractor().to(DEVICE)
audio_model = AudioFeatureExtractor().to(DEVICE)
classifier = DeepfakeClassifier().to(DEVICE)

print("Freezing feature extractors...")
for p in video_model.parameters():
    p.requires_grad = False
for p in audio_model.parameters():
    p.requires_grad = False

video_model.eval()
audio_model.eval()
classifier.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

EPOCHS = 15
print("Starting training loop...")

for epoch in range(EPOCHS):
    total_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for videos, audios, labels in train_loader:
        print("Processing batch...")

        videos = videos.to(DEVICE)
        audios = audios.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            video_features = video_model(videos)
            audio_features = audio_model(audios)

        outputs = classifier(video_features, audio_features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Train Acc: {acc:.2f}%")

print("Saving trained weights...")
torch.save(video_model.state_dict(), "video_model.pth")
torch.save(audio_model.state_dict(), "audio_model.pth")
torch.save(classifier.state_dict(), "classifier.pth")

print("Training complete. Weights saved.")
print("train.py is running")