import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from torchvision import transforms
import os

#ImageEncoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True) #Call pretrained resnet18
        self.resnet.fc = nn.Identity() #delete the last layer
    def forward(self, x):
        return self.resnet(x)
    
#TextEncoder
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
    def forward(self, input_ids, attention_mask):
        
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs.last_hidden_state[:, 0, :]
    
#Cross Multi-Head Attention
class Cross_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Cross_MHA, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask)
        return attn_output, attn_weights
    
#Add and Norm Layer
class AddNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=eps)
    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

#FusionModule
class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.text_fc = nn.Linear(768, 512)  # Map text vector to 512 dim (để khớp với image vector)
        self.fc_layer = nn.Linear(1024, 512)  # For the combined vector
        self.final_fc = nn.Linear(512, 10)   # Lớp cuối cho phân loại

        self.add_norm1 = AddNorm(embed_dim=512)
        self.add_norm2 = AddNorm(embed_dim=512)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, image, input_ids, attention_mask):
        # Encode ảnh và văn bản
        img = self.image_encoder(image)  # [batch, 512]
        text = self.text_encoder(input_ids, attention_mask)  # [batch, 768]
        text = self.text_fc(text)  # [batch, 512]
        
        combined = torch.cat((img, text), dim=1)  # [batch, 1024]
        
        # Feed Forward Layer
        ff_output = self.fc_layer(combined)
        ff_output = self.relu(ff_output)
        # Lớp phân loại cuối cùng
        output = self.final_fc(ff_output)
        return output

#Dataset
class MyDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer, max_length=512, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file, delimiter=";", encoding="latin1")
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
            return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        # Load image
        image_path = os.path.join(self.image_dir, str(item["img_path"]))
        if not os.path.exists(image_path):
            print(f"⚠️ Warning: File not found {image_path}, using default image...")
            image_path = "imgamazon/31bNhi6E3eL._AC_.jpg"

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Load text
        text = str(item["description"])
        encoded_text = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)

        # Load label
        label = torch.tensor(int(item["label_id"]) - 1, dtype=torch.long)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": label
        }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = MyDataset(
    csv_file="training_data.csv",
    image_dir="D:\pythonhaha",
    tokenizer=tokenizer
)

train_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("multimodal.pth")
model.eval()

correct = 0
total = 0
accuracy = 0

with torch.no_grad():
    for batch in test_loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_id = batch["label_id"].to(device)
        outputs = model(image, input_ids, attention_mask)
        outputs = nn.Softmax(dim=-1)(outputs)
        predicted_label = torch.argmax(outputs, dim=1)
        correct += (predicted_label == label_id).sum().item()
        total += label_id.size(0)
        accuracy = correct / total
        print(f"Test accuracy: {accuracy:.4f}")
        
    