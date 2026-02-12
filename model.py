import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
import os

#ImageEncoder
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()

        try:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #Call pretrained resnet18
        except Exception:
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

        # FIX: pass as keyword to avoid accidental position mismatch
        attn_output, attn_weights = self.mha(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )
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
        self.fc_layer = nn.Linear(512, 512)  # For the combined vector
        self.final_fc = nn.Linear(512, 10)   # Lớp cuối cho phân loại

        # Khởi tạo Cross Multi-Head Attention và Layer Normalization một lần trong __init__
        self.cross_attn_text = Cross_MHA(embed_dim=512, num_heads=8)   # FIX: tách 2 cross-attn (đỡ dùng chung trọng số)
        self.cross_attn_img  = Cross_MHA(embed_dim=512, num_heads=8)
        self.self_attn = Cross_MHA(embed_dim=512, num_heads=8)  # Dùng cho self attention của vector kết hợp
        self.add_norm1 = AddNorm(embed_dim=512)
        self.add_norm2 = AddNorm(embed_dim=512)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # FIX: Không được tạo Linear trong forward. Tạo sẵn ở đây.
        self.combine_fc = nn.Linear(1024, 512)
        
    def forward(self, image, input_ids, attention_mask):
        # Encode ảnh và văn bản
        img = self.image_encoder(image)  # [batch, 512]
        text = self.text_encoder(input_ids, attention_mask)  # [batch, 768]
        text = self.text_fc(text)  # [batch, 512]
        
        # FIX: MultiheadAttention cần input shape [batch, seq_len, embed_dim]
        # Ta coi mỗi modality là 1 token => seq_len = 1
        img = img.unsqueeze(1)    # [batch, 1, 512]
        text = text.unsqueeze(1)  # [batch, 1, 512]
        
        # Sử dụng module cross attention đã khởi tạo sẵn
        # Text Cross: text (query) attend image (key, value)
        text_cross, _ = self.cross_attn_text(query=text, key=img, value=img)
        
        # Image Cross: image (query) attend text (key, value)
        img_cross, _ = self.cross_attn_img(query=img, key=text, value=text)
        
        # Kết hợp hai vector
        combined = torch.cat((text_cross, img_cross), dim=-1)  # [batch, 1, 1024]
        # Nếu cần giảm kích thước, có thể thêm một layer linear để map về 512
        # Ví dụ:
        combined = self.combine_fc(combined)  # FIX: dùng layer đã khai báo trong __init__
        
        # Self Multi-Head Attention cho vector kết hợp
        self_attn_output, _ = self.self_attn(query=combined, key=combined, value=combined)
        
        # Add & Norm
        combined = self.add_norm1(combined, self_attn_output)
        
        # Feed Forward Layer
        ff_output = self.fc_layer(combined)
        ff_output = self.relu(ff_output)
        combined = self.add_norm2(combined, ff_output)
        
        # Lớp phân loại cuối cùng
        output = self.final_fc(combined)  # [batch, 1, 10]

        output = output.squeeze(1)        # [batch, 10]
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

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        
    
#train function
def train(model, dataloader, epochs=15, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss() #Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=lr) #Adam Optimization Algorithm
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_id = batch["label_id"].to(device)
            optimizer.zero_grad()
            output = model(image, input_ids, attention_mask)
            
            loss = criterion(output, label_id)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss:{total_loss / len(dataloader)}")
    
    
#training phase

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

dataset = MyDataset(
    csv_file="training_data.csv",
    image_dir="Multimodal-Classifying",
    tokenizer=tokenizer
)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModule().to(device)

train(model, train_loader)
