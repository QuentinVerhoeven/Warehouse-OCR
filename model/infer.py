import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn


#My lowest CER came from epoch 14, so I downloaded it and imported it into vscode
MODEL_PATH_NAME = "model/crnn_epoch14.pth"




class CRNNBackbone(nn.Module):
    def __init__(self, img_channel=3):
        super(CRNNBackbone, self).__init__()
        
        # VGG-style Convolutional Feature Extractor (CNN)
        self.cnn = nn.Sequential( # Sets the flow of data to be sequential, exactly as written
            # Block 1: Applies 64 kernels over your image to extract features
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1), #padding ensures image size doesn't shrink during this step
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Pooling layer: Downsizes image height by keeping maximum value in the 2x2 window
            
            # Block 2: expand to 128 feature channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Image height: 16 -> 8
            
            # Block 3: Run two convolutional layers back to back
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)), # Height: 8 -> 4 (preserves width)
            # We need to preserve width so the characters don't blend together (and to keep enough time steps to predict a long word)
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)), # Height: 4 -> 2 (preserves width)
            
            # Block 5 (Final compression)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), 
            nn.BatchNorm2d(512), 
            nn.ReLU(True) # Height: Shrink to one pixel tall
        )

    def forward(self, x):
        # Input shape: [Batch_Size, Channels, Height (32), Width (variable)]
        features = self.cnn(x)
        # when you pass an image batch x into forward(), it runs through the block above. 
        
        # Output shape: [Batch_Size, 512, Height (1), Width']
        return features


class CRNNWithRNN(nn.Module):
    def __init__(self, img_channel=3, num_classes=65, hidden_size=256): #num_classes is size of alphabet (64 characters, one blank token for CTC)
        #hidden size is the internal memory size of the LSTM cells (how much context can it remember)
        super(CRNNWithRNN, self).__init__()
        
        # The VGG style backbone from above
        self.backbone = CRNNBackbone(img_channel)
        
        # Bidirectional LSTM layers to model character sequence context (normal RNN forget old information quickly)
        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=False)
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=False)
        
        # FInal linear layer that takes the 512 dimensional output (forward and backward passes of 256 features) and squashes it into 65 probability scores
        self.embedding = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Pass through CNN backbone: [Batch, 512, 1, Width']
        features = self.backbone(x)
        
        # Reshape for RNN: squeeze height dimension and permute to [Width', Batch, Channels]
        features = features.squeeze(2)          # [Batch, 512, Width']
        features = features.permute(2, 0, 1)    # [Width', Batch, 512]   We do this to satisfy batch_first=False
        
        # Pass through Recurrent Layers, discard the cell memory states
        recurrent_out, _ = self.rnn1(features)
        recurrent_out, _ = self.rnn2(recurrent_out) # [Width', Batch, hidden_size * 2]
        
        # Map to character classes
        output = self.embedding(recurrent_out)    # [Width', Batch, num_classes]
        return output.permute(1,0,2)







characters = "½¾¼0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-'\" "
char_to_idx = {char: i + 1 for i, char in enumerate(characters)} # 0 is reserved for CTC blank
idx_to_char = {v: k for k, v in char_to_idx.items()}



device = torch.device('cpu') 
model = CRNNWithRNN(num_classes=len(characters) + 1).to(device)

# Load the weights (Ignore the 'module.' prefix if you saved with DataParallel)
state_dict = torch.load(MODEL_PATH_NAME, map_location=device)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# The Transform from kaggle
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# The Prediction Function
def predict_text(pil_image):
    pil_image = pil_image.convert('RGB')


    # Resize keeping aspect ratio, forcing height to 32
    w, h = pil_image.size
    new_w = int(32 * (w / h))
    pil_image = pil_image.resize((new_w, 32), Image.Resampling.BILINEAR)
    
    # Convert to tensor and add batch dimension: [1, 1, 32, W]
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(img_tensor)
        # [Width, Batch, Classes] -> [Batch, Width]
        preds_idx = preds.argmax(dim=2).permute(1, 0).cpu().numpy()
    
    # Decode CTC (dropping consecutive duplicates and blank 0s)
    raw_pred = preds_idx[0, :]
    pred_text = []
    prev = -1
    for idx in raw_pred:
        if idx != prev and idx != 0:
            pred_text.append(idx_to_char.get(idx, ''))
        prev = idx
        
    return "".join(pred_text)