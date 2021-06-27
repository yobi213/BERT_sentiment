from bert_embedding import bert_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F

#논문대로 gap
class Bert_sentiment(nn.Module):
    def __init__(self):
        super(Bert_sentiment, self).__init__()
        self.fc1 = nn.Linear(768,64)
        self.gap = nn.AvgPool2d((64,1))
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = bert_embedding(x)
        x = F.relu(self.fc1(x))
        x = x.view(1,-1,100,64)
        x = self.gap(x)
        x = x.view(-1,64)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


def online_learning(model, doc, label, optimizer, criterion):
    model.train()
    label = label.type(torch.FloatTensor).to('cuda')
    optimizer.zero_grad()
    predictions = model(doc).squeeze(1)
    loss = criterion(predictions, label)
    loss.backward()
    optimizer.step()
    print('온라인 학습 중')