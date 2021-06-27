from sentiment_model import Bert_sentiment, online_learning
from sklearn.datasets import load_files
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import numpy as np

reviews_train = load_files("data/aclImdb/train")
reviews_test = load_files("data/aclImdb/test")
def preprocess_text(text):
    t1 = text.decode("utf-8")
    t1 = t1.lower()
    t1 = t1.replace('<br /><br />','')
    t1 = t1.replace("\'","'")
    return t1

x_train, y_train = list(map(preprocess_text,reviews_train.data)), reviews_train.target
x_test , y_test = list(map(preprocess_text,reviews_test.data)), reviews_test.target



np.random.seed(5000)
indices = np.arange(25000)
np.random.shuffle(indices)
st_train =[]
for i in indices:
    st_train.append(x_train[i])
sy_train = y_train[indices]


def concat_list(list1,list2):
    return [list1,list2]

partial_x_train = st_train[:1000]
partial_y_train = sy_train[:1000]
x_test = st_train[10000:10200]
y_test = sy_train[10000:10200]

trainloader = torch.utils.data.DataLoader(list(map(concat_list,partial_x_train,partial_y_train)), batch_size=1,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(list(map(concat_list,x_test,y_test)), batch_size=200,
                                          shuffle=True, num_workers=2)
dataiter = iter(trainloader)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    print('모델 평가중')
    with torch.no_grad():
        for data in iterator:
            texts, labels = data
            labels = labels.type(torch.FloatTensor).to('cuda')

            predictions = model(texts).squeeze(1)

            loss = criterion(predictions, labels)

            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


model = Bert_sentiment()
model = model.to('cuda')
print(model.state_dict())

criterion = nn.BCELoss()
criterion = criterion.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_learning = len(partial_x_train)

for i in range(n_learning):
    if i%100 ==0:
        print(i)
    doc, label = next(dataiter)
    online_learning(model, doc, label, optimizer, criterion)

print("온라인 학습 완료")
torch.save(model.state_dict(), 'output/saveweight')

eval_loss, eval_acc = evaluate(model, testloader, criterion)


f = open("output/performance.txt", 'w')
print('평가완료')
result = "%d번 온라인 학습결과.\t accuracy : %f%% 입니다." %(n_learning,eval_acc*100)
f.write(result)
f.close()