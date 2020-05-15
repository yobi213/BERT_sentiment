from sentiment_model import Bert_sentiment, online_learning
import torch.optim as optim
import torch
import torch.nn as nn



model = Bert_sentiment()
model = model.to('cuda')
#print(model.state_dict())

criterion = nn.BCELoss()
criterion = criterion.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)


doc = ('yep, the topic is a straight quote from the movie and i think it\'s pretty accurate. i was so bored to dead with this pointless effort. all the flashes etc. making no sense after first 20 minutes is just bad film making + if you are epileptic, you would have died at least five times already. of course all the david lynch fans would raise a flag for this kind of turkey to be "the best film ever made" because it doesn\'t make any sense and when it doesn\'t make any sense it\'s got to be art, and art movie is always good. right? i say wrong. this kind of artificial art grab is just a pathetic way to try to show that you\'re a good film maker. anthony hopkins as a excellent actor should just stay acting.',)
label = torch.tensor([0])

online_learning(model,doc,label,optimizer,criterion)


torch.save(model.state_dict(),'save/save_weight')