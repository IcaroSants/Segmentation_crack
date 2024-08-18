import torch
from tqdm import tqdm
import time

def fit(model, train, val, optimizer, criterion, epochs, patience: int=3):
    best = 1000
    count_patince = 0
    for epoch in range(epochs):
        total_loss_training = 0
        total_loss_val = 0
        count_train = len(train)
        count_val = len(val)
        
        
        
        model.train()
        pbar =  tqdm(train, total=len(train),ncols=100)
        for batch, label in pbar:
            
            optimizer.zero_grad()

            output = model(batch)['out']
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss_training+=loss.item()
            
            pbar.set_description('loss_training:{:.2f}'.format(loss.item()), refresh=True)
            pbar.update(1)
           
        
        model.eval()
        pbar = tqdm(val, total=len(val), ncols=100)
        with torch.no_grad():
            for batch,label in pbar:
            
                output = model(batch)['out']
                loss = criterion(output, label)
                count_val+=1
                total_loss_val += loss.item()    
                pbar.set_description(' loss_val:{:.2f}'.format(loss.item()), refresh=True)
        
        mean_loss_val = total_loss_val/count_val
        mean_loss_training = total_loss_training/count_train
        print("loss_trainig: {:.2f}".format(mean_loss_training))
        print("loss_val: {:.2f}".format(mean_loss_val))

        if mean_loss_val < best:
            torch.save(model.state_dict(),'best.pt')
        else:
            count_patince+=1
        
        if count_patince == patience:
            break
        

        