import torch

def train_model(model, criterion, loader, optimizer, scheduler, weights, device):
    model.train()
    for step, (u_batch, Ri_batch, y_batch) in enumerate(loader):  # for each training step
        u_batch=u_batch.to(device)
        Ri_batch = Ri_batch.to(device)
        y_batch=y_batch.to(device)
                
        prediction = model(u_batch, Ri_batch)

        loss = criterion(prediction*weights, y_batch*weights).cpu()
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients to update weights
        
    scheduler.step()

def test_model(model, criterion, trainloader, weights, device, text = 'validation'):
    model.eval() # Evaluation mode (important when having dropout layers)
    test_loss = 0
    with torch.no_grad():
        for step, (u_batch, Ri_batch, y_batch) in enumerate(trainloader):  # for each training step
            u_batch=u_batch.to(device)
            Ri_batch = Ri_batch.to(device)
            y_batch=y_batch.to(device)

            prediction = model(u_batch, Ri_batch)

            loss = criterion(prediction*weights, y_batch*weights).cpu()
            test_loss = test_loss + loss.data.numpy() # Keep track of the loss 
        test_loss /= len(trainloader) # dividing by the number of batches
        print(text + ' loss:',test_loss)
    return test_loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
