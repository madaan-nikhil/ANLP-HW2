import os
import torch
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        epochs,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        dev_loader,
        device,
        name,
        num_classes):

        self.model        = model
        self.epochs       = epochs
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.criterion    = criterion
        self.train_loader = train_loader
        self.dev_loader   = dev_loader
        self.device       = device
        self.name         = name
        self.start_epoch  = 1
        self.num_classes  = num_classes

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()


    def fit(self):
        '''
            Fit model to training set over #epochs
        '''
        is_best = False

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss, train_acc = self.trainEpoch()
            dev_loss,   dev_acc   = self.validateEpoch()

            self.epochVerbose(epoch, train_loss, train_acc, dev_loss, dev_acc)

            # Check if better than previous models
            if epoch > 1:
                is_best = self.tracker.isLarger('dev_acc', dev_acc)


            self.save_checkpoint(epoch, is_best)


    def trainEpoch(self):
        '''
            Train model for ONE epoch
        '''
        # Set model to training mode
        self.model.train()

        # Progress bar over the current epoch
        batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, desc='Train') 

        # Number of correct classifications
        num_correct = 0
        # Cumulative loss over all batches or Avg Loss * num_batches
        total_loss_epoch = 0

        # Iterate one batch at a time
        for i_batch, (im_batch, id_batch) in enumerate(self.train_loader):
            im_batch = im_batch.to(self.device)
            id_batch = id_batch.to(self.device)

            # Get predictions (forward pass)
            self.optimizer.zero_grad()
        
            # Mixed precision
            with torch.cuda.amp.autocast():  
                logits = self.model(im_batch)
                loss_batch = self.criterion(logits.reshape(-1, self.num_classes), id_batch.reshape(-1))      
            
            self.scaler.scale(loss_batch).backward()
            self.scaler.step(self.optimizer) 
            self.scaler.update()

            # Performance metrics
            total_loss_epoch += loss_batch
            preds             = torch.argmax(logits.reshape(-1, self.num_classes), axis=1)
            target            = id_batch.reshape(-1)
            num_correct      += int((target == preds).sum())
            avg_loss_epoch    = float(total_loss_epoch / (i_batch + 1))
            num_datapoints   += target.shape[0]
            acc_epoch         = 100 * num_correct / num_datapoints
            
            self.scheduler.step(avg_loss_epoch)
            # Performance tracking verbose
            batch_bar.set_postfix(
                acc="{:.3f}%".format(acc_epoch),
                avg_train_loss="{:.04f}".format(avg_loss_epoch),
                num_correct=num_correct,
                lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr'])))
            batch_bar.update()

        batch_bar.close()

        return avg_loss_epoch, acc_epoch


    def validateEpoch(self):
        '''
            Val pass for ONE epoch
        '''

        # Set model to evaluation mode
        self.model.eval()

        total_loss_dev = 0
        num_correct    = 0
        num_datapoints = 0

        # Do not store gradients 
        with torch.no_grad():
            # Get batches from DEV loader
            for i_batch, (im_batch, id_batch) in enumerate(self.dev_loader):
                # Class predictions
                im_batch = im_batch.to(self.device)
                id_batch = id_batch.to(self.device)

                logits = self.model(im_batch)

                loss_batch = self.criterion(logits.reshape(-1, self.num_classes), id_batch.reshape(-1))

                total_loss_dev += loss_batch
                preds = torch.argmax(logits.reshape(-1, self.num_classes), axis=1)
                target = id_batch.reshape(-1)
                num_correct += int((target == preds).sum())
                num_datapoints += target.shape[0]

        acc_dev      = 100 * num_correct / num_datapoints
        avg_loss_dev = float(total_loss_dev / (i_batch + 1))

        return avg_loss_dev, acc_dev


    def save_checkpoint(self, epoch, is_best):
        '''
            Save model dict and hyperparams
        '''
        state = {"epoch": epoch,
                 "model": self.model,
                 "optimizer": self.optimizer,
                 "scheduler": self.scheduler }

        # Save checkpoint to resume training later
        checkpoint_path = os.path.join(self.name, "checkpoint.pth")
        torch.save(state, checkpoint_path)
        print('Checkpoint saved: {}'.format(checkpoint_path))

        # Save best model weights
        if is_best:
            best_path = os.path.join(self.name, "best_weights.pth")
            torch.save(self.model.state_dict(), best_path)
            print("Saving best model: {}".format(best_path))

    def epochVerbose(self, epoch, train_loss, train_acc, dev_loss, dev_acc):
        log = "\nEpoch: {}/{} summary:".format(epoch, self.epochs)
        log += "\n             Train loss  |  {:.6f}".format(train_loss)
        log += "\n               Val loss  |  {:.6f}".format(dev_loss)
        log += "\n     Train accuracy (%)  |  {:.6f}".format(train_acc)
        log += "\n       Val accuracy (%)  |  {:.6f}".format(dev_acc)
        print(log)