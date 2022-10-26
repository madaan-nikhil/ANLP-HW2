import os
import torch
from tqdm import tqdm
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import f1_score

class Tracker:
    def __init__(self, metrics, filename, load=False):
        '''
        '''
        self.filename = os.path.join(filename, 'tracker.csv')

        if load:
            self.metrics_dict = self.load()
        else:        
            self.metrics_dict = {}
            for metric in metrics:
                self.metrics_dict[metric] = []


    def update(self, **args):
        '''
        '''
        for metric in args.keys():
            assert(metric in self.metrics_dict.keys())
            self.metrics_dict[metric].append(args[metric])

        self.save()


    def isLarger(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[-1] < value


    def isSmaller(self, metric, value):
        '''
        '''
        assert(metric in self.metrics_dict.keys())
        return sorted(self.metrics_dict[metric])[0] > value


    def save(self):
        '''
        '''
        df = pd.DataFrame.from_dict(self.metrics_dict)
        df = df.set_index('epoch')
        df.to_csv(self.filename)


    def load(self):
        '''
        '''
        df = pd.read_csv(self.filename)  
        metrics_dict = df.to_dict(orient='list')
        return metrics_dict


    def __len__(self):
        '''
        '''
        return len(self.metrics_dict)


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
        num_classes,
        resume,
        inference=False,
        test_loader=None):

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
        self.inference    = inference
        self.test_loader  = test_loader

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.tracker = Tracker(['epoch',
                                'train_loss',
                                'train_acc',
                                'dev_loss',
                                'dev_acc'], name, load=resume)


    def fit(self, pred_path=None):
        '''
            Fit model to training set over #epochs
        '''

        if self.inference:
            predictions = self.test_epoch(self.test_loader)
            return predictions

        is_best = False

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss, train_acc = self.trainEpoch()
            dev_loss,   dev_acc   = self.validateEpoch()

            self.epochVerbose(epoch, train_loss, train_acc, dev_loss, dev_acc)

            # Check if better than previous models
            if epoch > 1:
                is_best = self.tracker.isLarger('dev_acc', dev_acc)

            
            self.tracker.update(epoch=epoch,
                                train_loss=train_loss,
                                train_acc=train_acc,
                                dev_loss=dev_loss,
                                dev_acc=dev_acc)

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
        num_datapoints = 0

        # Iterate one batch at a time
        for i_batch, (im_batch, id_batch) in enumerate(self.train_loader):
            im_batch = {k: torch.as_tensor(v).to(device=self.device) for k, v in im_batch.items()}
            id_batch = id_batch.to(self.device)

            if torch.where(id_batch!=-100)[0].shape[0] == 0:
              continue

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
            mask              = (target != -100)
            num_correct      += int((target[mask] == preds[mask]).sum())

            avg_loss_epoch    = float(total_loss_epoch / (i_batch + 1))
            num_datapoints   += mask.sum().item()
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

        all_preds = []

        class_correct = defaultdict(int)
        class_points = defaultdict(int)
        y_true = []
        y_predicted = []

        # Do not store gradients 
        with torch.no_grad():
            # Get batches from DEV loader
            for i_batch, (im_batch, id_batch) in enumerate(self.dev_loader):
                # Class predictions
                im_batch = {k: torch.as_tensor(v).to(device=self.device) for k, v in im_batch.items()}
                id_batch = id_batch.to(self.device)

                logits = self.model(im_batch)

                loss_batch = self.criterion(logits.reshape(-1, self.num_classes), id_batch.reshape(-1))

                total_loss_dev += loss_batch
                preds = torch.argmax(logits.reshape(-1, self.num_classes), axis=1)
                all_preds.append(preds.detach())
                target = id_batch.reshape(-1)

                y_true.extend(target.squeeze().squeeze().cpu().numpy().tolist())
                y_predicted.extend(preds.detach().squeeze().cpu().numpy().tolist())

                mask = (target != -100)
                masked_target = target[mask]
                masked_preds = preds[mask]
                for c in range(self.num_classes):
                  class_targets_idx = (masked_target == c)
                  class_correct[c] += int((masked_target[class_targets_idx] == masked_preds[class_targets_idx]).sum())
                  class_points[c] += class_targets_idx.sum()
                num_correct += int((masked_target == masked_preds).sum())
                num_datapoints += mask.sum().item()
        all_preds = torch.cat(all_preds)

        print(torch.unique(all_preds, return_counts = True))

        acc_dev      = 100 * num_correct / num_datapoints
        avg_loss_dev = float(total_loss_dev / (i_batch + 1))

        for c in range(self.num_classes):
          print(f"Class {c}: Acc {100*class_correct[c]/class_points[c] :.6f}%")

        F1_score = f1_score(y_true, y_predicted, average=None)
        for c in range(self.num_classes):
            print(f"Class {c}: F1 Score {F1_score[c] :.6f}")

        return avg_loss_dev, acc_dev

    def test_epoch(self, data_loader):

        self.model.eval()
        
        predictions = []

        # Do not store gradients 
        with torch.no_grad():
            # Get batches from DEV loader
            for i_batch, (im_batch) in enumerate(data_loader):
                # Class predictions
                im_batch = {k: torch.as_tensor(v).to(device=self.device) for k, v in im_batch.items()}
                logits = self.model(im_batch)
                print(f'logits {logits.shape}')
                preds = torch.argmax(logits, axis=-1)
                print(f'preds {preds.shape}')
                predictions.append({'input_ids': im_batch['input_ids'].squeeze(), 'preds':preds.squeeze()})
        
        return predictions
                

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

    def save_predictions(self, predictions, pred_path):
        with open(pred_path, 'wb') as f:
            pickle.dump(predictions, f)

    def epochVerbose(self, epoch, train_loss, train_acc, dev_loss, dev_acc):
        log = "\nEpoch: {}/{} summary:".format(epoch, self.epochs)
        log += "\n             Train loss  |  {:.6f}".format(train_loss)
        log += "\n               Val loss  |  {:.6f}".format(dev_loss)
        log += "\n     Train accuracy (%)  |  {:.6f}".format(train_acc)
        log += "\n       Val accuracy (%)  |  {:.6f}".format(dev_acc)
        print(log)