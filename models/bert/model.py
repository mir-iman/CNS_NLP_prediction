# Inspired from https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
# and from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

# UPDATED: use PyTorch AdamW instead of transformers.AdamW
# This is the recommended optimizer now
from torch.optim import AdamW

# UPDATED: modern TorchMetrics classification API
# Original code used Accuracy(multiclass=True)
from torchmetrics.classification import BinaryAccuracy
from transformers import BertModel, get_linear_schedule_with_warmup
from evaluators.calculate_metrics import calculate_metrics


class BERT(pl.LightningModule):
    # Set up the classifier
    def __init__(self, config, loss_fn, steps_per_epoch):
        super().__init__()
        self.save_hyperparameters()
        pretrained_model_path = os.path.join(config.pretrained_dir, config.pretrained_file)
        self.bert = BertModel.from_pretrained(pretrained_model_path, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, out_features=1)  # Change if multi-label
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = config.epochs
        self.lr = config.lr
        self.loss_fn = loss_fn
        self.weight_decay = config.weight_decay

        # Initialize the metrics

        # Balanced Accuracy
        #self.tr_bal = Accuracy(average='macro',
        #                       num_classes=2,
        #                       multiclass=True)
        #self.val_bal = Accuracy(average='macro',
        #                        num_classes=2,
        #                        multiclass=True)
        #self.test_bal = Accuracy(average='macro',
        #                        num_classes=2,
        #                        multiclass=True)

        #update to torchmetrics binaryaccuracy 
        self.train_bal = BinaryAccuracy()
        self.val_bal = BinaryAccuracy()
        self.test_bal = BinaryAccuracy()

        #automatics epoch outputs have been removed so manually store predictions 
        self._train_preds = []
        self._train_targets = []

        self._val_preds = []
        self._val_targets = []

        self._test_preds = []
        self._test_targets = []

    def forward(self, input_ids, attn_mask, labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)
        sigmoided_output = torch.sigmoid(output)  # Loss function usually will include a sigmoid layer

        loss = 0
        if labels is not None:
            #updated- makes sure labels shape matchets logits 
            labels = labels.float().view(-1, 1)
            
            loss = self.loss_fn(output, labels)

        return loss, sigmoided_output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_labels = batch['label']

        loss, pred_labels = self(input_ids, attention_mask, target_labels)
        self.log('train_loss', loss, prog_bar=False, logger=False)
        target_labels_int = target_labels.int().view(-1)

        #update - torchmetrics 
        self.tr_bal.update(pred_labels.view(-1), target_labels_int)

        #update - store predictions to compute epoch metrics 
        self._train_preds.append(probs.detach().view(-1).cpu())
        self._train_targets.append(labels_float.detach().view(-1).cpu())


        return {"loss": loss, "predictions": pred_labels.detach(), "labels": target_labels}

   # update to on_train_epoch_end 
   #  def training_epoch_end(self, output):
        # Take all the target and predicted labels from the epoch and flatten into 1-dim tensors,
        # then make target into int
        pred_labels, target_labels = self.flatten_epoch_output(output)
        target_labels_int = target_labels.int()

        # Calculate and then log the metrics being used for this proejct
        train_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)
        self.log('train_perf',
                 train_epoch_metrics,
                 prog_bar=False,
                 logger=True,
                 on_epoch=True)

        # Keep balanced accuracy a full on torchmetrics module, as part of this class, for progress bar
        bal = self.tr_bal(pred_labels, target_labels_int)
        self.log('tr_bal', bal, prog_bar=False, logger=False)

        # print(f'\nTraining epoch completed, used {len(pred_labels)} examples')

    def on_train_epoch_end(self):

        pred_labels = torch.cat(self._train_preds)
        target_labels = torch.cat(self._train_targets)

        target_labels_int = target_labels.int()

        train_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)

        self.log("train_perf",
                 train_epoch_metrics,
                 prog_bar=False,
                 logger=True)

        bal = self.tr_bal.compute()

        self.log("tr_bal", bal, prog_bar=False, logger=False)

        self.tr_bal.reset()

        self._train_preds.clear()
        self._train_targets.clear()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_labels = batch['label']

        loss, pred_labels = self(input_ids, attention_mask, target_labels)
        self.log("val_loss", loss, prog_bar=False, logger=True, on_epoch=True)
        #update 
        target_labels_int = target_labels.int().view(-1)

        self.val_bal.update(pred_labels.view(-1), target_labels_int)
        self._val_preds.append(pred_labels.detach().view(-1).cpu())
        self._val_targets.append(target_labels.detach().view(-1).cpu())

        return {"loss": loss, "predictions": pred_labels.detach(), "labels": target_labels}
    #update- replace to on_validation_epoch_end 
    #def validation_epoch_end(self, output):
        # Take all the target and predicted labels from the epoch and flatten into 1-dim tensors,
        # then make target into int
        try:
            pred_labels, target_labels = self.flatten_epoch_output(output)
        except:
            print(f'This is output len: {len(output)}')
            print(f'Error, this is output: {output}')
        target_labels_int = target_labels.int()

        # Calculate and then log the metrics being used for this proejct
        dev_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)
        self.log('dev_perf',
                 dev_epoch_metrics,
                 prog_bar=False,
                 logger=True,
                 on_epoch=True)

        # Keep balanced accuracy a full on torchmetrics module, as part of this class, for early stopping
        bal = self.val_bal(pred_labels, target_labels_int)
        self.log('val_bal', bal, prog_bar=True, logger=False)

        # print(f'\nValidation epoch completed, used {len(pred_labels)} examples')

    def on_validation_epoch_end(self):

        pred_labels = torch.cat(self._val_preds)
        target_labels = torch.cat(self._val_targets)

        target_labels_int = target_labels.int()

        dev_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)

        self.log("dev_perf",
                 dev_epoch_metrics,
                 prog_bar=False,
                 logger=True)

        bal = self.val_bal.compute()

        self.log("val_bal", bal, prog_bar=True, logger=False)

        self.val_bal.reset()

        self._val_preds.clear()
        self._val_targets.clear()


    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_labels = batch['label']

        loss, pred_labels = self(input_ids, attention_mask, target_labels)
        self.log("test_loss", loss, prog_bar=False, logger=True, on_epoch=True)

        #update 
        target_labels_int = target_labels.int().view(-1)

        self.test_bal.update(pred_labels.view(-1), target_labels_int)

        self._test_preds.append(pred_labels.detach().view(-1).cpu())
        self._test_targets.append(target_labels.detach().view(-1).cpu())

        return {"loss": loss, "predictions": pred_labels.detach(), "labels": target_labels}

    #update replace test_epoch_end
    #def test_epoch_end(self, output):
        # Take all the target and predicted labels from the epoch and flatten into 1-dim tensors,
        # then make target into int
        print("test_epoch_end")
        try:
            pred_labels, target_labels = self.flatten_epoch_output(output)
        except:
            print(f'This is output len: {len(output)}')
            print(f'Error, this is output: {output}')
        target_labels_int = target_labels.int()

        # Calculate and then log the metrics being used for this project
        test_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)
        self.log('test_perf',
                 test_epoch_metrics,
                 prog_bar=False,
                 logger=True,
                 on_epoch=True)

        # Keep balanced accuracy a full on TorchMetrics module, as part of this class, for early stopping
        bal = self.test_bal(pred_labels, target_labels_int)
        self.log('test_bal', bal, prog_bar=True, logger=False)

        print(f'\nTest epoch completed, used {len(pred_labels)} examples')

    def on_test_epoch_end(self):

        pred_labels = torch.cat(self._test_preds)
        target_labels = torch.cat(self._test_targets)

        target_labels_int = target_labels.int()

        test_epoch_metrics = calculate_metrics(pred_labels, target_labels_int)

        self.log("test_perf",
                 test_epoch_metrics,
                 prog_bar=False,
                 logger=True)

        bal = self.test_bal.compute()

        self.log("test_bal", bal, prog_bar=True, logger=False)

        self.test_bal.reset()

        print(f"\nTest epoch completed, used {len(pred_labels)} examples")

        self._test_preds.clear()
        self._test_targets.clear()


    def configure_optimizers(self):
        print(f'Here is our weight decay: {self.weight_decay}')
        #update - uses torch.optim.AdamW
        
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        # Overwrites progress bar to remove v_num as I'm not using this and it annoys me
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    @staticmethod
    def flatten_epoch_output(output):
        """
        Helper function that takes all of the batches in an epoch_end and flattens it
        into one tensor each for the predictions and labels

        :param output: output handed via a epoch_end hook
        :param device: cuda device for the tensors
        :return: a tensor with all the predicted in this epoch, and one for all the target labels
        """
        try:
            epoch_pred_labels = torch.cat([x['predictions'] for x in output]).squeeze()
        except RuntimeError:
            print(f'Could not cat, here is the first prediction output: {output[0]["predictions"]}')

        try:
            epoch_target_labels = torch.cat([x['labels'] for x in output]).squeeze()
        except RuntimeError:
            print(f'Could not cat, here is the first prediction output: {output[0]["labels"]}')

        return epoch_pred_labels, epoch_target_labels
