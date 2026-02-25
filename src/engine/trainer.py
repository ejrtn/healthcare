import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast

class CTTrainer:
    """
    Training engine for CT Trauma Detection model.
    Supports gradient accumulation, autocast, and multi-GPU (via DataParallel).
    """
    def __init__(self, model, optimizer, scheduler, criterion_dict, device, accumulation_steps=8):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion_dict = criterion_dict
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    def train_epoch(self, loader, epoch, class_names):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()

        loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [Train]", leave=False)
        for i, batch in loop:
            inputs = batch["image"].to(self.device)
            
            with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                outputs = self.model(inputs)
                
                loss = 0
                for k in class_names:
                    target = batch[k].to(self.device)
                    if hasattr(target, "as_tensor"):
                        target = target.as_tensor()
                    
                    if target.dim() > 1:
                        target = torch.argmax(target, dim=1)
                    target = target.long()

                    pred_k = outputs[k].as_tensor() if hasattr(outputs[k], "as_tensor") else outputs[k]
                    
                    # Weighting the gate head more
                    weight = 2.0 if k == 'any_injury' else 1.0
                    loss += self.criterion_dict[k](pred_k, target) * weight
                
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += (loss.item() * self.accumulation_steps)
            loop.set_postfix(loss=(loss.item() * self.accumulation_steps))

        return total_loss / len(loader)

    def validate(self, loader, epoch, class_names, metrics):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            loop = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for batch in loop:
                inputs = batch["image"].to(self.device)
                outputs = self.model(inputs)
                
                loss = 0
                for k in class_names:
                    target = batch[k].to(self.device)
                    if hasattr(target, "as_tensor"):
                        target = target.as_tensor()
                    
                    target_idx = torch.argmax(target, dim=1) if target.dim() > 1 else target.long()
                    
                    # Update metrics
                    metrics[k].update(torch.softmax(outputs[k], dim=1), target_idx)
                    
                    out_k = outputs[k].as_tensor() if hasattr(outputs[k], "as_tensor") else outputs[k]
                    loss += self.criterion_dict[k](out_k, target_idx)
                
                val_loss += loss.item()
                loop.set_postfix(val_loss=loss.item())

        avg_loss = val_loss / len(loader)
        auc_results = {k: metrics[k].compute().item() for k in class_names}
        for k in class_names: metrics[k].reset()
        
        return avg_loss, auc_results
