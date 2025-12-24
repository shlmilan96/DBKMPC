import torch
import torch.nn as nn
import numpy as np
import time
def bilinear_koopman_loss(model, x_seq, u_seq, cfg):
    batch_size, seq_len, _ = x_seq.shape

    alpha = cfg['training']['loss_weights']['alpha']
    beta = cfg['training']['loss_weights']['beta']
    gamma = cfg['training']['loss_weights']['gamma']

    # 1. Prediction Loss (L1) - 논문 수식 (19) 반영
    x_flat = x_seq.reshape(-1, x_seq.shape[-1])
    z_seq_true = model.get_full_state(x_flat).reshape(batch_size, seq_len, -1)

    loss_pred = 0.0
    z_k = z_seq_true[:, 0, :]

    for k in range(seq_len - 1):
        z_next_true = z_seq_true[:, k + 1, :]
        u_k = u_seq[:, k, :]
        z_k_pred = model.forward_step(z_k, u_k)

        step_loss = torch.mean((z_k_pred - z_next_true) ** 2)
        # 논문의 gamma^(k-1) 반영 (첫 예측 step k=1일 때 가중치 1)
        loss_pred += (gamma ** k) * step_loss
        z_k = z_k_pred

    # 2. Independence Loss (L2) - 논문 수식 (22) 반영
    # 행렬 A가 Full Rank(N)를 갖도록 특이값을 1로 유도 (미분 가능한 rank 제약)
    S = torch.linalg.svdvals(model.A)
    loss_indep = torch.mean((S - 1.0) ** 2)

    # 3. Sparsity Loss (L3) - 논문 수식 (23) 반영 [cite: 457]
    loss_spar = torch.norm(model.H, p=1)

    total_loss = loss_pred + alpha * loss_indep + beta * loss_spar

    return total_loss, {
        'loss_pred': loss_pred.item(),
        'loss_indep': loss_indep.item(),
        'loss_spar': loss_spar.item()
    }
# train_epoch, validate 함수는 기존과 동일하게 유지
def train_epoch(model, dataloader, optimizer, cfg, device, x_scaler, u_scaler, scaler=None):
    model.train()
    total_loss = 0
    logs = {'loss_pred':0, 'loss_indep':0, 'loss_spar':0}
    
    # Use scaler if provided (Mixed Precision)
    use_amp = (scaler is not None)
    
    for x_batch, u_batch in dataloader:
        x_batch = x_batch.float().to(device, non_blocking=True)
        u_batch = u_batch.float().to(device, non_blocking=True)
        
        # Normalize
        x_norm = x_scaler.transform(x_batch)
        u_norm = u_scaler.transform(u_batch)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Mixed Precision Context
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, batch_logs = bilinear_koopman_loss(model, x_norm, u_norm, cfg)
            
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, batch_logs = bilinear_koopman_loss(model, x_norm, u_norm, cfg)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        for k, v in batch_logs.items():
            logs[k] += v
            
    avg_loss = total_loss / len(dataloader)
    for k in logs:
        logs[k] /= len(dataloader)
        
    return avg_loss, logs

def validate(model, dataloader, cfg, device, x_scaler, u_scaler):
    model.eval()
    total_loss = 0
    logs = {'loss_pred':0, 'loss_indep':0, 'loss_spar':0}
    
    # Determine if we should use AMP for val
    use_amp = (device.type == 'cuda')
    
    with torch.no_grad():
        for x_batch, u_batch in dataloader:
            x_batch = x_batch.float().to(device, non_blocking=True)
            u_batch = u_batch.float().to(device, non_blocking=True)
            
            x_norm = x_scaler.transform(x_batch)
            u_norm = u_scaler.transform(u_batch)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, batch_logs = bilinear_koopman_loss(model, x_norm, u_norm, cfg)
            else:
                loss, batch_logs = bilinear_koopman_loss(model, x_norm, u_norm, cfg)
            
            total_loss += loss.item()
            for k, v in batch_logs.items():
                logs[k] += v
                
    avg_loss = total_loss / len(dataloader)
    for k in logs:
        logs[k] /= len(dataloader)
        
    return avg_loss, logs
