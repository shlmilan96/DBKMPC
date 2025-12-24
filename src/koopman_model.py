import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBilinearKoopman(nn.Module):
    def __init__(self, cfg):
        super(DeepBilinearKoopman, self).__init__()
        
        # Dimensions
        self.state_dim = cfg['dims']['state']
        self.control_dim = cfg['dims']['control']
        self.lifted_dim = cfg['dims']['lifted']
        self.full_dim = cfg['dims']['full_state']
        
        hidden_dim = 128
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.lifted_dim)
        )
        
        self.A = nn.Parameter(torch.randn(self.full_dim, self.full_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(self.full_dim, self.control_dim) * 0.01)
        
        # 최적화: ParameterList 대신 3차원 텐서 [control_dim, full_dim, full_dim] 사용
        self.H = nn.Parameter(torch.randn(self.control_dim, self.full_dim, self.full_dim) * 0.01)
        
    def encode(self, x):
        return self.encoder(x)
    
    def get_full_state(self, x):
        g_x = self.encode(x)
        return torch.cat([x, g_x], dim=1)
    
    def forward_step(self, z, u):
        """
        벡터화된 빌리니어 역학 계산
        """
        # 선형 부분: A z + B u
        z_next = z @ self.A.T + u @ self.B.T
        
        # 최적화: torch.einsum을 사용하여 루프 제거
        # H: [m, d, d], z: [B, d], u: [B, m] -> [B, d]
        # m: 제어 차원, d: 상태 차원, B: 배치 크기
        bilinear = torch.einsum('mdj,bj,bm->bd', self.H, z, u)
        z_next = z_next + bilinear
            
        return z_next

    def forward(self, x, u):
        z = self.get_full_state(x)
        z_next = self.forward_step(z, u)
        return z_next
