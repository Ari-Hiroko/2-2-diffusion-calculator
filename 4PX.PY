import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# --- 1. 基础配置与字形定义 ---
GLYPHS = {
    0: [[1, 1], [1, 1]], 1: [[0, 1], [0, 1]], 2: [[1, 1], [1, 0]], 3: [[1, 1], [0, 1]],
    4: [[1, 0], [0, 1]], 5: [[1, 0], [1, 1]], 6: [[1, 0], [1, 0]], 7: [[1, 1], [0, 0]],
    8: [[0, 1], [1, 0]], 9: [[0, 1], [1, 1]],
    '+': [[0, 0], [0, 1]], '-': [[0, 0], [1, 0]], '*': [[1, 0], [0, 0]],
    '/': [[0, 1], [0, 0]], '%': [[0, 0], [1, 1]], '=': [[1, 1], [1, 1]]
}
OPS_LIST = ['+', '-', '*', '/', '%']

def get_glyph(char):
    return np.array(GLYPHS[char])

def create_image(a, op, b, res):
    """ 创建 2x10 图像，值域 [-1, 1] """
    grid = np.zeros((2, 10))
    grid[:, 0:2] = get_glyph(a)
    grid[:, 2:4] = get_glyph(op)
    grid[:, 4:6] = get_glyph(b)
    
    res = int(res)
    res_str = f"{res:02d}"
    if len(res_str) > 2: res_str = "99"
    grid[:, 6:8] = get_glyph(int(res_str[0]))
    grid[:, 8:10] = get_glyph(int(res_str[1]))
    
    return (grid * 2) - 1

# --- 2. 数据集 (保持不变，增加一点数据量) ---
class MathGridDataset(Dataset):
    def __init__(self, size=8000):
        self.data = []
        for _ in range(size):
            op = np.random.choice(OPS_LIST)
            a = np.random.randint(0, 10)
            b = np.random.randint(0, 10)
            
            res = 0
            if op == '+': res = a + b
            elif op == '-': res = a - b
            elif op == '*': res = a * b
            elif op == '/': res = a // b if b != 0 else 0
            elif op == '%': res = a % b if b != 0 else 0
            
            if res < 0: res = 0
            if res > 99: res = 99 # 限制范围
            
            img = create_image(a, op, b, res)
            self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回 (1, 2, 10)
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)

# --- 3. 改进的模型：LogicMixer Diffuser ---
# 这是一个混合模型：CNN处理局部特征，中间夹着MLP处理全局数学逻辑

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LogicDiffuser(nn.Module):
    def __init__(self):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(32),
            nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        
        # 输入通道: 1(原图) + 1(Condition) + 2(坐标 x,y) = 4
        self.conv_in = nn.Conv2d(4, 64, 3, padding=1)
        
        # 编码器 (局部特征)
        self.enc1 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU())
        
        # 核心逻辑层 (MLP - 全局视野)
        # 将 2x10x128 展平为 2560 向量，通过全连接层进行"思考"
        self.flatten = nn.Flatten()
        self.logic_block = nn.Sequential(
            nn.Linear(128 * 2 * 10 + 64, 1024), # +64 是为了注入时间信息
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 128 * 2 * 10)
        )
        
        # 解码器 (还原为图像)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU())
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        B, C, H, W = x.shape
        
        # 1. 构建坐标编码 (Coordinate Embeddings)
        # 这让模型知道像素的位置 (左边是输入，右边是输出)
        y_coords = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, 1, H, W).to(x.device)
        x_coords = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, 1, H, W).to(x.device)
        
        # 拼接输入: x(含Mask), y_coords, x_coords
        model_input = torch.cat([x, y_coords, x_coords], dim=1)
        
        # 2. 初始卷积
        h = self.conv_in(model_input)
        h = self.enc1(h)
        h = self.enc2(h)
        
        # 3. 全局逻辑混合 (Bottleneck)
        t_emb = self.time_mlp(t) # (B, 64)
        flat = self.flatten(h)   # (B, 2560)
        
        # 将图像特征和时间特征拼在一起送入 MLP
        logic_in = torch.cat([flat, t_emb], dim=1)
        logic_out = self.logic_block(logic_in)
        
        # 还原回图像形状
        h_logic = logic_out.view(B, 128, H, W)
        
        # 残差连接 (让逻辑层只学习变化量)
        h = h + h_logic
        
        # 4. 输出
        h = self.dec1(h)
        return self.out(h)

# --- 4. 扩散参数 ---
T = 300
beta_start, beta_end = 0.0001, 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def get_index(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion(x_0, t, device):
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = get_index(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_bar_t = get_index(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise

# --- 5. 训练过程 ---
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    dataset = MathGridDataset(size=10000)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = LogicDiffuser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # AdamW 更好
    
    epochs = 100 
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_0 in dataloader:
            x_0 = x_0.to(device)
            B = x_0.shape[0]
            t = torch.randint(0, T, (B,), device=device).long()
            
            # 制造噪声图
            x_noisy, noise = forward_diffusion(x_0, t, device)
            
            # 制造条件图 (Condition)
            # 我们把结果部分 (后4列) 涂黑 (-1)
            condition = x_0.clone()
            condition[:, :, :, 6:10] = -1 
            
            # 输入给模型: (Noisy_Image, Condition)
            # 并在模型内部会加上坐标信息
            model_input = torch.cat([x_noisy, condition], dim=1)
            
            noise_pred = model(model_input, t)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.5f}")
    return model

# --- 6. 修正后的采样/推理 (Inpainting) ---
@torch.no_grad()
def visual_solve(model, a, op, b):
    device = next(model.parameters()).device
    model.eval()
    
    # 1. 准备题目 (Condition)
    cond_grid = create_image(a, op, b, 0) # 结果先填0，不重要
    cond_clean = torch.tensor(cond_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 构造 mask: 前6列是已知的(1)，后4列是未知的(0)
    mask = torch.ones_like(cond_clean)
    mask[:, :, :, 6:10] = 0
    
    # 构造模型输入的 Condition 通道 (未知区域填 -1)
    cond_input = cond_clean.clone()
    cond_input[:, :, :, 6:10] = -1
    
    # 2. 从纯噪声开始
    img = torch.randn_like(cond_clean)
    
    for i in range(T-1, -1, -1):
        t = torch.tensor([i], device=device).long()
        
        # --- 关键修改: RePaint 策略 ---
        # 我们不直接把干净的题目贴上去，而是贴上“带噪”的题目
        # 这样模型看到的整体噪声水平是协调的
        
        # 计算当前时刻题目部分的理论噪声形态
        noise_for_known = torch.randn_like(cond_clean)
        alpha_bar_t = alphas_cumprod[i].to(device)
        known_part_noisy = torch.sqrt(alpha_bar_t) * cond_clean + torch.sqrt(1 - alpha_bar_t) * noise_for_known
        
        # 将“已知部分的噪声图”和“生成部分的当前图”混合
        img = mask * known_part_noisy + (1 - mask) * img
        
        # 预测噪声
        model_input = torch.cat([img, cond_input], dim=1)
        predicted_noise = model(model_input, t)
        
        # 去噪一步 (DDPM Standard Step)
        beta = betas[i].to(device)
        alpha = alphas[i].to(device)
        alpha_bar = alphas_cumprod[i].to(device)
        
        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)
            
        img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(beta) * noise

    # 最后一步再强制覆盖一次干净的题目，确保显示完美
    img = mask * cond_clean + (1 - mask) * img
    
    return img.cpu().numpy().squeeze()

def plot_solution(grid, title="Prediction"):
    plt.figure(figsize=(6, 2))
    plt.imshow(grid, cmap='gray', vmin=-1, vmax=1)
    plt.axvline(x=1.5, color='red', linewidth=1)
    plt.axvline(x=3.5, color='red', linewidth=1)
    plt.axvline(x=5.5, color='red', linewidth=1)
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    model = train()
    
    # 测试
    cases = [
        (3, '+', 2, '5'),
        (8, '*', 9, '72'),
        (9, '/', 3, '3'),
        (1, '+', 1, '2')
    ]
    
    for a, op, b, exp in cases:
        print(f"Calculating: {a} {op} {b} = ?")
        res_grid = visual_solve(model, a, op, b)
        plot_solution(res_grid, title=f"{a} {op} {b} = ? (Expect {exp})")