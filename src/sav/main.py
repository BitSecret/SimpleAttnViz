from model import get_model
from data import get_datasets
from utils import get_config, set_seed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

config = get_config()


def train_one_epoch(device, model, data_loader, optimizer, criterion, epoch, epochs, training):
    if training is True:
        model.train()
        print_label = "Training"
    else:
        model.eval()
        print_label = "Evaluating"
    running_loss = []
    timing = time.time()

    time.sleep(0.5)  # 防止进度条与日志输出冲突
    loop = tqdm(data_loader, leave=False)  # 显示训练进度
    for image, label in loop:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs, _ = model(image)  # 预测

        loss = criterion(outputs, label)
        loss.backward()  # 反向传播和优化
        optimizer.step()  # 优化

        running_loss.append(loss.item())
        loop.set_description(f"{print_label} Epoch [{epoch}/{epochs}]")  # 更新进度条
        loop.set_postfix(loss=loss.item())
        loop.set_postfix(timing=time.time() - timing)
    loop.close()

    return sum(running_loss) / len(data_loader)


def main(device):
    set_seed()
    train_dataset, val_dataset, test_dataset = get_datasets()
    model = get_model().to(device)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    epochs = config["training"]["epochs"]
    bst_model_path = f"../../data/checkpoints/bst.pth"
    bst_val_loss = 1e5
    bst_loss_epoch = 1
    bst_val_pc = 0
    bst_val_pc_epoch = 1
    bst_test_pc = 0
    bst_test_pc_epoch = 1
    log = []
    for epoch in range(1, epochs + 1):
        timing = time.time()
        train_loss = train_one_epoch(device, model, train_loader, optimizer, criterion, epoch, epochs, training=True)
        val_loss = train_one_epoch(device, model, val_loader, optimizer, criterion, epoch, epochs, training=False)

        if val_loss < bst_val_loss:
            bst_val_loss = val_loss
            bst_loss_epoch = epoch

        val_pc = test(device, model, val_dataset)
        if val_pc > bst_val_pc:
            bst_val_pc = val_pc
            bst_val_pc_epoch = epoch
            torch.save(model.state_dict(), bst_model_path)

        test_pc = test(device, model, test_dataset)
        if test_pc > bst_test_pc:
            bst_test_pc = test_pc
            bst_test_pc_epoch = epoch

        log.append(f"Epoch {epoch}/{epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f} (Best={bst_val_loss:.4f}, epoch={bst_loss_epoch}), "
                   f"Val Precision: {val_pc:.4f} (Best={bst_val_pc:.4f}, epoch={bst_val_pc_epoch}), "
                   f"Test Precision: {test_pc:.4f} (Best={bst_test_pc:.4f}, epoch={bst_test_pc_epoch}), "
                   f"Timing: {time.time() - timing:.2f}s")
        print(log[-1])

    with open("../../data/outputs/train_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))


def test(device, model, dataset):
    data_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    model.eval()
    correct_count = 0
    with torch.no_grad():
        for image, label in data_loader:
            image, label = image.to(device), label.to(device)

            outputs, _ = model(image)
            outputs = torch.argmax(outputs, dim=1)

            correct = (outputs == label).sum().item()
            correct_count += correct

    return correct_count / len(data_loader)


def visualize(device, shape):
    set_seed()

    _, _, test_dataset = get_datasets()
    raw_images = [torch.squeeze(test_dataset.raw_data[i][0]) for i in range(shape[0] * shape[1])]
    image_patches = [test_dataset.data[i][0] for i in range(shape[0] * shape[1])]
    image_patches = torch.stack(image_patches).to(device)

    model = get_model().to(device)
    bst_model_path = f"../../data/checkpoints/bst.pth"
    model.load_state_dict(torch.load(bst_model_path, map_location=torch.device(device), weights_only=True))

    _, attention_scores = model(image_patches)  # attention_scores: [N, batch_size, h, seq_len, seq_len]
    for i in range(len(attention_scores)):
        # attention_scores[i]: [batch_size, h, seq_len, seq_len]
        attention_scores[i] = torch.sum(attention_scores[i], dim=1)
        # attention_scores[i]: [batch_size, seq_len, seq_len]

    eye_matrix = torch.eye(attention_scores[0].size(2)).to(device)
    attention_matrix = [torch.eye(attention_scores[0].size(2)).to(device) for _ in range(shape[0] * shape[1])]

    for i in range(len(attention_scores))[::-1]:  # 注意力矩阵前向传播
        for batch_idx in range(shape[0] * shape[1]):
            attention_matrix[batch_idx] = torch.mm(
                attention_matrix[batch_idx],
                torch.add(eye_matrix, attention_scores[i][batch_idx])
            )

    C, H, W = raw_images[0].shape
    h_block, w_block = config["data"]["patch_size"]

    n_h = H // h_block
    n_w = W // w_block

    for batch_idx in range(shape[0] * shape[1]):  # 注意力归一化
        attention_matrix[batch_idx] = attention_matrix[batch_idx][0][1:]  # 取 token [CLS] 的注意力，并去掉 [CLS]
        attention_matrix[batch_idx] = attention_matrix[batch_idx].reshape(n_h, n_w)  # 调整为block大小
        attention_matrix[batch_idx] = attention_matrix[batch_idx].unsqueeze(0).unsqueeze(0)  # 调整到4D
        attention_matrix[batch_idx] = F.interpolate(  # 插值
            attention_matrix[batch_idx],
            size=(H, W),
            mode='bicubic',  # 插值方式，也可以用trilinear
            align_corners=False  # 角像素对齐方式
        )
        attention_matrix[batch_idx] = attention_matrix[batch_idx].squeeze()  # 移除多余的维度
        min_val = attention_matrix[batch_idx].min()  # 缩放至 [0, 1]
        max_val = attention_matrix[batch_idx].max()
        attention_matrix[batch_idx] = (attention_matrix[batch_idx] - min_val) / (max_val - min_val + 1e-32)

    plt.figure(figsize=(shape[1] * 12, shape[0] * 12))
    for i in range(shape[0] * shape[1]):
        plt.subplot(shape[0], shape[1], i + 1)

        image = raw_images[i].numpy().transpose(1, 2, 0)  # 三通道 [H, W, 3], 值0~1

        heatmap = attention_matrix[i].cpu().detach().numpy()  # 单通道 [H, W], 值0~1
        cmap = plt.get_cmap('coolwarm')  # 热力图颜色映射（coolwarm红-蓝渐变）
        heatmap_rgb = cmap(heatmap)[:, :, :3]  # 转为RGB [H, W, 3]
        # heatmap_rgb = heatmap_rgb * (heatmap[:, :, np.newaxis] ** 0.5)  # 增强高权重区域

        alpha = 0.4  # 基础透明度
        overlay = image * (1 - alpha) + heatmap_rgb * alpha  # 叠加图像与热力图（保留原图色彩）

        plt.imshow(np.clip(overlay, 0, 1))  # 确保值在 0-1 范围内
        plt.axis('off')

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig("../../data/outputs/fig.pdf")
    plt.show()


if __name__ == '__main__':
    main(device="cuda:0" if torch.cuda.is_available() else "cpu")
    visualize(device="cuda:0" if torch.cuda.is_available() else "cpu", shape=(4, 6))
