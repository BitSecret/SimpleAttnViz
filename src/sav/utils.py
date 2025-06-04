import numpy as np
import torch
import torch.optim as optim
import json
import random


def get_config():
    with open("../../data/config.json", "r") as f:
        json_config = json.load(f)
    return json_config


def set_seed():
    seed = get_config()["data"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_multi_head_attention():
    n, h, d_v, d_model = 3, 4, 8, 4

    attentions = [torch.rand(n, d_v) for _ in range(h)]
    multi_head_attention = torch.cat(attentions, dim=1)

    weights = [torch.rand(d_v, d_model) for _ in range(h)]
    w_o = torch.cat(weights, dim=0)

    print(torch.mm(multi_head_attention, w_o))
    print(torch.sum(torch.stack([torch.mm(attentions[i], weights[i]) for i in range(h)]), dim=0))


def solve_axv(X, Y, p, q, max_iter=1000, lr=0.001, tol=1e-7):
    """
    给定 X 、Y ，通过优化问题求解 A 和 V ，满足 AXV ≈ Y ，其中 A 是行随机矩阵。
    :param X: m * n
    :param Y: p * q
    :param p: A的行数
    :param q: V的列数
    :param max_iter: 最大迭代次数
    :param lr: 学习率
    :param tol: 最小可接受误差
    :return: A 和 V 的近似解，不唯一
    """
    m, n = X.shape

    # 初始化 A 和 V（随机初始化）
    A = torch.randn(p, m, requires_grad=True)
    V = torch.randn(n, q, requires_grad=True)

    optimizer = optim.Adam([A, V], lr=lr)

    for epoch in range(max_iter):
        optimizer.zero_grad()

        # 对A应用Softmax约束，使其满足行随机矩阵
        A_softmax = torch.nn.functional.softmax(A, dim=1)
        AXV = torch.mm(torch.mm(A_softmax, X), V)

        # 计算损失（Fresenius 范数）
        loss = torch.norm(AXV - Y, p='fro')

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印进度
        if epoch % 100 == 0:
            print(f"Iter {epoch}, Loss: {loss.item():.6f}")

        # 检查收敛
        if loss.item() < tol:
            print(f"Converged at iteration {epoch}")
            break

    return torch.nn.functional.softmax(A, dim=1).detach().numpy(), V.detach().numpy()


def test_solve_axv():
    p, m, n, q = 4, 8, 16, 3
    X = torch.randn(m, n, requires_grad=True)
    Y = torch.randn(p, q, requires_grad=True)

    A, V = solve_axv(X, Y, p, q, max_iter=10000, lr=0.001)

    print("\nSolution:")
    print("A:\n", A)
    print("V:\n", V)

    print("\nY:\n", Y.detach().numpy())
    print("\nVerification (AXV):\n", np.dot(np.dot(A, X.detach().numpy()), V))


if __name__ == '__main__':
    set_seed()
    show_multi_head_attention()
    # test_solve_axv()
