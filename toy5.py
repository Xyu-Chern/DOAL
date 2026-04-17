import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 模型与环境定义 ---
class MLP(nn.Module):
    out_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x); x = nn.relu(x)
        x = nn.Dense(128)(x); x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)

@struct.dataclass
class MixtureEnv:
    gold_params: dict 
    gold_network: MLP
    def get_gold_q(self, action):
        return self.gold_network.apply(self.gold_params, action).squeeze()

# --- 2. 核心实验逻辑 ---
def run_full_experiment(master_seed=42, N=5):

    time_steps = 10000

    main_key = jax.random.PRNGKey(master_seed)
    env_fixed_key = jax.random.PRNGKey(0) 
    data_key, train_key = jax.random.split(main_key, 2)

    # A. 初始化 Ground Truth
    gt_model = MLP(out_dim=1)
    gt_params = gt_model.init(env_fixed_key, jnp.ones((1, 2)))
    env = MixtureEnv(gt_params, gt_model)

    # B. 网格定义
    res_search = 150
    val_r = np.linspace(-2, 2, res_search)
    AX, AY = np.meshgrid(val_r, val_r)
    valid_grid = jnp.stack([AX.ravel(), AY.ravel()], axis=-1)
    
    plt_r = np.linspace(-3, 3, 100)
    PX, PY = np.meshgrid(plt_r, plt_r)
    full_plot_grid = jnp.stack([PX.ravel(), PY.ravel()], axis=-1)

    # C. 生成数据 (N=5)
    dataset = jax.random.uniform(data_key, (N, 2), minval=-2.0, maxval=2.0)
    rewards = jax.vmap(env.get_gold_q)(dataset) + 0.1 * jax.random.normal(data_key, (N,))

    # D. 预训练代理 Q-Network (200步)
    q_net = MLP(out_dim=1)
    q_params = q_net.init(train_key, jnp.ones((1, 2)))
    q_opt = optax.adam(1e-3)
    q_state = q_opt.init(q_params)

    @jax.jit
    def train_step(p, s, b_a, b_r):
        loss_fn = lambda p_: jnp.mean((q_net.apply(p_, b_a).squeeze() - b_r) ** 2)
        grads = jax.grad(loss_fn)(p)
        upd, s = q_opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s

    for _ in range(time_steps):
        q_params, q_state = train_step(q_params, q_state, dataset, rewards)

    # --- 逻辑工具：计算点到数据集的最短距离遮罩 ---
    def get_data_proximity_mask(grid_points, delta):
        # 计算 grid_points 中每个点到 dataset 中 5 个点的距离
        # grid_points: [M, 2], dataset: [N, 2] -> dists: [M, N]
        dists = jnp.sqrt(jnp.sum((grid_points[:, None, :] - dataset[None, :, :])**2, axis=-1))
        min_dists = jnp.min(dists, axis=1)
        return jnp.where(min_dists <= delta, 1.0, 0.0)

    # --- 3. 绘图 ---
    fig = plt.figure(figsize=(28, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # --- 第一列：Ground Truth (展示热力值) ---
    ax_gt = fig.add_subplot(2, 6, 1)
    gt_q_full = jax.vmap(env.get_gold_q)(full_plot_grid).reshape(100, 100)
    c0 = ax_gt.contourf(PX, PY, gt_q_full, levels=30, cmap='viridis')
    ax_gt.add_patch(plt.Rectangle((-2, -2), 4, 4, lw=2, ec='white', fc='none', ls='--'))
    ax_gt.scatter(dataset[:, 0], dataset[:, 1], c='white', edgecolors='black', s=70)
    
    # 寻找 GT 在 [-2, 2] 内的最优
    gt_q_valid = jax.vmap(env.get_gold_q)(valid_grid)
    gt_best_a = valid_grid[jnp.argmax(gt_q_valid)]
    ax_gt.scatter(gt_best_a[0], gt_best_a[1], marker='+', color='lime', s=250, lw=3)
    
    ax_gt.set_title("Ground Truth Q\n(True Landscape)", fontsize=14, fontweight='bold')
    ax_gt.set_aspect('equal')
    fig.colorbar(c0, ax=ax_gt, fraction=0.046, pad=0.04)

    # --- 第一行后续：BRAC 系列 ---
    lams = [0, 0.1, 0.5, 1, 10]
    for i, lam in enumerate(lams):
        ax = fig.add_subplot(2, 6, i + 2)
        q_v = q_net.apply(q_params, valid_grid).squeeze()
        d_v = jnp.mean(jnp.sum((valid_grid[:, None, :] - dataset[None, :, :])**2, axis=-1), axis=-1)
        h_v = q_v - lam * d_v
        best_a = valid_grid[jnp.argmax(h_v)]
        real_q = env.get_gold_q(best_a)
        
        q_f = q_net.apply(q_params, full_plot_grid).squeeze()
        d_f = jnp.mean(jnp.sum((full_plot_grid[:, None, :] - dataset[None, :, :])**2, axis=-1), axis=-1)
        h_map = (q_f - lam * d_f).reshape(100, 100)
        
        c = ax.contourf(PX, PY, h_map, levels=30, cmap='magma')
        ax.add_patch(plt.Rectangle((-2, -2), 4, 4, lw=1, ec='white', fc='none', ls='--', alpha=0.4))
        ax.scatter(dataset[:, 0], dataset[:, 1], c='white', edgecolors='black', s=40)
        ax.scatter(best_a[0], best_a[1], marker='o', color='cyan', edgecolors='black', s=120)
        ax.set_title(f"BRAC (λ={lam})", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"Real Q: {real_q:.2f}", color='blue', fontweight='bold')
        ax.set_aspect('equal')
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    # --- 第二行：DOAL 系列 (基于数据点的局部 Mask) ---
    deltas = [0.2, 0.5, 1.0, 2.0]
    for i, delta in enumerate(deltas):
        ax = fig.add_subplot(2, 6, i + 7)
        
        # 1. 计算遮罩
        mask_v = get_data_proximity_mask(valid_grid, delta)
        mask_f = get_data_proximity_mask(full_plot_grid, delta)
        
        # 2. 寻找有效范围内的最优动作
        q_v = q_net.apply(q_params, valid_grid).squeeze()
        h_v_doal = q_v * mask_v
        best_a_doal = valid_grid[jnp.argmax(h_v_doal)]
        real_q_doal = env.get_gold_q(best_a_doal)
        
        # 3. 绘制热力图
        q_f = q_net.apply(q_params, full_plot_grid).squeeze()
        h_map_doal = (q_f * mask_f).reshape(100, 100)
        
        c = ax.contourf(PX, PY, h_map_doal, levels=30, cmap='magma')
        ax.add_patch(plt.Rectangle((-2, -2), 4, 4, lw=1, ec='white', fc='none', ls='--', alpha=0.4))
        ax.scatter(dataset[:, 0], dataset[:, 1], c='white', edgecolors='black', s=40)
        
        # 绘制每个数据点的圆圈范围
        for d_pt in dataset:
            circle = plt.Circle((d_pt[0], d_pt[1]), delta, color='white', fill=False, ls=':', alpha=0.2)
            ax.add_patch(circle)
            
        ax.scatter(best_a_doal[0], best_a_doal[1], marker='*', color='yellow', edgecolors='black', s=180)
        ax.set_title(f"DOAL (δ={delta})", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"Real Q: {real_q_doal:.2f}", color='darkred', fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Analysis: Data-Centric Constraints (N={N}, {time_steps} Steps)\nDOAL uses a union of balls around data points as the valid Q domain", 
                 fontsize=22, y=0.98, fontweight='bold')

    plt.savefig(f"final_{N}.png", bbox_inches='tight', dpi=300)
    print("图像已成功保存。GT 展示了地形，DOAL 现在以数据点为圆心进行约束。")
    plt.show()

if __name__ == "__main__":
    run_full_experiment(master_seed=42, N=5)
    run_full_experiment(master_seed=42, N=10)
    run_full_experiment(master_seed=42, N=50)
    run_full_experiment(master_seed=42, N=100)
    run_full_experiment(master_seed=42, N=500)