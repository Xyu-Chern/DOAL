

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import struct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D



class MLP(nn.Module):
    out_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x); x = nn.relu(x)
        x = nn.Dense(128)(x); x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)

@struct.dataclass
class MixtureEnv:
    # 存储随机生成的参数和网络结构
    gold_params: dict 
    gold_network: MLP  # 随机生成的 ground truth 网络
    noise_std: float = 0.5
    
    def get_reward(self, key, action):
        """
        在离线 RL 设定中，Reward 通常直接由当前 (s, a) 决定。
        这里我们假设 Q(s, a) 就是该环境的真实 Reward 来源。
        """
        key, subkey = jax.random.split(key)
        
        # 1. 调用模型计算真值。注意 Flax apply 的标准格式。
        # 假设 action 已经是连续向量或离散索引对应的 embedding
        q_value = self.gold_network.apply(self.gold_params, action).squeeze()
        
        # 2. 实际 reward = 真实 Q 值 + 观测噪声
        reward = q_value + self.noise_std * jax.random.normal(subkey, q_value.shape)
        
        return reward
        
    def get_gold_q(self, action):
        """返回给定动作的 ground truth Q 值 (用于绘图对比)"""
        # 确保输入维度正确，返回标量或数组
        return self.gold_network.apply(self.gold_params, action).squeeze()

# --- 2. 核心训练函数 ---
def train_and_get_actions(num_samples, key):
    env_key, data_key, train_key, model_key = jax.random.split(key, 4)
    # env = MixtureEnv(alphas=jnp.array([0.6, 0.4]), betas=jnp.array([[1.5, 1.5], [-1.5, -1.5]]))
    gt_model = MLP(out_dim=1)
    # 用独立的 model_key 初始化，保证环境是随机且固定的
    gt_params = gt_model.init(model_key, jnp.ones((1, 2)))
    env = MixtureEnv(gt_params, gt_model)

    # 找到理论最优解（用于绘图参考）
    res_search = 100
    s_range = np.linspace(-2, 2, res_search)
    SX, SY = np.meshgrid(s_range, s_range)
    search_grid = jnp.stack([SX.ravel(), SY.ravel()], axis=-1)
    best_action_gold = search_grid[jnp.argmax(jax.vmap(env.get_gold_q)(search_grid))]

    # 生成离线数据集
    actions_dataset = jax.random.uniform(data_key, (num_samples, 2), minval=-2.0, maxval=2.0)
    rewards_dataset = jax.vmap(env.get_reward)(jax.random.split(data_key, num_samples), actions_dataset)

    # 1. 训练 Q-Network (Critic)
    q_net = MLP(out_dim=1)
    q_params = q_net.init(train_key, jnp.ones((1, 2)))
    q_opt = optax.adam(1e-3)
    q_state = q_opt.init(q_params)

    @jax.jit
    def train_q_step(p, s, b_a, b_r):
        loss_fn = lambda p_: jnp.mean((q_net.apply(p_, b_a).squeeze() - b_r) ** 2)
        grads = jax.grad(loss_fn)(p)
        upd, s = q_opt.update(grads, s, p)
        return optax.apply_updates(p, upd), s

    for _ in range(15000):
        q_params, q_state = train_q_step(q_params, q_state, actions_dataset, rewards_dataset)

    # 2. 优化策略 (Actor)
    def optimize_actor(mode="brac", lam=1.2):
        actor_net = MLP(out_dim=2)
        a_params = actor_net.init(train_key, jnp.ones((1, 1)))
        a_opt = optax.adam(1e-3)
        a_state = a_opt.init(a_params)

        @jax.jit
        def actor_step(ap, opt_s):
            def loss_fn(p):
                # 策略输出限制在 [-2, 2]
                a = jnp.tanh(actor_net.apply(p, jnp.ones((1, 1)))) * 2.0
                if mode == "brac":
                    q_val = q_net.apply(q_params, a).squeeze()
                    # 行为约束：惩罚偏离数据集的动作
                    dist_penalty = jnp.mean(jnp.sum((a - actions_dataset)**2, axis=-1))
                    return -q_val + lam * dist_penalty
                elif mode == "doal":
                    # 扰动鲁棒优化：在最差的邻域内寻找最优
                    dq_da = jax.grad(lambda act: q_net.apply(q_params, act).squeeze())(a)
                    dx = -0.25 * jnp.sign(dq_da) # 寻找Q值下降的方向
                    dx = jax.lax.stop_gradient(dx)
                    return -q_net.apply(q_params, a + dx).squeeze()
                return 0.0
            grads = jax.grad(loss_fn)(ap)
            upd, opt_s = a_opt.update(grads, opt_s, ap)
            return optax.apply_updates(ap, upd), opt_s

        for _ in range(10000):
            a_params, a_state = actor_step(a_params, a_state)
        
        final_a = jnp.tanh(actor_net.apply(a_params, jnp.ones((1, 1)))) * 2.0
        perturb_a = None
        if mode == "doal":
            dq_da = jax.grad(lambda act: q_net.apply(q_params, act).squeeze())(final_a)
            dx = -0.25 * jnp.sign(dq_da)
            perturb_a = final_a + dx
            
        return final_a, perturb_a

    a_brac, _ = optimize_actor("brac")
    a_doal, a_doal_p = optimize_actor("doal")
    
    return {
        "dataset": actions_dataset, "gold_best": best_action_gold, 
        "brac": a_brac, "doal": a_doal, "doal_p": a_doal_p, 
        "q_params": q_params, "q_net": q_net, "env": env
    }

# --- 3. 绘图主程序 ---
sample_sizes = [5, 10, 30, 100]
fig, axes = plt.subplots(len(sample_sizes), 3, figsize=(15, 18))
plt.subplots_adjust(hspace=0.4, wspace=0.15)

master_key = jax.random.PRNGKey(42)

for i, n in enumerate(sample_sizes):
    res = train_and_get_actions(n, jax.random.fold_in(master_key, i))
    env = res["env"]
    
    # 准备等高线数据
    plot_r = np.linspace(-2.5, 2.5, 80)
    PX, PY = np.meshgrid(plot_r, plot_r)
    grid = jnp.stack([PX.ravel(), PY.ravel()], axis=-1)
    q_map_est = res["q_net"].apply(res["q_params"], grid).reshape(80, 80)
    q_map_gold = jax.vmap(env.get_gold_q)(grid).reshape(80, 80)

    # --- 第一列: Ground Truth Q ---
    ax_gt = axes[i, 0]
    im_gt = ax_gt.contourf(PX, PY, q_map_gold, levels=30, cmap='viridis', alpha=0.9)
    ax_gt.scatter(res["dataset"][:, 0], res["dataset"][:, 1], c='white', edgecolors='black', s=25, alpha=0.5)
    ax_gt.scatter(res["gold_best"][0], res["gold_best"][1], marker='+', s=150, color='lime', linewidths=2.5)
    ax_gt.set_ylabel(f"Samples N={n}", fontsize=12, fontweight='bold')
    if i == 0: ax_gt.set_title("Ground Truth Q\n(True Landscape)", fontsize=13, color='darkgreen', fontweight='bold')
    ax_gt.set_xlim(-2.5, 2.5); ax_gt.set_ylim(-2.5, 2.5)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])

    # --- 循环绘制 BRAC 和 DOAL ---
    for j, mode in enumerate(["brac", "doal"]):
        ax = axes[i, j + 1]
        im_est = ax.contourf(PX, PY, q_map_est, levels=30, cmap='magma', alpha=0.9)
        ax.scatter(res["dataset"][:, 0], res["dataset"][:, 1], c='white', edgecolors='black', s=25, alpha=0.6)
        
        if mode == "brac":
            curr_a = res["brac"]
            ax.scatter(curr_a[0,0], curr_a[0,1], marker='o', s=120, color='cyan', edgecolors='black', zorder=10)
        else:
            curr_a = res["doal"]
            a_orig = curr_a[0]; a_pert = res["doal_p"][0]
            ax.scatter(a_orig[0], a_orig[1], marker='*', s=180, color='yellow', edgecolors='black', zorder=10)
            ax.plot([a_orig[0], a_pert[0]], [a_orig[1], a_pert[1]], 'w--', lw=1.5, alpha=0.8)
            ax.scatter(a_pert[0], a_pert[1], marker='x', s=80, color='red', linewidths=2, zorder=11)
            
            # 在数据集点周围画出扰动容忍半径（示意）
            for pt in res["dataset"]:
                circle = plt.Circle((pt[0], pt[1]), 0.15, color='white', fill=False, linestyle=':', alpha=0.3, lw=0.8)
                ax.add_artist(circle)

        # 标注性能
        real_q = env.get_gold_q(curr_a.squeeze())
        max_q = env.get_gold_q(res["gold_best"])
        ax.set_xlabel(f"Real Q: {real_q:.2f} (Target: {max_q:.2f})", fontsize=10, fontweight='bold')

        if i == 0: 
            title = "BRAC (Constraint)" if mode=="brac" else "DOAL (Perturb Robust)"
            ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([]); ax.set_yticks([])

# --- 4. 颜色条与图例 ---
# 真实值色条
cbar_ax_gt = fig.add_axes([0.07, 0.3, 0.015, 0.4])
fig.colorbar(im_gt, cax=cbar_ax_gt).set_label('Ground Truth Score')
cbar_ax_gt.yaxis.set_ticks_position('left')

# 估计值色条
cbar_ax_est = fig.add_axes([0.93, 0.3, 0.015, 0.4])
fig.colorbar(im_est, cax=cbar_ax_est).set_label('Estimated Q Value')

# 图例
legend_elements = [
    Line2D([0], [0], marker='+', color='lime', label='True Optimal', markersize=10, linestyle='None'),
    Line2D([0], [0], marker='o', color='white', markerfacecolor='cyan', label='BRAC Action', markersize=10, markeredgecolor='black', linestyle='None'),
    Line2D([0], [0], marker='*', color='white', markerfacecolor='yellow', label='DOAL Action', markersize=12, markeredgecolor='black', linestyle='None'),
    Line2D([0], [0], marker='x', color='red', label='Worst-case Perturb', markersize=8, linestyle='None'),
    Line2D([0], [0], marker='o', color='white', markerfacecolor='white', label='Offline Data', markersize=6, markeredgecolor='black', linestyle='None')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02), fontsize=11)

plt.suptitle("Offline RL Analysis: Landscape Truth vs Model Approximation", fontsize=20, y=0.97, fontweight='bold')
plt.savefig("offline_rl_final_with_values.png", dpi=150, bbox_inches='tight')
plt.show()