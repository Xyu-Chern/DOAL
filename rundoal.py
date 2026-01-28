
import jax
import jax.numpy as jnp
import wandb
import numpy as np
from doal import DOAL
import time

wandb.init(
    project="doal-training-fixed",
    config={"env": "Pendulum-v1"},
)

print("=" * 50)
print("开始训练DOAL算法")
print("=" * 50)

# 保存原始的train方法
original_train = DOAL.train

def train_with_wandb_fixed(self, rng=None, train_state=None):
    """修复版本的WandB训练方法"""
    if train_state is None and rng is None:
        raise ValueError("Either train_state or rng must be provided")

    ts = train_state or self.init_state(rng)
    
    print(f"初始化完成，总步数: {self.total_timesteps}")
    print(f"评估频率: {self.eval_freq}")
    print(f"并行环境数: {self.num_envs}")

    # 首先记录初始评估（在步骤0）
    if not self.skip_initial_evaluation:
        initial_evaluation = self.eval_callback(self, ts, ts.rng)
        if initial_evaluation is not None and len(initial_evaluation) >= 2:
            returns, lengths = initial_evaluation[0], initial_evaluation[1]
            if returns.size > 0:
                wandb.log({
                    "eval/return": float(jnp.mean(returns)),
                    "eval/length": float(jnp.mean(lengths)),
                    "train/step": 0
                }, step=0)
                print(f"[Step 0] 初始评估: Return={float(jnp.mean(returns)):.2f}")

    # 计算实际的迭代次数
    total_eval_cycles = int(np.ceil(self.total_timesteps / self.eval_freq))
    print(f"总共需要 {total_eval_cycles} 个评估周期")
    
    # 收集所有结果
    all_evaluations = []
    
    # 手动循环而不是使用scan
    for cycle in range(total_eval_cycles):
        start_time = time.time()
        
        # 执行一个评估周期的训练
        steps_per_train_it = self.num_envs * self.policy_delay
        num_train_its = int(np.ceil(self.eval_freq / steps_per_train_it))
        
        print(f"[Cycle {cycle+1}/{total_eval_cycles}] 需要 {num_train_its} 次训练迭代")
        
        # 执行训练迭代
        for i in range(num_train_its):
            ts = self.train_iteration(ts)
            if i % max(1, num_train_its // 10) == 0:  # 每10%打印一次
                print(f"  训练迭代 {i+1}/{num_train_its}, 当前步数: {ts.global_step}")

        # 评估
        eval_result = self.eval_callback(self, ts, ts.rng)
        all_evaluations.append(eval_result)
        
        # 记录到WandB
        if eval_result is not None and len(eval_result) >= 2:
            returns, lengths = eval_result[0], eval_result[1]
            if returns.size > 0:
                step = int(ts.global_step)
                mean_return = float(jnp.mean(returns))
                mean_length = float(jnp.mean(lengths))
                
                wandb.log({
                    "eval/return": mean_return,
                    "eval/length": mean_length,
                    "train/step": step,
                    "cycle": cycle,
                }, step=step)
                
                cycle_time = time.time() - start_time
                print(f"[Cycle {cycle+1}] Step {step}: Return={mean_return:.2f}, Time={cycle_time:.1f}s")
    
    print(f"训练完成！最终步数: {ts.global_step}")
    
    # 转换评估结果为jax数组格式
    if all_evaluations:
        # 将列表转换为jax数组
        evaluation = jax.tree.map(
            lambda *args: jnp.stack(args),
            *all_evaluations
        )
        
        if not self.skip_initial_evaluation and initial_evaluation is not None:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )
    else:
        evaluation = None

    return ts, evaluation

# 替换train方法
DOAL.train = train_with_wandb_fixed

# 创建并训练算法（使用更合理的参数）
algo = DOAL.create(
    env="Pendulum-v1",
    total_timesteps=10000,  # 先用小步数测试
    eval_freq=1000,         # 每1000步评估
    num_envs=1,
    learning_rate=0.001,
    batch_size=32,          # 小批量
    fill_buffer=100,        # 小的填充缓冲区
    flow_steps=10,
    max_q_samples=4,
    policy_delay=2,
    alpha=0.2,
    delta=2.0,
    exploration_noise=0.1,
    target_noise=0.2,
    target_noise_clip=0.5,
)

print("算法配置:")
print(f"  环境: Pendulum-v1")
print(f"  总步数: {algo.total_timesteps}")
print(f"  评估频率: {algo.eval_freq}")

rng = jax.random.PRNGKey(42)
print("\n开始训练...")
start_time = time.time()

try:
    train_state, evaluation = algo.train(rng=rng)
    
    total_time = time.time() - start_time
    print(f"\n训练完成！总用时: {total_time:.1f}秒")
    
    if evaluation is not None:
        print(f"评估结果: {evaluation}")
        
finally:
    wandb.finish()
    print("WandB会话结束")