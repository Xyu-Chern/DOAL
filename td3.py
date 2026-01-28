
import jax
import wandb
import argparse
import json
import time
import numpy as np
from rejax import TD3
from rejax.algos import get_algo
from rejax.evaluate import make_evaluate as make_evaluate_vanilla

# ========== 配置参数 ==========
config = {
    "env": "Pendulum-v1",
    "env_params": {},
    "total_timesteps": 100_000,
    "eval_freq": 10_000,
    "num_envs": 1,
    "learning_rate": 0.001,
    "batch_size": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 10,
    "exploration_noise": 0.1,
    "target_noise": 0.2,
    "target_noise_clip": 0.5,
    "policy_delay": 2,
    "actor_kwargs": {
        "activation": "relu",
        "hidden_layer_sizes": (256, 256),
    },
    "critic_kwargs": {
        "activation": "relu", 
        "hidden_layer_sizes": (256, 256),
    },
}

# ========== 初始化WandB ==========
wandb.init(
    project="td3-rejax-logger",
    config=config,
    name="td3-pendulum",
)

print(f"WandB运行中: {wandb.run.url}")

# ========== 简化版的Logger类 ==========
class SimpleLogger:
    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb
        self.last_step = 0
        self._log_step = []
        self.timer = time.process_time()
        self.last_time = self.timer
        
    def log(self, data, step):
        """记录数据到WandB和本地"""
        step = step.item() if hasattr(step, 'item') else step
        
        # 记录到本地
        self._log_step.append({**data, 'step': step})
        
        # 记录到WandB
        if self.use_wandb and step > self.last_step:
            # 计算平均值（如果有多个种子）
            if len(self._log_step) > 0:
                wandb_data = {}
                for key in data.keys():
                    values = [entry.get(key) for entry in self._log_step if key in entry]
                    if values:
                        wandb_data[f"mean/{key}"] = np.mean(values)
                        wandb_data[f"std/{key}"] = np.std(values) if len(values) > 1 else 0.0
                
                wandb.log(wandb_data, step=step)
            
            self._log_step = []  # 清空缓冲区
            self.last_step = step
    
    def save_logs(self, filename="td3_logs.json"):
        """保存日志到文件"""
        with open(filename, 'w') as f:
            json.dump(self._log_step, f, indent=2)
        wandb.save(filename)
        print(f"日志已保存到 {filename}")

# ========== 创建评估回调 ==========
def make_evaluate_with_logger(logger, env, env_params, num_eval_episodes=5):
    """创建带有日志记录的评估函数"""
    evaluate_vanilla = make_evaluate_vanilla(env, env_params, num_eval_episodes)
    
    def evaluate(config, ts, rng):
        lengths, returns = evaluate_vanilla(config, ts, rng)
        
        # 记录到logger
        logger.log({
            "episode_length": lengths.mean(axis=0),
            "episode_length_std": lengths.std(axis=0),
            "return": returns.mean(axis=0),
            "return_std": returns.std(axis=0),
        }, ts.global_step)
        
        # 打印到控制台
        step = int(ts.global_step)
        print(f"[Step {step}] Eval: return={returns.mean():.2f}±{returns.std():.2f}, length={lengths.mean():.1f}")
        
        return lengths, returns
    
    return evaluate

# ========== 主训练函数 ==========
def train_td3_with_logger(num_seeds=3):
    """使用Logger训练多个TD3智能体"""
    print(f"训练 {num_seeds} 个TD3智能体...")
    
    # 初始化Logger
    logger = SimpleLogger(use_wandb=True)
    
    # 创建TD3算法配置
    algo_class, config_class = get_algo("td3")
    train_config = config_class.create(**config)
    
    # 创建评估回调
    evaluate = make_evaluate_with_logger(logger, train_config.env, train_config.env_params)
    train_config = train_config.replace(eval_callback=evaluate)
    
    # 准备随机种子
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_seeds)
    
    # 创建vmap训练函数
    vmap_train = jax.jit(jax.vmap(algo_class.train, in_axes=(None, 0)))
    
    print("编译训练函数...")
    start_time = time.time()
    
    # 编译
    lowered = vmap_train.lower(train_config, keys)
    compile_time = time.time() - start_time
    compiled = lowered.compile()
    
    print(f"编译完成，用时 {compile_time:.1f}秒")
    
    # 训练
    print("开始训练...")
    train_start = time.time()
    train_state, evaluation = compiled(train_config, keys)
    train_time = time.time() - train_start
    
    print(f"训练完成！用时 {train_time:.1f}秒")
    
    # 保存日志
    logger.save_logs()
    
    # 记录最终结果
    if evaluation is not None and len(evaluation) >= 2:
        returns_batch, lengths_batch = evaluation[0], evaluation[1]
        
        final_returns = []
        for i in range(num_seeds):
            returns = returns_batch[i] if len(returns_batch) > 1 else returns_batch
            final_return = float(np.mean(returns))
            final_returns.append(final_return)
            
            wandb.log({f"final/seed_{i}": final_return})
            print(f"种子 {i} 最终回报: {final_return:.2f}")
        
        # 记录统计信息
        stats = {
            "final/mean_return": np.mean(final_returns),
            "final/std_return": np.std(final_returns),
            "final/best_return": np.max(final_returns),
            "final/worst_return": np.min(final_returns),
        }
        wandb.log(stats)
        wandb.summary.update(stats)
        
        print(f"\n最终统计:")
        print(f"  平均回报: {stats['final/mean_return']:.2f} ± {stats['final/std_return']:.2f}")
        print(f"  最佳回报: {stats['final/best_return']:.2f}")
        print(f"  最差回报: {stats['final/worst_return']:.2f}")
    
    return train_state, evaluation

# ========== 执行训练 ==========
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    args = parser.parse_args()
    
    # 更新配置
    config["env"] = args.env
    config["total_timesteps"] = args.total_timesteps
    config["eval_freq"] = args.eval_freq
    
    print("=" * 50)
    print(f"TD3训练配置:")
    print(f"  环境: {config['env']}")
    print(f"  总步数: {config['total_timesteps']}")
    print(f"  评估频率: {config['eval_freq']}")
    print(f"  种子数量: {args.num_seeds}")
    print("=" * 50)
    
    try:
        train_state, evaluation = train_td3_with_logger(num_seeds=args.num_seeds)
        print("\n训练成功完成！")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()
        print("WandB会话已结束")