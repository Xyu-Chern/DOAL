
import jax
import jax.numpy as jnp
import wandb
from doal import DOAL

# 初始化WandB（在线模式）
wandb.init(
    project="my-daal-project", 
    config={"env": "Pendulum-v1"},
    # 添加重试设置
    settings=wandb.Settings(
        start_method="thread",
        _disable_stats=True,
        _disable_meta=True,
        _disable_viewer=True
    )
)

print(f"WandB在线运行: {wandb.run.url if wandb.run else '离线'}")

# ==================== 修复回调函数 ====================
# 保存原始评估回调
original_eval_callback = DOAL.eval_callback

def safe_sync_callback(algo, ts, rng):
    """安全的同步回调函数，完全避免异步问题"""
    try:
        # 运行原始评估
        if original_eval_callback is None:
            # 如果没有回调，返回空结果
            class EmptyResult:
                episode_returns = jnp.array([])
                episode_lengths = jnp.array([])
            return EmptyResult()
        
        result = original_eval_callback(algo, ts, rng)
        
        # 同步记录到WandB - 直接调用，不使用任何异步
        try:
            step = 0
            if hasattr(ts, 'global_step'):
                if hasattr(ts.global_step, 'item'):
                    step = int(ts.global_step.item())
                elif isinstance(ts.global_step, (int, float)):
                    step = int(ts.global_step)
            
            log_data = {"train/step": step}
            
            # 检查结果并记录
            if hasattr(result, 'episode_returns') and result.episode_returns.size > 0:
                returns = result.episode_returns
                avg_return = float(jnp.mean(returns))
                log_data["eval/return_mean"] = avg_return
                log_data["eval/return_std"] = float(jnp.std(returns))
                
                # 打印到控制台
                print(f"Step {step}: Return = {avg_return:.2f} ± {float(jnp.std(returns)):.2f}")
            
            if hasattr(result, 'episode_lengths') and result.episode_lengths.size > 0:
                lengths = result.episode_lengths
                log_data["eval/length_mean"] = float(jnp.mean(lengths))
            
            # 同步记录 - 这是关键，完全同步
            wandb.log(log_data, step=step)
            
        except Exception as log_error:
            print(f"WandB日志错误（非致命）: {log_error}")
            # 本地备份
            try:
                with open("wandb_backup.log", "a") as f:
                    f.write(f"Step {step}: {log_data}, Error: {log_error}\n")
            except:
                pass
        
        return result
        
    except Exception as e:
        print(f"回调函数错误: {e}")
        # 返回一个空结果以避免中断训练
        class EmptyResult:
            episode_returns = jnp.array([])
            episode_lengths = jnp.array([])
        return EmptyResult()

# 替换DOAL的回调函数
DOAL.eval_callback = safe_sync_callback

# ==================== 创建并训练算法 ====================
print("创建DOAL算法...")
try:
    algo = DOAL.create(
        env="Pendulum-v1",
        total_timesteps=1000,  # 增加一点步数以看到更多日志
        eval_freq=100,  # 每100步评估一次
        num_envs=1,
        learning_rate=0.001,
        batch_size=32,
        gamma=0.99,
        max_grad_norm=1.0,
        fill_buffer=100,
        flow_steps=10,
        max_q_samples=4,
        policy_delay=2,
        alpha=0.2,
        delta=2.0,
        exploration_noise=0.1,
        target_noise=0.2,
        target_noise_clip=0.5,
    )
    
    print("开始训练...")
    rng = jax.random.PRNGKey(42)
    train_state, eval_result = algo.train(rng=rng)
    
    print("训练完成！")
    
    # 记录最终结果
    if eval_result is not None and hasattr(eval_result, 'episode_returns'):
        final_return = float(jnp.mean(eval_result.episode_returns))
        wandb.log({"final/return": final_return})
        print(f"最终评估回报: {final_return:.2f}")
    
except Exception as e:
    print(f"训练过程中出错: {e}")
    import traceback
    traceback.print_exc()

# ==================== 确保WandB正确结束 ====================
try:
    wandb.finish()
    print("WandB会话已结束")
except Exception as e:
    print(f"WandB结束错误: {e}")

print("程序执行完毕！")