import os
import numpy as np
import ogbench
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

# Optional: make BLAS use fewer threads to avoid oversubscription
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("MKL_NUM_THREADS", "1")

rng = np.random.default_rng(seed=42)

# Config
MAX_TRAIN_SAMPLES = 5000          # cap per environment for mean model
MAX_VARIANCE_SAMPLES = 5000       # cap for variance model
PROBE_FOR_GAMMA = 5000             # subset size to estimate median distance
RBF_ALPHA = 1e-1                   # ridge
USE_STANDARDIZATION = True         # standardize observations (and actions for stability)
DATASET_DIR = "/home/bml/storage/.ogbench/data"

env_names = [
    'antmaze-large-navigate-singletask-task1-v0',
    'antmaze-giant-navigate-singletask-task1-v0',
    'humanoidmaze-medium-navigate-singletask-task1-v0',
    'humanoidmaze-large-navigate-singletask-task1-v0',
    'antsoccer-arena-navigate-singletask-task4-v0',
    'cube-single-play-singletask-task2-v0',
    'cube-double-play-singletask-task2-v0',
    'scene-play-singletask-task2-v0',
    'puzzle-3x3-play-singletask-task4-v0',
    'puzzle-4x4-play-singletask-task4-v0'
]

def to_float32(x):
    return np.asarray(x, dtype=np.float32, order='C')

def subsample_indices(n, k, rng):
    k = min(k, n)
    if k == n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)

def median_pairwise_dist(X, probe=PROBE_FOR_GAMMA, rng=rng):
    # Use a small random subset to estimate median L2 distance
    n = X.shape[0]
    idx = subsample_indices(n, min(probe, n), rng)
    Xp = X[idx]
    # Compute pairwise distances efficiently in blocks if needed
    # For moderate probe sizes, a full pairwise is fine
    diffs = Xp[:, None, :] - Xp[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    # exclude zeros on diagonal
    dists = dists[np.triu_indices_from(dists, k=1)]
    med = np.median(dists)
    # Fallback in degenerate cases
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return med

print("环境名称及其action_dim:")
print("=" * 50)

for env_name in env_names:
    try:
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            env_name, dataset_dir=DATASET_DIR
        )

        actions = to_float32(train_dataset["actions"])
        observations = to_float32(train_dataset["observations"])
        action_dim = actions.shape[-1]

        print(f"{env_name}: actions {actions.shape}")
        print(f"{env_name}: observations {observations.shape}")

        n_train = observations.shape[0]

        # Standardize features (and optionally targets for stability)
        obs_scaler = StandardScaler(with_mean=True, with_std=True) if USE_STANDARDIZATION else None
        act_scaler = StandardScaler(with_mean=True, with_std=True) if USE_STANDARDIZATION else None

        if obs_scaler is not None:
            observations_std = obs_scaler.fit_transform(observations)
        else:
            observations_std = observations

        if act_scaler is not None:
            actions_std = act_scaler.fit_transform(actions)
        else:
            actions_std = actions

        # Subsample for mean model
        idx_mean = subsample_indices(n_train, MAX_TRAIN_SAMPLES, rng)
        X_mean = observations_std[idx_mean]
        Y_mean = actions_std[idx_mean]

        # Robust gamma via median heuristic on standardized features
        med = median_pairwise_dist(X_mean, probe=PROBE_FOR_GAMMA, rng=rng)
        # RBF gamma ~ 1/(2*sigma^2); sigma ≈ median distance
        gamma = 1.0 / (2.0 * (med ** 2) + 1e-12)

        model_mean = KernelRidge(kernel='rbf', gamma=gamma, alpha=RBF_ALPHA)
        model_mean.fit(X_mean, Y_mean)

        # Predict mean on full training to compute residuals efficiently in chunks
        # to limit memory if needed
        def batched_predict(model, X, batch=50000):
            n = X.shape[0]
            out = np.empty((n, action_dim), dtype=np.float64)  # sklearn returns float64
            for s in range(0, n, batch):
                e = min(n, s + batch)
                out[s:e] = model.predict(X[s:e])
            return out

        pred_mean_train_std = batched_predict(model_mean, observations_std)
        # Residuals in standardized space
        residuals_std = actions_std - pred_mean_train_std
        squared_residuals_std = residuals_std ** 2

        # Subsample for variance model
        idx_var = subsample_indices(n_train, MAX_VARIANCE_SAMPLES, rng)
        X_var = observations_std[idx_var]
        Y_var = squared_residuals_std[idx_var]

        # Use same gamma for variance model (features identical)
        model_variance = KernelRidge(kernel='rbf', gamma=gamma, alpha=RBF_ALPHA)
        model_variance.fit(X_var, Y_var)

        # Prepare test data
        test_observations = to_float32(val_dataset["observations"])
        test_actions = to_float32(val_dataset["actions"])

        if obs_scaler is not None:
            test_observations_std = obs_scaler.transform(test_observations)
        else:
            test_observations_std = test_observations

        # Predict mean and variance on test
        mean_pred_std = batched_predict(model_mean, test_observations_std)
        var_pred_std = batched_predict(model_variance, test_observations_std)
        var_pred_std = np.clip(var_pred_std, a_min=0.0, a_max=None)
        std_dev_pred_std = np.sqrt(var_pred_std + 1e-12)

        # Inverse-transform predictions back to original action space if standardized
        if act_scaler is not None:
            # For mean: inverse scaling
            mean_pred = act_scaler.inverse_transform(mean_pred_std)
            # For variance: if y was standardized per-dimension with scale s,
            # then Var[y] scales by s^2. StandardScaler stores scale_ per feature.
            scales = act_scaler.scale_.astype(np.float64)  # shape [action_dim]
            var_pred = var_pred_std * (scales[None, :] ** 2)
            std_dev_pred = np.sqrt(var_pred + 1e-12)
        else:
            mean_pred = mean_pred_std
            std_dev_pred = std_dev_pred_std

        print(f"{env_name}: std_dev_pred shape {std_dev_pred.shape}, "
              f"mean={np.mean(std_dev_pred):.4f}, median={np.median(std_dev_pred):.4f}, "
              f"max={np.max(std_dev_pred):.4f}")

        env.close()

    except Exception as e:
        print(f"Error processing {env_name}: {e}")