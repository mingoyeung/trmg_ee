# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
import optuna
from sklearn.preprocessing import StandardScaler
import time
import random
import os
import json
from datetime import datetime

# %%
tf.config.run_functions_eagerly(True)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 添加GPU配置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError:
            pass

set_random_seed(42)
# 验证GPU是否可用
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class PositiveConstraint(tf.keras.constraints.Constraint):

    def __init__(self, min_value=1e-7):
        self.min_value = min_value

    def __call__(self, w):
        w = tf.nn.relu(w)
        w = w + self.min_value
        return w

class McqrnnInputDense(tf.keras.layers.Layer):
    def __init__(self, out_features, activation, penalty=0, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.activation = activation
        self.penalty = penalty

    def build(self, input_shape):
        self.w_inputs = self.add_weight(
            name="w_inputs",
            shape=(input_shape[-1], self.out_features),
            initializer="random_normal",
            regularizer=regularizers.l2(self.penalty),
            trainable=True
        )
        self.w_tau = self.add_weight(
            shape=(1, self.out_features),
            initializer="random_normal",
            constraint=PositiveConstraint(),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.out_features,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs, tau):
        tau_weighted = tf.matmul(tau, self.w_tau)
        inputs_weighted = tf.matmul(inputs, self.w_inputs)
        outputs = tau_weighted + inputs_weighted + self.b
        return self.activation(outputs)


class McqrnnDense(tf.keras.layers.Layer):
    def __init__(self, dense_features, activation, penalty=0, **kwargs):
        super().__init__(**kwargs)
        self.dense_features = dense_features
        self.activation = activation
        self.penalty = penalty

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.dense_features),
            initializer="random_normal",
            regularizer=regularizers.l2(self.penalty),
            constraint=PositiveConstraint(),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.dense_features,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return self.activation(outputs)

class McqrnnOutputDense(tf.keras.layers.Layer):

    def __init__(self, penalty=0, **kwargs):
        super().__init__(**kwargs)
        self.penalty = penalty

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            regularizer=regularizers.l2(self.penalty),
            constraint=PositiveConstraint(),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs

class Mcqrnn(tf.keras.Model):
    def __init__(self, n_hidden, n_hidden2, penalty=0, activation=tf.nn.sigmoid, **kwargs):
        super().__init__(**kwargs)
        self.input_dense = McqrnnInputDense(n_hidden, activation, penalty)
        self.dense = McqrnnDense(n_hidden2, activation, penalty)
        self.output_dense = McqrnnOutputDense(penalty)

    def call(self, inputs, tau):
        x = self.input_dense(inputs, tau)
        x = self.dense(x)
        return self.output_dense(x)

class TiltedAbsoluteLoss(tf.keras.losses.Loss):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self._tau = tf.cast(tau, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        loss = tf.where(error >= 0, self._tau * error, (self._tau - 1) * error)
        return tf.reduce_mean(loss)

class DataScaler:

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, X, y):

        X_scaled = self.scaler_X.fit_transform(X).astype('float32')
        y_scaled = self.scaler_Y.fit_transform(y.reshape(-1, 1)).astype('float32')
        self.is_fitted = True
        return X_scaled, y_scaled

    def transform(self, X, y):

        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit_transform first.")
        X_scaled = self.scaler_X.transform(X).astype('float32')
        y_scaled = self.scaler_Y.transform(y.reshape(-1, 1)).astype('float32')
        return X_scaled, y_scaled

    def inverse_transform_y(self, y_scaled):

        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet.")
        return self.scaler_Y.inverse_transform(y_scaled)

class DataTransformer:
    def __init__(self, x, taus, y=None):
        self.x = x.astype('float32')
        self.y = y.astype('float32') if y is not None else None
        self.taus = taus.astype('float32')
        self._transform()

    def _transform(self):
        n_taus = len(self.taus)
        n_x = len(self.x)

        self.x_trans = np.repeat(self.x, n_taus, axis=0)
        self.tau_trans = np.tile(self.taus, n_x).reshape((-1, 1))

        if self.y is not None:
            self.y_trans = np.repeat(self.y, n_taus, axis=0).reshape((-1, 1))

    def __call__(self):
        return self.x_trans, self.y_trans, self.tau_trans

@tf.function
def train_step(model, inputs, output, tau, loss_func, optimizer):
    with tf.GradientTape() as tape:
        predicted = model(inputs, tau)
        loss = loss_func(output, predicted)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(x_train, y_train, taus, n_hidden, n_hidden2, penalty,
                epochs=1000, learning_rate=0.01, verbose=False):
    """训练单个模型配置"""
    data_transformer = DataTransformer(x_train, taus, y_train)
    x_trans, y_trans, tau_trans = data_transformer()

    model = Mcqrnn(n_hidden, n_hidden2, penalty)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_func = TiltedAbsoluteLoss(tau_trans)

    for epoch in range(epochs):
        loss = train_step(model, x_trans, y_trans, tau_trans, loss_func, optimizer)

        if verbose and epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

    return model, loss.numpy()

def evaluate_model_denormalized(model, x_val, y_val, taus, scaler):

    test_transformer = DataTransformer(x_val, taus, y_val)
    x_val_trans, y_val_trans, tau_val_trans = test_transformer()

    # 模型预测（归一化空间）
    y_pred_normalized = model(x_val_trans, tau_val_trans)

    # 反归一化
    y_true_denorm = scaler.inverse_transform_y(y_val_trans)
    y_pred_denorm = scaler.inverse_transform_y(y_pred_normalized.numpy())

    # 转换为TensorFlow张量
    y_true_denorm_tf = tf.constant(y_true_denorm, dtype=tf.float32)
    y_pred_denorm_tf = tf.constant(y_pred_denorm, dtype=tf.float32)

    # 计算总体损失（反归一化空间）
    loss_func = TiltedAbsoluteLoss(tau_val_trans)
    total_loss = loss_func(y_true_denorm_tf, y_pred_denorm_tf).numpy()

    # 各分位数的损失
    quantile_losses = {}
    n_samples = len(x_val)

    for i, tau in enumerate(taus):
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples

        y_true_q = y_true_denorm_tf[start_idx:end_idx]
        y_pred_q = y_pred_denorm_tf[start_idx:end_idx]
        tau_q = tau_val_trans[start_idx:end_idx]

        loss_q = TiltedAbsoluteLoss(tau_q)(y_true_q, y_pred_q).numpy()
        quantile_losses[f'tau={tau:.2f}'] = float(loss_q)

    return total_loss, quantile_losses


def objective(trial, x_train, y_train, x_val, y_val, taus, scaler,
              epochs=1000, learning_rate=0.01):

    n_hidden = trial.suggest_int('n_hidden', 1, 10)
    n_hidden2 = trial.suggest_int('n_hidden2', 1, 10)
    penalty = trial.suggest_float('penalty', 1e-4, 1e-1)

    model, train_loss = train_model(
        x_train, y_train, taus, n_hidden, n_hidden2, penalty,
        epochs, learning_rate, verbose=False
    )

    val_loss, _ = evaluate_model_denormalized(model, x_val, y_val, taus, scaler)

    del model
    tf.keras.backend.clear_session()

    return val_loss


def optuna_search_hyperparameters(x_train, y_train, x_val, y_val, taus, scaler,
                                  n_trials=30, epochs=1000, learning_rate=0.01):

    optuna_start_time = time.time()

    # 创建 Optuna study
    study = optuna.create_study(
        direction='minimize',
        study_name='mcqrnn_hyperparameter_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print("开始 Optuna 超参数搜索...")
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, x_val, y_val, taus, scaler,
                                epochs, learning_rate),
        n_trials=n_trials,
        show_progress_bar=True
    )

    optuna_end_time = time.time()
    optuna_duration = optuna_end_time - optuna_start_time

    print("\n" + "=" * 50)
    print("最优超参数:")
    print(f"  n_hidden: {study.best_params['n_hidden']}")
    print(f"  n_hidden2: {study.best_params['n_hidden2']}")
    print(f"  penalty: {study.best_params['penalty']:.6f}")
    print(f"  验证集损失（反归一化后）: {study.best_value:.4f}")
    print(f"  Optuna运行时间: {optuna_duration:.2f} 秒")
    print("=" * 50)

    print("\n使用最优参数重新训练模型...")
    train_start_time = time.time()

    best_model, _ = train_model(
        x_train, y_train, taus,
        study.best_params['n_hidden'],
        study.best_params['n_hidden2'],
        study.best_params['penalty'],
        epochs, learning_rate, verbose=True
    )

    train_end_time = time.time()
    train_duration = train_end_time - train_start_time

    timing_info = {
        'optuna_duration_seconds': optuna_duration,
        'final_train_duration_seconds': train_duration,
        'optuna_duration_formatted': f"{optuna_duration / 60:.2f} minutes",
    }

    return study, best_model, timing_info

def save_results_to_json(study, timing_info, test_metrics, filename='optimization_results.json'):

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_hyperparameters': {
            'n_hidden': int(study.best_params['n_hidden']),
            'n_hidden2': int(study.best_params['n_hidden2']),
            'penalty': float(study.best_params['penalty'])
        },
        'best_validation_loss_denormalized': float(study.best_value),
        'timing': timing_info,
        'test_metrics': test_metrics,
        'optimization_info': {
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n结果已保存到: {filename}")

def split_data(data, horizon, window):
    n, m = data.shape
    X = np.zeros((n - window - horizon, window))
    Y = np.zeros((n - window - horizon, 1))

    for i in range(n - window - horizon):
        start = i
        end = start + window

        X[i, :] = data[start:end, 0]
        Y[i] = data[end + horizon - 1, 0]

    return X, Y

#fujian dataset
#def split_data(data, horizon, window):
#    n, m = data.shape
#    X = np.zeros((n - window - horizon, window))
#    Y = np.zeros((n - window - horizon, 1))
#    EX = np.zeros((n - window - horizon, 2))
#    for i in range(n - window - horizon):
#        start = i
#        end = start + window
#        X[i, :] = data[start:end, 0]
#        Y[i] = data[end + horizon - 1, 0]
#        EX[i, :] = data[end + horizon - 1, 5:6]
#    return X, Y, EX

if __name__ == "__main__":

    set_random_seed(42)

    # 读取数据
    print("加载数据...")
    df = pd.read_csv('/home/a/桌面/Simulation code and data/ermg/emdreconstruction.csv')#emdfujian.csv
    a = np.array(df.values[:, 4])#fujian不适用
    a = np.expand_dims(a, axis=1)#fujian不适用
    #X_arg = np.hstack((X, EX))#fujian dataset
    # 创建时间序列数据集
    print("创建时间序列数据集...")
    X, y = split_data(a, horizon=1, window=96)

    # 数据划分：80% 训练，10% 验证，10% 测试
    n1 = X.shape[0]
    train_X, train_y = X[:round(0.8 * n1)], y[:round(0.8 * n1)]
    val_X, val_y = X[round(0.8 * n1):round(0.9 * n1)], y[round(0.8 * n1):round(0.9 * n1)]
    test_X, test_y = X[round(0.9 * n1):], y[round(0.9 * n1):]
    print(f"训练集: {train_X.shape}, 验证集: {val_X.shape}, 测试集: {test_X.shape}")

    # 数据归一化
    print("数据归一化...")
    scaler = DataScaler()
    x_train, y_train = scaler.fit_transform(train_X, train_y)
    x_val, y_val = scaler.transform(val_X, val_y)
    x_test, y_test = scaler.transform(test_X, test_y)

    EPOCHS = 1000
    LEARNING_RATE = 0.01
    N_TRIALS = 100
    taus = np.array([round(i, 2) for i in np.arange(0.05, 1.0, 0.05)])

    # 运行Optuna超参数搜索
    print(f"\n开始超参数优化 (trials={N_TRIALS}, epochs={EPOCHS})...")
    study, best_model, timing_info = optuna_search_hyperparameters(
        x_train, y_train, x_val, y_val, taus, scaler,
        n_trials=N_TRIALS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    print("\n" + "=" * 50)
    print("验证集评估（反归一化后）:")
    print("=" * 50)
    val_total_loss, val_quantile_losses = evaluate_model_denormalized(
        best_model, x_val, y_val, taus, scaler
    )
    print(f"总体损失: {val_total_loss:.4f}")
    for quantile, loss in val_quantile_losses.items():
        print(f"  {quantile}: {loss:.4f}")

    print("\n" + "=" * 50)
    print("测试集评估（反归一化后）:")
    print("=" * 50)
    test_start_time = time.time()
    test_total_loss, test_quantile_losses = evaluate_model_denormalized(
        best_model, x_test, y_test, taus, scaler
    )
    test_duration = time.time() - test_start_time

    print(f"总体损失: {test_total_loss:.4f}")
    for quantile, loss in test_quantile_losses.items():
        print(f"  {quantile}: {loss:.4f}")
    print(f"测试时间: {test_duration:.4f} 秒")

    # 保存结果到JSON
    timing_info['test_duration_seconds'] = test_duration
    timing_info['test_duration_formatted'] = f"{test_duration:.4f} seconds"

    test_metrics = {
        'total_loss_denormalized': float(test_total_loss),
        'quantile_losses_denormalized': test_quantile_losses
    }

    save_results_to_json(
        study, timing_info, test_metrics,
        filename='ermg5_results.json'
    )

    # 生成并保存预测结果
    print("\n生成预测结果...")
    test_transformer = DataTransformer(x_test, taus, y_test)
    x_test_trans, y_test_trans, tau_test_trans = test_transformer()

    y_pred_normalized = best_model(x_test_trans, tau_test_trans)
    y_pred_denormalized = scaler.inverse_transform_y(y_pred_normalized.numpy())

    pred_df = pd.DataFrame(
        y_pred_denormalized.reshape(len(x_test), len(taus)),
        columns=[f'quantile_{tau:.2f}' for tau in taus]
    )
    pred_df.to_excel('ermg5.xlsx', index=True)
