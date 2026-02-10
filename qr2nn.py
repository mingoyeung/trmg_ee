#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
import time
from sklearn.preprocessing import StandardScaler
import random
import os
import json

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
#%%
class MultiQuantileLoss(keras.losses.Loss):


    def __init__(self, tau_list, **kwargs):
        super().__init__(**kwargs)

        self.tau_tensor = tf.constant(tau_list, dtype=tf.float32)
        self.tau_tensor = tf.reshape(self.tau_tensor, (1, -1))
        self._one = tf.constant(1.0, dtype=tf.float32)

    def call(self, y_true, y_pred):

        y_true_expanded = tf.tile(y_true, [1, tf.shape(y_pred)[1]])

        error = y_true_expanded - y_pred  # shape: (batch_size, n_quantiles)

        loss = tf.where(
            error > 0,
            self.tau_tensor * error,
            (self.tau_tensor - self._one) * error
        )

        return tf.reduce_mean(loss)
#%%
def build_qrnn_model(input_dim, n_quantiles, n_hidden=15, n_hidden2=15, activation='sigmoid', penalty=0.0):

    regularizer = keras.regularizers.l2(penalty) if penalty > 0 else None

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(n_hidden, activation=activation, kernel_regularizer=regularizer),
        layers.Dense(n_hidden2, activation=activation, kernel_regularizer=regularizer),
        layers.Dense(n_quantiles, kernel_regularizer=regularizer)
    ])
    return model


class QRNNModel:

    def __init__(
            self,
            n_hidden=15,
            n_hidden2=15,
            tau_list=[0.1, 0.5, 0.9],
            penalty=0.0,
            activation='sigmoid'
    ):
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.tau_list = tau_list
        self.penalty = penalty
        self.activation = activation

        self.model = None  # 单个多输出模型
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(
            self,
            X_train,
            y_train,
            epochs=500,
            batch_size=32,
            learning_rate=0.01,
            validation_data=None,
            verbose=1,
            early_stopping_patience=30
    ):
        # 数据标准化
        X_scaled = self.scaler_x.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))  # shape: (n_samples, 1)

        # 准备验证数据
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler_x.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
            val_data = (X_val_scaled, y_val_scaled)

        if verbose:
            print(f"\n训练多输出模型 (同时预测 {len(self.tau_list)} 个分位数)...")

        self.model = build_qrnn_model(
            X_train.shape[1],
            len(self.tau_list),  # 输出维度 = 分位数数量
            self.n_hidden,
            self.n_hidden2,
            self.activation,
            self.penalty
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=MultiQuantileLoss(tau_list=self.tau_list)
        )

        callbacks = []
        if early_stopping_patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_data else 'loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                )
            )

        history = self.model.fit(
            X_scaled,
            y_scaled,  # shape: (n_samples, 1)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data if val_data else None,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X_test):

        X_scaled = self.scaler_x.transform(X_test)

        y_pred_scaled = self.model.predict(X_scaled, verbose=0)  # shape: (n_samples, n_quantiles)

        # 对每个分位数分别反标准化
        predictions = {}
        for i, tau in enumerate(self.tau_list):
            y_pred_i = y_pred_scaled[:, i:i + 1]  # 保持2D形状
            y_pred = self.scaler_y.inverse_transform(y_pred_i)
            predictions[tau] = y_pred.flatten()

        return predictions


#%%
def objective(trial, train_X, train_y, val_X, val_y):
    """
    Optuna 目标函数,最小化验证集上的 multi-quantile loss
    """
    # 定义超参数搜索空间
    n_hidden = trial.suggest_int('n_hidden', 1, 10)
    n_hidden2 = trial.suggest_int('n_hidden2', 1, 10)
    penalty = (trial.suggest_float('penalty', 1e-4, 1e-1))

    # tau列表保持不变
    tau_list = [round(i, 2) for i in np.arange(0.05, 1.0, 0.05).tolist()]

    # 创建模型
    model = QRNNModel(
        n_hidden=n_hidden,
        n_hidden2=n_hidden2,
        tau_list=tau_list,
        penalty=penalty
    )

    # 训练模型
    history = model.fit(
        train_X,
        train_y,
        epochs=200,
        batch_size=64,
        learning_rate=0.01,
        validation_data=(val_X, val_y),
        verbose=0,
        early_stopping_patience=15
    )

    # 在验证集上预测
    val_predictions = model.predict(val_X)

    # 计算验证损失
    total_val_loss = 0
    val_y_flat = val_y.flatten()

    for tau, pred in val_predictions.items():
        error = val_y_flat - pred
        loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
        total_val_loss += loss

    avg_val_loss = total_val_loss / len(tau_list)

    return avg_val_loss
#%%
def optimize_hyperparameters(train_X, train_y, val_X, val_y, n_trials=50):
    """使用Optuna进行超参数优化"""
    print("开始超参数优化...")

    study = optuna.create_study(
        direction='minimize',
        study_name='qrnn_hyperparam_opt',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 使用lambda传递数据
    study.optimize(
        lambda trial: objective(trial, train_X, train_y, val_X, val_y),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n优化完成!")
    print(f"最佳试验编号: {study.best_trial.number}")
    print(f"最佳验证损失: {study.best_trial.value:.6f}")
    print(f"最佳超参数: {study.best_trial.params}")

    return study
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
    df = pd.read_csv('/home/a/桌面/Simulation code and data/data/time_series_15min_cleaned.csv')#res_new.csv
    a = np.array(df.values[:, 2])#fujian不适用
    a = np.expand_dims(a, axis=1)#fujian不适用
    #X_arg = np.hstack((X, EX))#fujian dataset
    # 创建时间序列数据集
    print("创建时间序列数据集...")
    X, Y = split_data(a, horizon=1, window=96)

    # 数据划分:80% 训练,10% 验证,10% 测试
    n1 = X.shape[0]
    train_X, train_y = X[:round(0.8 * n1)], Y[:round(0.8 * n1)]
    val_X, val_y = X[round(0.8 * n1):round(0.9 * n1)], Y[round(0.8 * n1):round(0.9 * n1)]
    test_X, test_y = X[round(0.9 * n1):], Y[round(0.9 * n1):]
    print(f"训练集: {train_X.shape}, 验证集: {val_X.shape}, 测试集: {test_X.shape}")

    EPOCHS = 1000
    LEARNING_RATE = 0.01
    N_TRIALS = 100
    tau_list = [round(i, 2) for i in np.arange(0.05, 1.0, 0.05)]

    # 运行Optuna超参数搜索
    print(f"\n开始超参数优化 (trials={N_TRIALS}, epochs={EPOCHS})...")
    study = optimize_hyperparameters(train_X, train_y, val_X, val_y, n_trials=N_TRIALS)

    print("\n" + "=" * 50)
    print("最优超参数:")
    print(f"  n_hidden: {study.best_params['n_hidden']}")
    print(f"  n_hidden2: {study.best_params['n_hidden2']}")
    print(f"  penalty: {study.best_params['penalty']:.6f}")
    print(f"  验证集损失: {study.best_value:.4f}")
    print("=" * 50)

    print("\n使用最佳超参数重新训练模型...")
    train_start_time = time.time()

    best_params = study.best_trial.params
    final_model = QRNNModel(
        n_hidden=best_params['n_hidden'],
        n_hidden2=best_params['n_hidden2'],
        tau_list=tau_list,
        penalty=best_params['penalty']
    )

    history = final_model.fit(
        train_X,
        train_y,
        epochs=EPOCHS,
        batch_size=512,
        learning_rate=LEARNING_RATE,
        validation_data=(val_X, val_y),
        verbose=0,
        early_stopping_patience=30
    )

    train_duration = time.time() - train_start_time
    print(f"\n训练耗时: {train_duration:.3f}秒")

    # 验证集评估
    print("\n" + "=" * 50)
    print("验证集评估:")
    print("=" * 50)
    val_predictions = final_model.predict(val_X)
    val_total_loss = 0
    val_quantile_losses = {}

    for tau, pred in val_predictions.items():
        error = val_y.flatten() - pred
        loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
        val_total_loss += loss
        val_quantile_losses[f'quantile_{tau:.2f}'] = float(loss)
        print(f"  分位数 {tau:.2f}: {loss:.4f}")

    print(f"总体损失: {val_total_loss / len(val_predictions):.4f}")

    # 测试集评估
    print("\n" + "=" * 50)
    print("测试集评估:")
    print("=" * 50)
    test_start_time = time.time()
    test_predictions = final_model.predict(test_X)
    test_duration = time.time() - test_start_time

    test_total_loss = 0
    test_quantile_losses = {}

    for tau, pred in test_predictions.items():
        error = test_y.flatten() - pred
        loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
        test_total_loss += loss
        test_quantile_losses[f'quantile_{tau:.2f}'] = float(loss)
        print(f"  分位数 {tau:.2f}: {loss:.4f}")

    print(f"总体损失: {test_total_loss / len(test_predictions):.4f}")
    print(f"测试时间: {test_duration:.4f} 秒")

    # 保存结果到JSON
    timing_info = {
        'train_duration_seconds': train_duration,
        'test_duration_seconds': test_duration,
        'train_duration_formatted': f"{train_duration:.3f} seconds",
        'test_duration_formatted': f"{test_duration:.4f} seconds"
    }

    test_metrics = {
        'total_loss': float(test_total_loss / len(test_predictions)),
        'quantile_losses': test_quantile_losses
    }

    results = {
        'best_hyperparameters': {
            'n_hidden': int(best_params['n_hidden']),
            'n_hidden2': int(best_params['n_hidden2']),
            'penalty': float(best_params['penalty'])
        },
        'best_validation_loss': float(study.best_value),
        'timing': timing_info,
        'test_metrics': test_metrics,
        'optimization_info': {
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
    }

    with open('qrnn2_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n结果已保存到: qrnn2_results.json")

    # 生成并保存预测结果
    print("\n生成预测结果...")
    pred_df = pd.DataFrame(test_predictions)
    pred_df.columns = [f'quantile_{tau:.2f}' for tau in pred_df.columns]
    pred_df.to_excel('qrnn2pred.xlsx', index=True)
    print("预测结果已保存到: qrnn2pred.xlsx")