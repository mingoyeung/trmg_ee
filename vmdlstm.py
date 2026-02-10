import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
import time
import json
from sklearn.preprocessing import StandardScaler
import random
import os
from datetime import datetime

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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class MultiQuantileLoss(keras.losses.Loss):
    """
    多分位数联合损失函数
    同时计算所有分位数的tilted absolute loss
    """

    def __init__(self, tau_list, **kwargs):
        super().__init__(**kwargs)
        # 将tau_list转换为shape (1, n_quantiles)的张量,方便广播
        self.tau_tensor = tf.constant(tau_list, dtype=tf.float32)
        self.tau_tensor = tf.reshape(self.tau_tensor, (1, -1))
        self._one = tf.constant(1.0, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        y_true: shape (batch_size, 1) - 真实值
        y_pred: shape (batch_size, n_quantiles) - 预测的各分位数
        """
        # 扩展y_true维度以匹配y_pred: (batch_size, 1) -> (batch_size, n_quantiles)
        y_true_expanded = tf.tile(y_true, [1, tf.shape(y_pred)[1]])

        # 计算误差
        error = y_true_expanded - y_pred  # shape: (batch_size, n_quantiles)

        # 计算每个分位数的tilted absolute loss
        loss = tf.where(
            error > 0,
            self.tau_tensor * error,
            (self.tau_tensor - self._one) * error
        )

        # 返回所有分位数的平均损失
        return tf.reduce_mean(loss)


def build_lstm_model(input_shape, n_quantiles, n_hidden=32, n_hidden2=32, dropout=0.2):
    """
    构建多输出LSTM分位数神经网络

    参数:
        input_shape: 输入形状 (timesteps, features)
        n_quantiles: 分位数数量(输出维度)
        n_hidden: 第一LSTM层单元数
        n_hidden2: 第二LSTM层单元数
        dropout: Dropout比率
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(n_hidden, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(n_hidden2, return_sequences=False),  # 只返回最后一个时间步的输出
        layers.Dropout(dropout),
        layers.Dense(n_quantiles)  # 输出层:同时输出所有分位数
    ])
    return model


class LSTMQRModel:
    """
    多输出LSTM分位数回归神经网络模型包装器
    一次性预测所有分位数

    参数:
        n_hidden: 第一LSTM层单元数
        n_hidden2: 第二LSTM层单元数
        tau_list: 分位数列表
        dropout: Dropout比率
    """

    def __init__(
            self,
            n_hidden=32,
            n_hidden2=32,
            tau_list=[0.1, 0.5, 0.9],
            dropout=0.2
    ):
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.tau_list = tau_list
        self.dropout = dropout

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
            early_stopping_patience=50
    ):
        """训练多输出模型"""
        # 将X reshape为3D: (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        else:
            X_train_reshaped = X_train

        # 数据标准化 - 对于LSTM,我们需要在reshape前标准化
        n_samples, n_timesteps, n_features = X_train_reshaped.shape
        X_2d = X_train_reshaped.reshape(-1, n_timesteps * n_features)
        X_scaled_2d = self.scaler_x.fit_transform(X_2d)
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))  # shape: (n_samples, 1)

        # 准备验证数据
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

            X_val_2d = X_val.reshape(-1, n_timesteps * n_features)
            X_val_scaled_2d = self.scaler_x.transform(X_val_2d)
            X_val_scaled = X_val_scaled_2d.reshape(X_val.shape[0], n_timesteps, n_features)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
            val_data = (X_val_scaled, y_val_scaled)

        if verbose:
            print(f"\n训练多输出LSTM模型 (同时预测 {len(self.tau_list)} 个分位数)...")

        # 构建模型
        self.model = build_lstm_model(
            (n_timesteps, n_features),
            len(self.tau_list),  # 输出维度 = 分位数数量
            self.n_hidden,
            self.n_hidden2,
            self.dropout
        )

        # 编译模型 - 使用多分位数损失函数
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=MultiQuantileLoss(tau_list=self.tau_list)
        )

        # 训练回调
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

        # 训练模型
        history = self.model.fit(
            X_scaled,
            y_scaled,  # shape: (n_samples, 1)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data if val_data else None,
            validation_split=0.1 if val_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X_test):
        """预测多个分位数 - 一次性输出所有分位数"""
        # Reshape输入
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        n_samples, n_timesteps, n_features = X_test.shape
        X_test_2d = X_test.reshape(-1, n_timesteps * n_features)
        X_scaled_2d = self.scaler_x.transform(X_test_2d)
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        # 预测并反标准化
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)  # shape: (n_samples, n_quantiles)

        # 对每个分位数分别反标准化
        predictions = {}
        for i, tau in enumerate(self.tau_list):
            y_pred_i = y_pred_scaled[:, i:i + 1]  # 保持2D形状
            y_pred = self.scaler_y.inverse_transform(y_pred_i)
            predictions[tau] = y_pred.flatten()

        return predictions


def evaluate_model_denormalized(model, X_val, y_val):
    """
    评估模型在反归一化数据上的性能
    返回总体损失和各分位数损失
    """
    # 获取预测结果(已反归一化)
    predictions = model.predict(X_val)

    # 计算各分位数的损失
    quantile_losses = {}
    total_loss = 0
    y_val_flat = y_val.flatten()

    for tau, pred in predictions.items():
        error = y_val_flat - pred
        loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
        quantile_losses[f'tau={tau:.2f}'] = float(loss)
        total_loss += loss

    avg_loss = total_loss / len(predictions)

    return avg_loss, quantile_losses


def objective(trial, train_X, train_y, val_X, val_y, tau_list):
    """
    Optuna 目标函数,最小化验证集上的 multi-quantile loss
    """
    # 定义超参数搜索空间
    n_hidden = trial.suggest_int('n_hidden', 1, 10)
    n_hidden2 = trial.suggest_int('n_hidden2', 1, 10)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # 创建模型
    model = LSTMQRModel(
        n_hidden=n_hidden,
        n_hidden2=n_hidden2,
        tau_list=tau_list,
        dropout=dropout
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

    # 在验证集上评估
    avg_val_loss, _ = evaluate_model_denormalized(model, val_X, val_y)

    return avg_val_loss


def optuna_search_hyperparameters(train_X, train_y, val_X, val_y, tau_list,
                                  n_trials=50, epochs=1000, learning_rate=0.01):
    """使用Optuna进行超参数优化"""

    optuna_start_time = time.time()

    print("开始 Optuna 超参数搜索...")

    study = optuna.create_study(
        direction='minimize',
        study_name='lstm_qrnn_hyperparam_opt',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 使用lambda传递数据
    study.optimize(
        lambda trial: objective(trial, train_X, train_y, val_X, val_y, tau_list),
        n_trials=n_trials,
        show_progress_bar=True
    )

    optuna_end_time = time.time()
    optuna_duration = optuna_end_time - optuna_start_time

    print("\n" + "=" * 50)
    print("最优超参数:")
    print(f"  n_hidden: {study.best_params['n_hidden']}")
    print(f"  n_hidden2: {study.best_params['n_hidden2']}")
    print(f"  dropout: {study.best_params['dropout']:.2f}")
    print(f"  验证集损失(反归一化后): {study.best_value:.4f}")
    print(f"  Optuna运行时间: {optuna_duration:.2f} 秒")
    print("=" * 50)

    print("\n使用最优参数重新训练模型...")
    train_start_time = time.time()

    best_model = LSTMQRModel(
        n_hidden=study.best_params['n_hidden'],
        n_hidden2=study.best_params['n_hidden2'],
        tau_list=tau_list,
        dropout=study.best_params['dropout']
    )

    history = best_model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=512,
        learning_rate=learning_rate,
        validation_data=(val_X, val_y),
        verbose=1,
        early_stopping_patience=30
    )

    train_end_time = time.time()
    train_duration = train_end_time - train_start_time

    timing_info = {
        'optuna_duration_seconds': optuna_duration,
        'final_train_duration_seconds': train_duration,
        'optuna_duration_formatted': f"{optuna_duration / 60:.2f} minutes",
    }

    return study, best_model, timing_info


def save_results_to_json(study, timing_info, test_metrics, filename='qrlstm_optimization_results.json'):
    """保存优化结果到JSON文件"""

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_hyperparameters': {
            'n_hidden': int(study.best_params['n_hidden']),
            'n_hidden2': int(study.best_params['n_hidden2']),
            'dropout': float(study.best_params['dropout'])
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
    """创建时间序列数据集"""
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
    df = pd.read_csv('/home/a/桌面/Simulation code and data/vrlg/vmd_reconstruction.csv')#vmd_decomposition_fujian.csv
    a = np.array(df.values[:, 4])#fujian不适用
    a = np.expand_dims(a, axis=1)#fujian不适用
    # X_arg = np.hstack((X, EX))#fujian dataset
    # 创建时间序列数据集
    print("创建时间序列数据集...")
    X, Y = split_data(a, horizon=1, window=96)

    # 数据划分: 80% 训练, 10% 验证, 10% 测试
    n1 = X.shape[0]
    train_X, train_y = X[:round(0.8 * n1)], Y[:round(0.8 * n1)]
    val_X, val_y = X[round(0.8 * n1):round(0.9 * n1)], Y[round(0.8 * n1):round(0.9 * n1)]
    test_X, test_y = X[round(0.9 * n1):], Y[round(0.9 * n1):]
    print(f"训练集: {train_X.shape}, 验证集: {val_X.shape}, 测试集: {test_X.shape}")

    # 定义超参数
    EPOCHS = 1000
    LEARNING_RATE = 0.01
    N_TRIALS = 100
    tau_list = [round(i, 2) for i in np.arange(0.05, 1.0, 0.05)]

    # 运行Optuna超参数搜索
    print(f"\n开始超参数优化 (trials={N_TRIALS}, epochs={EPOCHS})...")
    study, best_model, timing_info = optuna_search_hyperparameters(
        train_X, train_y, val_X, val_y, tau_list,
        n_trials=N_TRIALS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # 验证集评估
    print("\n" + "=" * 50)
    print("验证集评估(反归一化后):")
    print("=" * 50)
    val_total_loss, val_quantile_losses = evaluate_model_denormalized(
        best_model, val_X, val_y
    )
    print(f"总体损失: {val_total_loss:.4f}")
    for quantile, loss in val_quantile_losses.items():
        print(f"  {quantile}: {loss:.4f}")

    # 测试集评估
    print("\n" + "=" * 50)
    print("测试集评估(反归一化后):")
    print("=" * 50)
    test_start_time = time.time()
    test_total_loss, test_quantile_losses = evaluate_model_denormalized(
        best_model, test_X, test_y
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
        filename='qrlstm5_results.json'
    )

    # 生成并保存预测结果
    print("\n生成预测结果...")
    predictions = best_model.predict(test_X)

    # 整理预测结果为DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df.columns = [f'quantile_{tau:.2f}' for tau in pred_df.columns]
    pred_df.to_excel('vmdlstm5.xlsx', index=True)