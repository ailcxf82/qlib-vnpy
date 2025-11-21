"""模型训练器"""
from __future__ import annotations

# 必须在所有其他导入之前抑制 joblib 警告
import suppress_joblib_warnings  # noqa: F401

import warnings
# 额外的警告过滤（双重保险）
# 注意：warnings.filterwarnings 的 message 参数只接受字符串，不支持正则表达式
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', category=UserWarning, message='resource_tracker')
warnings.filterwarnings('ignore', category=UserWarning, message='FileNotFoundError')

import pickle
from typing import Optional

import numpy as np
import pandas as pd

from model.lgb_model import LGBModel
from model.metrics import Metrics
from utils.logger import default_logger
from utils.timer import timing_decorator
from utils.tools import ensure_dir


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_config, pipeline_config: Optional[dict] = None):
        self.model_config = model_config
        self.pipeline_config = pipeline_config or {}
        self.model = None
        self.model_type = model_config.get('model_type', 'lightgbm')
        self.training_backend = model_config.get('training_backend', 'native')
        self.qlib_config = model_config.get('qlib', {})
        self.feature_names = None
        self._qlib_dataset = None
    
    def create_model(self):
        """创建模型"""
        if self._use_qlib_backend():
            try:
                from qlib.contrib.model.gbdt import LGBModel as QlibLGBModel
            except ImportError as exc:
                raise ImportError("未安装qlib，请先在当前环境中安装后再使用Qlib训练后端") from exc
            
            qlib_params = self._get_qlib_model_params()
            default_logger.info(f"Qlib模型参数: {qlib_params}")
            self.model = QlibLGBModel(**qlib_params)
        elif self.model_type == 'lightgbm':
            lgb_config = self.model_config.get('lightgbm', {})
            self.model = LGBModel(lgb_config)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return self.model
    
    @timing_decorator
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签
        
        Returns:
            训练好的模型
        """
        default_logger.info(f"开始训练 {self.model_type} 模型")
        default_logger.info(f"训练集大小: {len(X_train)}")
        
        if X_valid is not None:
            default_logger.info(f"验证集大小: {len(X_valid)}")
        
        if self._use_qlib_backend():
            dataset = self._train_with_qlib(X_train, y_train)
            if dataset is not None and 'valid' in dataset.segments:
                self._evaluate_qlib_validation(dataset)
        else:
            # 创建模型
            if self.model is None:
                self.create_model()
            
            # 训练
            self.model.train(X_train, y_train, X_valid, y_valid)
            
            # 评估
            if X_valid is not None and y_valid is not None:
                pred_valid = self.model.predict(X_valid)
                
                if isinstance(y_valid, pd.DataFrame):
                    y_valid_arr = y_valid.values.ravel()
                elif isinstance(y_valid, pd.Series):
                    y_valid_arr = y_valid.values
                else:
                    y_valid_arr = y_valid
                
                metrics = Metrics.evaluate_model(pred_valid, y_valid_arr)
                Metrics.print_metrics(metrics)
        
        # 打印特征重要度
        self.print_feature_importance()
        
        return self.model
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        if self._use_qlib_backend():
            booster = getattr(self.model, 'model', None)
            if booster is None:
                raise ValueError("Qlib模型尚未初始化或加载")
            
            if isinstance(X, pd.DataFrame):
                features = X.values
            else:
                features = X
            
            return booster.predict(features)
        
        return self.model.predict(X)
    
    def save_model(self, model_path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        # 确保目录存在（从文件路径中提取目录）
        import os
        model_dir = os.path.dirname(model_path)
        if model_dir:
            ensure_dir(model_dir)
        
        # 如果文件已存在，先删除（避免文件锁定问题）
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                default_logger.warning(f"删除已存在的模型文件: {model_path}")
            except Exception as e:
                default_logger.warning(f"无法删除已存在的模型文件 {model_path}: {e}")
        
        try:
            if self._use_qlib_backend():
                booster = getattr(self.model, 'model', None)
                if booster is None:
                    raise ValueError("Qlib模型未包含Booster对象，无法保存")
                booster.save_model(model_path)
                default_logger.info(f"Qlib模型已保存: {model_path}")
            elif self.model_type == 'lightgbm':
                self.model.save_model(model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
            
            default_logger.info(f"模型已保存: {model_path}")
        except Exception as e:
            default_logger.error(f"保存模型失败: {model_path}, 错误: {e}")
            raise
    
    def load_model(self, model_path):
        """加载模型"""
        if self._use_qlib_backend():
            import lightgbm as lgb
            if self.model is None:
                self.create_model()
            self.model.model = lgb.Booster(model_file=model_path)
            self.feature_names = self.model.model.feature_name()
        elif self.model_type == 'lightgbm':
            self.model.load_model(model_path)
        else:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        default_logger.info(f"模型已加载: {model_path}")
        
        return self.model

    # --------------------------
    # Qlib相关私有方法
    # --------------------------

    def _use_qlib_backend(self) -> bool:
        return self.training_backend.lower() == 'qlib'

    def _get_qlib_model_params(self) -> dict:
        """构造Qlib LGB模型参数"""
        lgb_config = self.model_config.get('lightgbm', {}).copy()
        num_boost_round = self.qlib_config.get(
            'num_boost_round',
            lgb_config.pop('n_estimators', None)
        )
        if num_boost_round is None:
            num_boost_round = 100
        
        early_stopping = self.qlib_config.get('early_stopping_rounds', 50)
        loss = self.qlib_config.get('loss')
        if loss is None:
            objective = lgb_config.get('objective', 'regression')
            if 'binary' in objective:
                loss = 'binary'
            else:
                loss = 'mse'
        
        qlib_params = {
            'loss': loss,
            'early_stopping_rounds': early_stopping,
            'num_boost_round': num_boost_round,
            'max_bin' : '255',
            'min_data_in_leaf' : '30',
            'lambda_l2' : '2',
        }
        qlib_params.update(lgb_config)
        return qlib_params

    def _train_with_qlib(self, features, labels):
        """使用Qlib Dataset & Model训练"""
        dataset = self._build_qlib_dataset(features, labels)
        if dataset is None:
            raise ValueError("构建Qlib数据集失败，无法训练")
        
        if self.model is None:
            self.create_model()
        
        experiment_name = self.qlib_config.get('experiment_name', 'qlib_training')
        
        try:
            from qlib.workflow import R
            from qlib.workflow.record_temp import SignalRecord
        except ImportError as exc:
            raise ImportError("未安装qlib.workflow，请确认环境中已有qlib") from exc
        
        with R.start(experiment_name=experiment_name):
            R.log_params(model_type=self.model_type, backend='qlib')
            self.model.fit(dataset)
            recorder = R.get_recorder()
            if recorder is not None:
                if 'test' in dataset.segments:
                    sr = SignalRecord(self.model, dataset, recorder)
                    sr.generate()
                else:
                    default_logger.warning(
                        "Qlib数据集缺少'test'分段，跳过SignalRecord生成（仅 train/valid 分段）"
                    )
        
        self._qlib_dataset = dataset
        return dataset

    def _build_qlib_dataset(self, features, labels):
        """将特征和标签转换为Qlib Dataset"""
        try:
            from qlib.data.dataset import DatasetH
            from qlib.data.dataset.handler import DataHandlerLP
        except ImportError as exc:
            raise ImportError("未安装qlib.data模块，无法构建DatasetH") from exc
        
        feature_df = self._prepare_feature_frame(features)
        label_df = self._prepare_label_frame(labels)
        
        common_index = feature_df.index.intersection(label_df.index)
        if len(common_index) == 0:
            raise ValueError("特征与标签索引没有交集，无法构建Qlib数据集")
        
        feature_df = feature_df.loc[common_index]
        label_df = label_df.loc[common_index]
        
        combined = pd.concat(
            {
                'feature': feature_df,
                'label': label_df
            },
            axis=1
        )
        
        handler = DataHandlerLP.from_df(combined)
        segments = self._build_segments(feature_df.index)
        dataset = DatasetH(handler=handler, segments=segments)
        return dataset

    def _build_segments(self, index) -> dict:
        """根据时间索引构造训练/验证分段"""
        training_cfg = self.pipeline_config.get('training', {})
        split_ratio = training_cfg.get('split_ratio', 0.8)
        
        dates = pd.Index(index.get_level_values(0)).unique().sort_values()
        if len(dates) < 2:
            raise ValueError("可用日期不足以划分训练/验证集")
        
        split_idx = max(int(len(dates) * split_ratio), 1)
        if split_idx >= len(dates):
            split_idx = len(dates) - 1
        
        train_segment = (str(dates[0])[:10], str(dates[split_idx - 1])[:10])
        valid_segment = (str(dates[split_idx])[:10], str(dates[-1])[:10])
        
        segments = {'train': train_segment}
        if valid_segment[0] <= valid_segment[1]:
            segments['valid'] = valid_segment
        return segments

    def _prepare_feature_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Qlib训练要求特征为DataFrame格式")
        
        df = features.copy()
        df = self._ensure_multiindex(df)
        
        # 移除instrument列
        if 'instrument' in df.columns:
            df = df.drop(columns=['instrument'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]
        self.feature_names = numeric_cols
        return df

    def _prepare_label_frame(self, labels) -> pd.DataFrame:
        if isinstance(labels, pd.Series):
            df = labels.to_frame('label')
        elif isinstance(labels, pd.DataFrame):
            df = labels.copy()
        else:
            df = pd.DataFrame(labels, columns=['label'])
        
        if 'label' not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'label'})
        
        df = self._ensure_multiindex(df)
        return df[['label']]

    def _ensure_multiindex(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保索引为(datetime, instrument)格式"""
        result = df.copy()
        if isinstance(result.index, pd.MultiIndex):
            names = list(result.index.names)
            if names != ['datetime', 'instrument']:
                if 'datetime' in names and 'instrument' in names:
                    desired = ['datetime', 'instrument']
                    result = result.reorder_levels(desired).sort_index()
                else:
                    result.index = result.index.set_names(['datetime', 'instrument'])
                    result = result.sort_index()
        else:
            if 'instrument' not in result.columns:
                raise ValueError("数据缺少instrument信息，无法构建MultiIndex")
            if result.index.name != 'datetime':
                result.index = result.index.rename('datetime')
            result = result.set_index('instrument', append=True)
            result = result.sort_index()
            result.index = result.index.rename(['datetime', 'instrument'])
        return result

    def _evaluate_qlib_validation(self, dataset):
        """对Qlib验证集做评估"""
        if 'valid' not in dataset.segments:
            return
        
        from qlib.data.dataset.handler import DataHandlerLP
        
        pred_valid = self.model.predict(dataset, segment='valid')
        y_valid = dataset.prepare('valid', col_set='label', data_key=DataHandlerLP.DK_L)
        
        if isinstance(y_valid, pd.DataFrame):
            y_arr = y_valid.values.ravel()
        else:
            y_arr = y_valid
        
        metrics = Metrics.evaluate_model(pred_valid.values, y_arr)
        Metrics.print_metrics(metrics)

    def print_feature_importance(self, top_n: int = 20):
        """打印特征重要度（兼容Qlib & 原生LightGBM）"""
        if not self.feature_names:
            self.feature_names = getattr(self.model, 'feature_names', None)
        
        if self._use_qlib_backend():
            booster = getattr(self.model, 'model', None)
            if booster is None:
                return
            importance = booster.feature_importance(importance_type='gain')
            names = booster.feature_name()
            if not names:
                names = self.feature_names
            self._log_feature_importance(names, importance, top_n)
        else:
            if hasattr(self.model, 'print_feature_importance'):
                self.model.print_feature_importance(top_n)

    def _log_feature_importance(self, names, importance, top_n):
        if names is None or importance is None:
            return
        importance_df = pd.DataFrame({'feature': names, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        default_logger.info("=" * 60)
        default_logger.info(f"Top {top_n} 特征重要度:")
        for _, row in importance_df.iterrows():
            default_logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        default_logger.info("=" * 60)

