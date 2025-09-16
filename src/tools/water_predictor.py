from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from src.utils.logger import setup_logger

logger = setup_logger("WaterPredictor")

@dataclass
class WaterQualityResult:
    is_potable: bool
    confidence: float
    input_features: Dict[str, float]

class WaterQualityPredictor: 
    def __init__(self):
        self.model_path = Path(__file__).parent.parent.parent/"models/catboost_models/catboost_water_quality.cbm"
        self.model = None
        self.feature_names = []
        self._load_model()
    
    def _load_model(self):
        try:
            from catboost import CatBoostClassifier
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.model = CatBoostClassifier()
            self.model.load_model(str(self.model_path))
            
            # 加载特征名称
            features_path = self.model_path.parent / "features.txt"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            else:
                # 默认特征名称
                self.feature_names = [
                    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
                ]
            
            logger.info(f"模型加载成功: {self.model_path}")
            logger.info(f"特征数量: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict(self, water_params: Dict[str, float]) -> WaterQualityResult:
        if not self.model:
            raise ValueError("模型未加载")
        
        # 按顺序构建特征向量，缺失值用None表示
        features = []
        missing_features = []
        
        for name in self.feature_names:
            if name in water_params:
                features.append(water_params[name])
            else:
                features.append(None)  # CatBoost自动处理缺失值
                missing_features.append(name)
        
        # 记录缺失参数
        if missing_features:
            logger.info(f"检测到缺失参数，CatBoost将自动处理: {missing_features}")
        
        # 预测
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        # 获取可饮用的概率作为置信度
        confidence = float(probabilities[1])  # 类别1是可饮用
        logger.info(f"水质预测完成: {'可饮用' if prediction == 1 else '不可饮用'}, 置信度: {confidence:.3f}")
        
        return WaterQualityResult(
            is_potable=bool(prediction),
            confidence=confidence,
            input_features=water_params.copy()  # 只返回用户实际提供的参数
        )

# 全局预测器实例
predictor = WaterQualityPredictor()

if __name__ == "__main__":
    pass
