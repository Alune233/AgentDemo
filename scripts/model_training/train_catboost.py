import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def load_and_preprocess_data():
    print("加载数据...")
    df = pd.read_csv('data/water_potability.csv')
    # 中位数填充空值
    df = df.fillna(df.median())
    # 设置X和Y
    x = df.drop('Potability', axis=1)
    y = df['Potability']

    print(f"特征数: {x.shape[1]}")
    print(f"样本数: {x.shape[0]}")
    print(f"正负样本: {y.value_counts().to_dict()}")
    
    return x, y

def split_data(x, y):
    print("\n划分数据集...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {len(x_train)} | 测试集: {len(x_test)}")
    return x_train, x_test, y_train, y_test

def grid_search_catboost(x_train, y_train):
    print("\n开始网格搜索...")
    
    # 定义参数网格
    param_grid = {
        'iterations': [100, 300, 500],
        'learning_rate': [0.03, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
    }
    # 基础模型
    base_model = CatBoostClassifier(
        random_seed=42,
        verbose=False,
        eval_metric='AUC',
        train_dir=None,  # 禁用训练日志目录
        allow_writing_files=False,  # 禁止写入文件
    )
    # 网格搜索
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=10,  # 10折交叉验证
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2  # 显示详细进度
    )
    
    grid_search.fit(x_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳CV分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, x_test, y_test):
    # 测试集评估
    test_pred = model.predict(x_test)
    test_proba = model.predict_proba(x_test)[:, 1]
    # 评估指标
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print("测试集:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, test_pred))
    
    return {
        'test': {'accuracy': test_acc, 'f1': test_f1, 'auc': test_auc}
    }

def save_model_and_results(model, best_params, results, feature_names):
    print("\n保存模型...")
    
    # 创建模型目录
    Path('models/catboost_models').mkdir(parents=True, exist_ok=True)
    # 保存模型
    model_path = 'models/catboost_models/catboost_water_quality.cbm'
    model.save_model(model_path)
    print(f"模型保存: {model_path}")
    
    # 保存最佳参数
    with open('models/catboost_models/best_params.txt', 'w', encoding='utf-8') as f:
        f.write("CatBoost最佳参数\n")
        f.write("=" * 20 + "\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    
    # 保存评估结果
    with open('models/catboost_models/results.txt', 'w', encoding='utf-8') as f:
        f.write("CatBoost评估结果\n")
        f.write("=" * 25 + "\n\n")
        for dataset, metrics in results.items():
            f.write(f"{dataset}集:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
    
    # 保存特征名称
    with open('models/catboost_models/features.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(feature_names))
    
def test_model_loading():
    print("\n测试模型加载...")
    
    # 加载模型
    loaded_model = CatBoostClassifier()
    loaded_model.load_model('models/catboost_models/catboost_water_quality.cbm')
    
    print(f"模型加载成功")

def main():
    # 数据准备
    x, y = load_and_preprocess_data()
    x_train, x_test, y_train, y_test = split_data(x, y)  
    # 网格搜索
    best_model, best_params = grid_search_catboost(x_train, y_train) 
    # 模型评估
    results = evaluate_model(best_model, x_test, y_test)
    # 保存模型
    save_model_and_results(best_model, best_params, results, list(x.columns))
    # 测试加载
    test_model_loading()

if __name__ == "__main__":
    main()