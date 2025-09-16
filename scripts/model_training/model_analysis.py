import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# 解决中文和负号显示问题
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def load_data():
    print("加载数据...")
    df = pd.read_csv('data/water_potability.csv')
    df = df.fillna(df.median())
    
    x = df.drop('Potability', axis=1)
    y = df['Potability']
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"数据加载完成: {x.shape[0]} 样本, {x.shape[1]} 特征")
    return x_train, x_test, y_train, y_test, list(x.columns)

def plot_feature_importance(model, feature_names, save_dir='models/catboost_models'):
    print("生成特征重要性图表...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取特征重要性
    importance = model.get_feature_importance()
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_df, x='importance', y='feature', palette='viridis')
    plt.title('特征重要性排序')
    plt.xlabel('重要性分数')
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据
    feature_df.to_csv(save_dir / 'feature_importance.csv', index=False)
    print(f"特征重要性图表已保存")
    
    return feature_df

def plot_shap_analysis(model, x_test, feature_names, save_dir='models/catboost_models', sample_size=200):
    print("生成SHAP分析...")
    
    save_dir = Path(save_dir)
    
    try:
        # 计算SHAP值
        explainer = shap.Explainer(model)
        x_sample = x_test.sample(n=min(sample_size, len(x_test)), random_state=42)
        shap_values = explainer(x_sample)
        
        # SHAP汇总图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_sample, feature_names=feature_names, show=False)
        plt.title('SHAP特征影响分析')
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP条形图
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, x_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.title('SHAP特征重要性')
        plt.tight_layout()
        plt.savefig(save_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP分析图表已保存")
        return shap_values
        
    except Exception as e:
        print(f"SHAP分析失败: {e}")
        return None

def generate_report(feature_df, shap_values, save_dir='models/catboost_models'):
    save_dir = Path(save_dir)
    
    with open(save_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("模型分析报告\n")
        f.write("=" * 20 + "\n\n")
        f.write("前5重要特征:\n")
        for i, row in feature_df.head(5).iterrows():
            f.write(f"{i+1}. {row['feature']}: {row['importance']:.3f}\n")
        
        if shap_values is not None:
            f.write(f"\nSHAP分析样本数: {len(shap_values)}\n")
        
        f.write("\n生成的图表:\n")
        f.write("- feature_importance.png: 特征重要性\n")
        if shap_values is not None:
            f.write("- shap_summary.png: SHAP影响分析\n")
            f.write("- shap_importance.png: SHAP重要性\n")
    
    print(f"分析报告已保存")

def main():
    # 检查模型文件是否存在
    model_path = Path('models/catboost_models/catboost_water_quality.cbm')
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train_catboost.py 训练模型")
        return
    
    # 加载模型
    print("加载模型...")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    print("模型加载成功")
    # 加载数据
    x_train, x_test, y_train, y_test, feature_names = load_data()
    # 开始分析
    print("\n开始模型分析...")
    # 特征重要性分析
    feature_df = plot_feature_importance(model, feature_names)
    # SHAP分析
    shap_values = plot_shap_analysis(model, x_test, feature_names)
    # 生成报告
    generate_report(feature_df, shap_values)

if __name__ == "__main__":
    main()