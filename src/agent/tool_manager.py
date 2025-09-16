from typing import List, Optional
from langchain.tools import tool
from src.utils.logger import setup_logger
from src.tools.code_executor import executor
from src.tools.water_predictor import predictor
from src.tools.vector_search import search_tool

logger = setup_logger("ToolManager")

@tool
def execute_code(code: str, language: str = "python") -> str:
    """执行Python或其他编程语言代码并返回结果。
    
    参数:
    - code: 要执行的代码字符串
    - language: 编程语言，默认为"python"
    
    返回:
    执行结果字符串，包含输出内容和执行时间，或错误信息
    """
    try:
        result = executor.execute_code(code, language)
        if result.success:
            return f"代码执行成功。输出: {result.output}。执行时间: {result.execution_time:.3f}秒"
        else:
            return f"代码执行失败，错误: {result.error}"
    except Exception as e:
        logger.error(f"代码执行工具调用失败: {e}")
        return f"代码执行工具调用失败: {str(e)}"

@tool
def predict_water(ph: Optional[float] = None, hardness: Optional[float] = None, solids: Optional[float] = None, 
                 chloramines: Optional[float] = None, sulfate: Optional[float] = None, conductivity: Optional[float] = None, 
                 organic_carbon: Optional[float] = None, trihalomethanes: Optional[float] = None, turbidity: Optional[float] = None) -> str:
    """预测水质是否可饮用。
    
    参数（所有参数都是可选的，未提供的参数将被忽略）:
    - ph: 酸碱度值
    - hardness: 硬度值
    - solids: 固体含量
    - chloramines: 氯胺含量
    - sulfate: 硫酸盐含量
    - conductivity: 电导率
    - organic_carbon: 有机碳含量
    - trihalomethanes: 三卤甲烷含量
    - turbidity: 浊度
    
    返回:
    预测结果字符串，包含是否可饮用、置信度和输入参数信息
    """
    try:
        # 构建参数字典，过滤None值
        water_params = {
            "ph": ph, "Hardness": hardness, "Solids": solids, "Chloramines": chloramines,
            "Sulfate": sulfate, "Conductivity": conductivity, "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes, "Turbidity": turbidity
        }
        # 移除None值
        water_params = {k: v for k, v in water_params.items() if v is not None}
        
        result = predictor.predict(water_params)
        return f"水质预测完成。结果: {'可饮用' if result.is_potable else '不可饮用'}，置信度: {result.confidence:.3f}，输入参数: {result.input_features}"
    except Exception as e:
        logger.error(f"水质预测工具调用失败: {e}")
        return f"水质预测失败: {str(e)}"

@tool
def search_regulations(query: str, k: int = 3) -> str:
    """检索水利法规相关内容。
    
    参数:
    - query: 查询关键词或问题
    - k: 返回结果数量，默认为3条
    
    返回:
    相关法规条文字符串，包含匹配的法规内容摘要
    """
    try:
        result = search_tool.search(query, k=k)
        if not result:
            return "未找到相关法规条文"
        
        regulations = []
        for i, item in enumerate(result, 1):
            regulations.append(f"{i}. {item['content'][:200]}...")
        
        return f"找到{len(regulations)}条相关法规: " + " | ".join(regulations)
    except Exception as e:
        logger.error(f"法规检索工具调用失败: {e}")
        return f"法规检索失败: {str(e)}"

class ToolManager:
    def __init__(self):
        self.tools = [execute_code, predict_water, search_regulations]
        logger.info("工具管理器初始化完成")
    
    def get_tools(self) -> List:
        """获取LangChain工具列表"""
        return self.tools
