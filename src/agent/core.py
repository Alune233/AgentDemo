from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from src.utils.logger import setup_logger
from src.agent.llm_manager import LLMManager
from src.agent.tool_manager import ToolManager

logger = setup_logger("Agent")

class SimpleCallback(BaseCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        logger.info(f"调用工具: {action.tool}, 参数: {action.tool_input}")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"工具输出: {output[:100]}...")

class WaterAgent:
    def __init__(self, llm_type: str = "auto", enable_streaming: bool = True):
        self.llm_manager = LLMManager(llm_type)
        self.tool_manager = ToolManager()
        self.enable_streaming = enable_streaming
        # 创建系统提示词
        self.system_prompt = """你是水利法规智能体，可以查询法规、生成代码、预测水质。
                            请根据用户的需求选择合适的工具来帮助用户。如果不需要使用工具，直接回答用户的问题。
                            回答时请：
                            1. 保持友好和专业的语调
                            2. 如果要使用了工具，请执行工具调用并解释工具的结果
                            3. 如果使用了代码执行工具，请返回代码运行结果并解释代码的作用和结果
                            4. 如果使用了水质预测工具，请解释预测结果的含义
                            5. 如果使用了法规查询工具，请总结相关要点，最后对法规结果给出来源片段（哪部法律的哪一条）与相似度分数
                            6. 请在回答时使用markdown格式，并使用```python代码块包裹代码
                            7. 如果某工具出现根本性调用错误（如无法连接/无法加载等）则不再重复调用此工具并请根据报错信息给出解决方案"""
        # 创建对话记忆
        self.memory = ConversationBufferWindowMemory(
            k=10,  # 保留最近10轮对话
            memory_key="chat_history",
            return_messages=True
        )
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        # 确保LLM已初始化
        if not self.llm_manager.llm:
            raise ValueError("LLM初始化失败")
        # 创建工具调用agent
        self.agent = create_tool_calling_agent(
            llm=self.llm_manager.llm,
            tools=self.tool_manager.get_tools(),
            prompt=self.prompt
        )
        # 创建agent执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tool_manager.get_tools(),
            memory=self.memory,
            verbose=True,  # 保持现有的verbose
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",  # 处理超限情况
            callbacks=[SimpleCallback()],
        )
        
        logger.info(f"智能体初始化完成 (LLM: {self.llm_manager.llm_type})")
    
    def chat(self, user_input: str) -> str:
        logger.info(f"用户输入: {user_input}")
        
        try:
            # 使用LangChain AgentExecutor处理用户输入
            response = self.agent_executor.invoke({"input": user_input})
            result = response["output"]
            
            # logger.info(f"\n智能体: {result}")
            logger.info("响应生成完成")
            return result
            
        except Exception as e:
            logger.error(f"处理用户请求失败: {e}")
            return "抱歉，处理您的请求时出现了错误。"