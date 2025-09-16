from src.agent.core import WaterAgent
from src.utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    try:
        # 初始化智能体
        agent = WaterAgent()
        logger.info("智能体启动成功")
        
        print(f"✅ 智能体已启动 (LLM: {agent.llm_manager.llm_type})")
        print("💬 开始对话 (输入 'quit' 或 'exit' 退出)")
        print("-" * 50)
        
        # 对话循环
        while True:
            try:
                user_input = input("\n用户: ").strip()
                # 退出命令
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 再见！")
                    break
                # 空输入跳过
                if not user_input:
                    continue
                # 获取智能体响应
                response = agent.chat(user_input)
                # print(f"\n智能体: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见！")
                break
            except Exception as e:
                logger.error(f"对话处理失败: {e}")
                print(f"❌ 处理失败: {e}")
        
    except ValueError as e:
        print(f"❌ 智能体启动失败: {e}")
        print("请检查.env文件中的LLM配置")
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"❌ 系统错误: {e}")

if __name__ == "__main__":
    main()