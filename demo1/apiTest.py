import requests
import json
import logging
import time

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}

def test_api_call(input_text, user_id="test_user", conversation_id="test_conversation", stream=False):
    """Make an API call with the given input text and parameters."""
    # 封装请求的参数
    data = {
        "messages": [{"role": "user", "content": input_text}],
        "stream": stream,
        "userId": user_id,
        "conversationId": conversation_id
    }
    
    logger.info(f"发送请求: {input_text}")
    
    # 接收流式输出处理
    if stream:
        full_response = ""
        try:
            with requests.post(url, stream=True, headers=headers, data=json.dumps(data)) as response:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if not line_text.startswith("data: "):
                            continue
                        json_str = line_text.strip("data: ")
                        # 检查是否为空或不合法的字符串
                        if not json_str:
                            logger.info(f"收到空字符串，跳过...")
                            continue
                        # 确保字符串是有效的JSON格式
                        if json_str.startswith('{') and json_str.endswith('}'):
                            try:
                                data = json.loads(json_str)
                                if 'choices' in data and 'delta' in data['choices'][0]:
                                    delta_content = data['choices'][0]['delta'].get('content', '')
                                    full_response += delta_content
                                    # 不打印每个块，以便看清完整响应
                                    # logger.info(f"流式输出，响应部分是: {delta_content}")
                                if data['choices'][0].get('finish_reason') == "stop":
                                    logger.info(f"完整响应是: {full_response}")
                                    return full_response
                            except json.JSONDecodeError as e:
                                logger.info(f"JSON解析错误: {e}")
                        else:
                            logger.info(f"无效JSON格式: {json_str}")
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            return None
    
    # 接收非流式输出处理
    else:
        try:
            # 发送post请求
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                logger.info(f"响应内容: {content}")
                return content
            else:
                logger.error(f"API请求失败，状态码: {response.status_code}")
                logger.error(f"错误信息: {response.text}")
                return None
        except Exception as e:
            logger.error(f"请求处理错误: {e}")
            return None

def run_memory_persistence_tests():
    """Test memory persistence with the PostgreSQL store."""
    # 生成唯一的用户ID和会话ID
    user_id = f"user_{int(time.time())}"
    conversation_id = f"conv_{int(time.time())}"
    
    logger.info("=== 测试1: 记住名字 ===")
    test_api_call("记住你的名字是南哥。", user_id, conversation_id)
    
    logger.info("=== 测试2: 询问套餐 ===")
    test_api_call("200元以下，流量大的套餐有啥？", user_id, conversation_id)
    
    logger.info("=== 测试3: 询问名字 ===")
    test_api_call("你叫什么名字？", user_id, conversation_id)
    
    logger.info("=== 测试4: 询问套餐细节 ===")
    test_api_call("就刚刚提到的这个套餐，是多少钱？", user_id, conversation_id)

def run_multiple_user_tests():
    """Test handling multiple users simultaneously."""
    logger.info("=== 测试多用户处理 ===")
    
    # 用户1记忆测试
    user_id_1 = "user_1"
    conv_id_1 = "conv_1"
    logger.info(f"用户1 ({user_id_1}) 存储记忆")
    test_api_call("记住你的名字是小明。", user_id_1, conv_id_1)
    
    # 用户2记忆测试
    user_id_2 = "user_2"
    conv_id_2 = "conv_2"
    logger.info(f"用户2 ({user_id_2}) 存储记忆")
    test_api_call("记住你的名字是小红。", user_id_2, conv_id_2)
    
    # 检查用户1记忆
    logger.info(f"用户1 ({user_id_1}) 查询记忆")
    test_api_call("你叫什么名字？", user_id_1, conv_id_1)
    
    # 检查用户2记忆
    logger.info(f"用户2 ({user_id_2}) 查询记忆")
    test_api_call("你叫什么名字？", user_id_2, conv_id_2)

def run_streaming_test():
    """Test streaming responses."""
    logger.info("=== 测试流式输出 ===")
    user_id = f"stream_user_{int(time.time())}"
    conversation_id = f"stream_conv_{int(time.time())}"
    
    # 先记住一个名字
    test_api_call("记住你的名字是流式测试助手。", user_id, conversation_id)
    
    # 然后用流式响应询问
    test_api_call(
        "你能详细介绍一下你作为客服代表能提供哪些服务吗？请至少列出5点。", 
        user_id, 
        conversation_id,
        stream=True
    )

if __name__ == "__main__":
    logger.info("开始测试PostgreSQL集成的LangGraph应用")
    
    # 基础记忆持久化测试
    run_memory_persistence_tests()
    
    # 多用户测试
    run_multiple_user_tests()
    
    # 流式输出测试
    run_streaming_test()
    
    logger.info("测试完成")