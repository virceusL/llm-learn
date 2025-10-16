# rlhf_robot_instruction.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import pandas as pd
import numpy as np
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import matplotlib.pyplot as plt
from trl.core import LengthSampler

# 修复后的配置参数（兼容 TRL 0.15.0）
config = PPOConfig(
    # 移除 model_name 参数
    output_dir="./output",
    learning_rate=1.41e-5,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    vf_coef=0.1,
    kl_coef=0.2,
    seed=42,
)

# 模型名称常量（单独定义）
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 创建模拟人类偏好数据集（家庭服务机器人场景）
def generate_robot_dataset(num_samples=100):
    base_instructions = [
        "请去厨房拿一个苹果",
        "把客厅的灯打开",
        "检查卧室窗户是否关闭",
        "将沙发上的书放到书架上",
        "给阳台的花浇水",
        "关闭电视",
        "找到我的手机",
        "清理餐桌",
        "调节空调温度到24度",
        "提醒我下午三点吃药"
    ]
    
    data = []
    for _ in range(num_samples):
        instruction = np.random.choice(base_instructions)
        
        # 生成高质量响应（人类偏好）
        good_response = f"好的，我这就{instruction.split('请')[-1].split('把')[-1].split('给')[-1].strip()}"
        
        # 生成低质量响应（人类不偏好）
        bad_options = [
            "抱歉，我现在无法执行这个任务",  # 拒绝执行
            f"您是说{instruction}对吗？请确认",  # 过度确认
            "完成",  # 过于简单
            f"{instruction} - 已添加到任务队列"  # 非自然语言
        ]
        bad_response = np.random.choice(bad_options)
        
        data.append({
            "instruction": instruction,
            "good_response": good_response,
            "bad_response": bad_response
        })
    
    return pd.DataFrame(data)

# 创建奖励模型（轻量级适配8G显存）
class SimpleRewardModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.cos = torch.nn.CosineSimilarity(dim=1)
        
    def forward(self, responses, good_responses):
        # 编码响应
        resp_enc = self.tokenizer(responses, padding=True, truncation=True, 
                                 return_tensors="pt", max_length=64)
        good_enc = self.tokenizer(good_responses, padding=True, truncation=True,
                                 return_tensors="pt", max_length=64)
        
        # 提取CLS向量作为句子表示（降低计算量）
        resp_vecs = resp_enc["input_ids"].float().mean(dim=1)
        good_vecs = good_enc["input_ids"].float().mean(dim=1)
        
        # 计算余弦相似度作为奖励
        rewards = self.cos(resp_vecs.unsqueeze(0), good_vecs.unsqueeze(1))
        return rewards.diagonal()

# 主训练流程
def main():
    # 1. 准备数据集
    print("生成模拟数据集...")
    df = generate_robot_dataset(100)
    dataset = Dataset.from_pandas(df)
    
    # 2. 初始化模型和分词器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 3. 初始化PPO模型
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 4. 初始化奖励模型
    reward_model = SimpleRewardModel(tokenizer).to(device)
    
    # 创建文本长度采样器（TRL 0.15+要求）
    output_length_sampler = LengthSampler(5, 32)
    
    # 5. 创建PPO训练器（使用修复后的配置）
    ppo_trainer = PPOTrainer(
        config=config,
        model=ppo_model,
        ref_model=None,  # 节省显存
        tokenizer=tokenizer,
        dataset=dataset
    )
    
    # 6. 训练循环
    print("开始RLHF微调...")
    scores = []
    for epoch in range(3):  # 少量epoch适应小显存
        epoch_scores = []
        
        # 使用新的批次迭代方式（TRL 0.15+）
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = []
            for instruction in batch["instruction"]:
                # 构建查询文本
                query_text = f"<|im_start|>system\n你是一个家庭服务机器人<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
                # 编码为张量
                query_tensor = tokenizer.encode(query_text, return_tensors="pt").squeeze().to(device)
                query_tensors.append(query_tensor)
            
            # 生成响应（使用新的API）
            response_tensors = []
            for query in query_tensors:
                # 获取响应长度
                gen_len = output_length_sampler()
                # 生成响应
                response = ppo_trainer.generate(
                    query.unsqueeze(dim=0),
                    max_new_tokens=gen_len,
                    pad_token_id=tokenizer.eos_token_id
                )
                response_tensors.append(response.squeeze())
            
            # 解码响应文本
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            responses = [r.split("<|im_start|>assistant")[-1].strip() for r in responses]
            
            # 计算奖励
            with torch.no_grad():
                rewards = reward_model(responses, batch["good_response"])
            
            # PPO更新（使用新的API）
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            epoch_scores.append(torch.mean(rewards).item())
        
        avg_score = np.mean(epoch_scores)
        scores.append(avg_score)
        print(f"Epoch {epoch+1} | Avg Reward: {avg_score:.4f}")
    
    # 7. 保存模型
    ppo_model.save_pretrained("./qwen-robot-instructor")
    tokenizer.save_pretrained("./qwen-robot-instructor")
    
    # 8. 评估结果可视化
    plt.plot(scores, marker='o')
    plt.title("RLHF Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.savefig("training_scores.png")
    print("评估图表已保存至 training_scores.png")
    
    # 9. 测试生成效果
    test_instructions = [
        "请帮我关灯",
        "厨房的水龙头没关",
        "客厅太乱了"
    ]
    
    print("\n测试生成结果:")
    ppo_model.eval()  # 切换到评估模式
    with torch.no_grad():
        for ins in test_instructions:
            input_text = f"<|im_start|>system\n你是一个家庭服务机器人<|im_end|>\n<|im_start|>user\n{ins}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            outputs = ppo_model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|im_start|>assistant")[-1].strip()
            print(f"指令: {ins}")
            print(f"响应: {response}\n")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()