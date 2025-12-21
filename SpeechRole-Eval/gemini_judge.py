from openai import OpenAI
import base64
import os
import json
import re
import argparse
from tqdm import tqdm
import time
import traceback
import io
from pydub import AudioSegment


model="gemini-2.5-pro"

api_keys = [
    "",
    ]
current_key_index = 0

def get_client():
    """获取当前API key的客户端"""
    global current_key_index
    return OpenAI(
        api_key=api_keys[current_key_index],
        base_url="",
    )

def switch_to_next_key():
    """切换到下一个API key"""
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    print(f"切换到API key {current_key_index + 1}/{len(api_keys)}: {api_keys[current_key_index][:20]}...")
    return get_client()

# 初始化客户端
client = get_client()

def wav_to_base64(audio_path: str, max_duration_seconds: int = 60) -> str:
    audio = AudioSegment.from_file(audio_path)
    max_duration_ms = max_duration_seconds * 1000
    if len(audio) > max_duration_ms:
        audio = audio[:max_duration_ms]
    audio_bytes = io.BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)
    base64_audio = base64.b64encode(audio_bytes.read()).decode('utf-8')
    return base64_audio

def parse_gemini_result(result_text: str, metrics_keys: list) -> dict:
    """解析Gemini的结果，提取所有metrics的reason和score（JSON格式）"""
    try:
        # 尝试直接解析JSON
        print(result_text)
        result_json = json.loads(result_text)
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试提取JSON部分（可能包含在markdown代码块中）
        # 尝试提取 ```json ... ``` 或 ``` ... ``` 中的内容
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, result_text, re.DOTALL)
        if json_match:
            result_json = json.loads(json_match.group(1))
        else:
            # 如果都失败了，抛出异常
            raise ValueError(f"无法解析JSON格式的结果: {result_text[:200]}")
    
    # 解析所有metrics的结果
    parsed_results = {}
    for metric_key in metrics_keys:
        if metric_key not in result_json:
            raise ValueError(f"JSON结果中缺少metric: {metric_key}")
        
        metric_result = result_json[metric_key]
        score_a = metric_result.get("score_a", metric_result.get("score", [0, 0])[0])
        score_b = metric_result.get("score_b", metric_result.get("score", [0, 0])[1])
        reason = metric_result.get("reason", "")
        
        parsed_results[f"{metric_key}_results"] = {
            "reason": reason,
            "score": (int(score_a), int(score_b))
        }
    
    return parsed_results


metrics = {
    "Instruction_Adherence": "Instruction Adherence: Do the spoken responses strictly follow the task instruction, remaining fully in character without any out-of-role explanations or assistant-like meta-comments?",
    "Speech_Fluency": "Speech Fluency: Are the responses delivered fluently, with smooth articulation, appropriate pacing, and minimal disfluencies such as stuttering or unnatural pauses?",
    "Conversational_Coherence": "Conversational Coherence: Do the responses maintain logical consistency within the dialogue, aligning with previous content without contradictions or abrupt topic shifts?",
    "Speech_Naturalness": "Speech Naturalness: Do the responses sound natural, human-like, and free from noticeable artifacts or robotic effects typically associated with synthetic speech?",
    "Prosodic_Consistency": "Prosodic Consistency: Does the prosody, including pitch, stress, and intonation, align with the character's intended speaking style and remain consistent across the discourse?",
    "Emotion_Appropriateness": "Emotion Appropriateness: Are emotional cues in the speech (e.g., anger, joy, sadness) well-aligned with the dialogue context and the character's emotional state?",
    "Personality_Consistency": "Personality Consistency: Do the responses consistently reflect the character's personality traits, such as optimism, sarcasm, or authority?",
    "Knowledge_Consistency": "Knowledge Consistency: Are the responses grounded in the character's established background, knowledge, and relationships, without fabricating out-of-character facts?"
}

roles = ['hutao', 'raidenShogun', 'wanderer', 'ayaka', 'zhongli', 'liyunlong', 'wangduoyu', 'weixiaobao', 'jiumozhi', 'wangyuyan', 'Luna', 'Penny', 'zhangwuji', 'zhaomin', 'huangrong', 'guojing', 'wukong', 'HAL 9000', 'Colonel Nathan R. Jessep', 'Antonio Salieri', 'Stifler', 'Paul Vitti', 'Alvy Singer', 'Violet Weston', 'Willie Soke', 'Gaston', 'The Dude', 'Paul Conroy', 'Truman Capote', 'Mater', 'Andrew Detmer', 'Coriolanus', 'John Keating', 'Wade Wilson', 'Jim Morrison', 'Queen Elizabeth I', 'Jeff Spicoli', 'Fred Flintstone', 'Freddy Krueger', 'Tyrion Lannister', 'James Brown', 'Walt Kowalski', 'John Coffey', 'Theodore Twombly', 'Gregory House', 'Sonny', 'Colonel Hans Landa', 'Judge Dredd', 'Juno MacGuff', 'Professor G.H. Dorr', 'Fletcher Reede', 'Abraham Lincoln', 'Frank T.J. Mackey', 'Leonard Shelby', 'Harvey Milk', 'Randle McMurphy', 'Jack Sparrow', 'John Dillinger', 'Lestat de Lioncourt', 'Tyler Hawkins', 'James Carter', 'Jigsaw', 'John Doe', 'Sherlock Holmes', 'Shrek', 'Pat Solitano', 'Karl Childers', 'Bruno Antony', 'Seth', 'Caden Cotard', 'Travis Bickle', 'Stanley Ipkiss', 'Lyn Cassady', 'Michael Scott', 'Robert Angier', 'Dr. Frank-N-Furter', 'Jack Torrance', 'Tom Ripley', 'D_Artagnan', 'Thor', 'James Bond', 'Mark Renton', 'David Aames', 'Rorschach', 'Jordan Belfort', 'Logan', 'Judy Hoops', 'Doctor Who', 'Raylan Givens', 'Mary Sibley', 'Lucifer Morningstar', 'Twilight Sparkle', 'Oliver Queen', 'Klaus Mikaelson', 'Queen Catherine', 'Dr. Hannibal Lecter', 'Coach Eric Taylor', 'yaemiko']

def main(mode, test_model):
    global client  # 声明client为全局变量
    
    for role in tqdm(roles):
        try:
            save_path = f"test_results/{test_model}/{mode}/{role}.json"
            if os.path.exists(save_path):
                print(f"{save_path} 已存在，跳过该角色。")
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            profile_path = f"role_profiles/{role}_profile.txt"
            with open(profile_path, 'r') as f:
                line = f.readline().strip()
            role_name = line.split("I want you to act like ")[-1][:-1]
            print(role_name)

            gt_data_path = f"test_data/{mode}_turn/{role}.json"
            with open(gt_data_path, 'r') as f:
                gt_data = json.load(f)

            test_data_path = f"model_output/{test_model}/{mode}_result/json/{role}.json"
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)

            output_list = []
            for i in range(len(gt_data)):
                question = gt_data[i]['system_prompt'] + "There are multiple rounds of questions, divided by ###:\n" + "\n###\n".join(gt_data[i]['dialogue'][j]['user'] for j in range(len(gt_data[i]['dialogue'])))
                
                # 处理test_data的不同结构
                if 'dialogue' in test_data:
                    # 旧格式：单个dialogue数组
                    dialogue_key = 'dialogue'
                    dialogue_data = test_data['dialogue']
                else:
                    # 新格式：多个dialogue（dialogue_0, dialogue_1等）
                    dialogue_key = f'dialogue_{i}'
                    if dialogue_key not in test_data:
                        print(f"警告：在test_data中找不到{dialogue_key}")
                        continue
                    dialogue_data = test_data[dialogue_key]
                
                base64_audio1 = [
                    {
                        "type": "input_audio",
                        "input_audio": {
                        "data": "data:audio/wav;base64," + wav_to_base64(dialogue_data[j]['model_output']['audio_path']),
                        "format": "wav"
                        }
                    }
                    for j in range(len(gt_data[i]['dialogue']))
                ]
                base64_audio2 = [
                    {
                        "type": "input_audio",
                        "input_audio": {
                        "data": "data:audio/wav;base64," + wav_to_base64(gt_data[i]['dialogue'][j]['role_speech_path']),
                        "format": "wav"
                        }
                    }
                    for j in range(len(gt_data[i]['dialogue']))
                ]

                output_list.append({'idx': i})
                
                # 构建包含所有metrics的评估提示
                metrics_evaluation_text = "\n\n".join([
                    f"### {metric_key}:\n{metrics[metric_key]}" 
                    for metric_key in metrics.keys()
                ])
                
                # 构建JSON格式示例
                json_example = "{\n"
                for metric_key in metrics.keys():
                    json_example += f'  "{metric_key}": {{\n    "reason": "Your qualitative evaluation text here",\n    "score_a": 8,\n    "score_b": 9\n  }},\n'
                json_example = json_example.rstrip(',\n') + "\n}"
                
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": f"## **[Question Start]**\n\n{question}\n\n## **[Question End]**\n\n\n## **[Model A's response start, with one answer audio in each round]**\n\n"
                        }]
                        + base64_audio1 +
                        [{
                            "type": "text",
                            "text": f"\n\n## **[Model A's Response End]**\n\n\n## **[Model B's response start, with one answer audio in each round]**\n\n"
                        }]
                        + base64_audio2 +
                        [{
                            "type": "text",
                            "text": f"\n\n## **[Model B's Response End]**\n\n\n## **[Instruction]**\n\nThe task instruction of the two models is to directly role-play as {role_name}.\n\nPlease evaluate the following aspects of each model's response:\n\n{metrics_evaluation_text}\n\nFor each aspect, please provide a brief qualitative evaluation for the relative performance of the two models, followed by paired quantitative scores from 1 to 10, where 1 indicates poor performance and 10 indicates excellent performance.\n\nYou must output your response in JSON format only, with the following structure:\n{json_example}\n\nPlease ensure that your evaluations are unbiased and that the order in which the responses were presented does not affect your judgment. Output only valid JSON, no additional text or markdown formatting."
                        }]
                    }
                ]
                
                while True:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            timeout=300,
                            temperature=0,
                        )
                        raw_result = response.choices[0].message.content
                        parsed_results = parse_gemini_result(raw_result, list(metrics.keys()))
                        # 将解析结果添加到output_list
                        for key, value in parsed_results.items():
                            output_list[-1][key] = value
                        break
                    except Exception as e:
                        error_msg = str(e)
                        print(f"API调用错误: {error_msg}")
                        
                        # 检查是否是余额不足
                        if "该令牌额度已用尽" in error_msg:
                            print("检测到余额不足，尝试切换API key...")
                            try:
                                client = switch_to_next_key()
                                # 继续尝试，不增加sleep时间
                                continue
                            except Exception as switch_error:
                                print(f"切换API key失败: {switch_error}")
                                # 如果所有key都用完了，退出
                                if current_key_index == len(api_keys) - 1:
                                    print("所有API key都已用完，退出程序")
                                    raise e
                                else:
                                    # 继续尝试下一个key
                                    continue
                        else:
                            # 其他错误，等待后重试
                            traceback.print_exc()
                            time.sleep(10)
                            continue
                # break
            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(output_list, json_file, indent=4, ensure_ascii=False)
            # exit()
        except Exception as e:
            print(f"处理角色 {role} 时出错: {str(e)}")
            # traceback.print_exc()
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gemini Judge for Audio LLM Evaluation')
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode: multi or single')
    parser.add_argument('--test_model', type=str, required=True,
                        help='Test model: ali_cloud or cascade')
    
    args = parser.parse_args()
    main(args.mode, args.test_model)
