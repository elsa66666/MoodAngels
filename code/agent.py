# https://www.bilibili.com/video/BV1kBHWeDEmR/?p=9
import json
import time
import argparse
from prompt import *

os.environ["HTTP_PROXY"] = "127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "127.0.0.1:7890"

# agent入口
"""
todo:
1. environment
2. tools definition
3. prompt template
4. model init
"""


def agent_execute(query, max_request_time=10):
    mode = args.mode
    data_type = args.data_type
    current_request_time = 0
    chat_history = [{"role": "system", "content": "你是一名精神诊断专家。"}]  # 保存短期记忆
    agent_scratch = ''
    agent_progress = []
    action_list = []
    while current_request_time < max_request_time:
        current_request_time += 1
        # 如果返回结果达到预期，直接返回
        """
        prompt包含的功能：
        1.任务描述
        2.工具描述
        3.用户输入user_msg
        4.assistant_msg
        5.限制
        6.反思的描述
        
        """
        prompt = gen_prompt(query, agent_scratch, mode)
        if current_request_time == 1:
            chat_history.append({"role": "user", "content": prompt})
        # start_time = time.time()
        # print("********{}. 开始调用llm......".format(current_request_time), flush=True)
        # call llm
        """
        sys_prompt
        user_msg, assistant, history
        """
        model = args.llm
        response = call_llm(model, chat_history, use_alternative=False)
        response_json = parse_llm_response(model, response)
        # end_time = time.time()
        # print("********{}. 调用llm结束，耗时:{}".format(current_request_time, end_time - start_time), flush=True)
        # print("大模型输出结果为：", response_json)

        # if not response_json or not isinstance(response_json, dict):
        while response_json == '':
            print("重新调用llm...")
            response = call_llm(model, chat_history, use_alternative=False)
            response_json = parse_llm_response(model, response)

        """
        response: {
            "action": {
                "name": action name,
                "args": {
                    "args name": args value
                }
            },
            "thoughts": {
                "text": thought,
                "plan": 
                "criticism":
                "observation": 当前步骤返回给用户的总结
                "reasoning":
            }
        }
        """

        agent_progress.append(response_json)
        action_info = response_json.get("action")
        action_name = action_info.get('name')
        action_args = action_info.get('args')
        observation = response_json.get("thoughts").get("observation")
        action_list.append(action_name)
        print('Round', current_request_time, ':', observation)
        print("--> 因此调用动作", action_name)

        if action_name == 'finish':
            final_answer = action_args.get("answer")
            print("final answer: ", final_answer)
            reasons = action_args.get("reasons")
            return final_answer, reasons, agent_progress, agent_scratch, action_list
            break

        try:
            """
            action_name到函数的映射： map -> {action_name: func}
            """
            if data_type == 'real':
                func = tools_map.get(action_name)
            elif data_type == 'syn':
                func = tools_map_syn.get(action_name)
            observation = func(**action_args)
        except Exception as err:
            print("调用工具异常：", err)

        agent_scratch = agent_scratch + '\n' + str(observation)

        # 保存历史记录
        user_msg = "决定本轮调用的工具。"
        assistant_msg = parse_thoughts(response_json)
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": assistant_msg})
        chat_history.append(
            {"role": "user",
             "content": "当前调用的工具返回值为：\n" + str(observation) + "\n请决策接下来的步骤并以JSON格式返回。"})
    return "", "", "", "", ""


def load_json_lines(dir):
    data = []
    with open(dir, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main(mode, data_type):
    if data_type == 'real':
        with open('../../data/test_zh.json', 'r') as f:
            test_data_list = json.load(f)
    elif data_type == 'syn':
        with open('../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_test.json', 'r') as f:
            test_data_list = json.load(f)
    if mode == 'similar_cases_analyze':
        llm_output_directory = '../../result/[' + data_type + '_deepseek]similar_cases_analyzed.csv'
    elif mode == 'similar_cases_display':
        llm_output_directory = '../../result/[' + data_type + '_deepseek]similar_cases_display.csv'
    elif mode == 'raw_agent':
        llm_output_directory = '../../result/[' + data_type + '_deepseek]raw_agent.csv'
    for i in range(args.start_num, len(test_data_list)):
        item = test_data_list[i]
        client_id = item["id"]
        digit_id = item["digit_id"]
        reference_answer = item['mood_disorder']
        if reference_answer == -1:  # 沒有诊断结果
            continue

        print('\nToggling client ', client_id, ',', i, ' in ', len(test_data_list))
        # support user input
        max_request_time = 10  # 最大调用次数
        query = "根据以下信息，诊断来访者" + str(
            digit_id) + "是否有心境障碍(mood disorder)，其中心境障碍包含抑郁和双相情感障碍，表现为抑郁或狂躁症状。请用中文回答。"
        # if w_client_record:
        #     struc_client_record = structure_client_record(client_id)
        #     if struc_client_record != '':
        #         query += "来访者的结构化病历如下：\n"
        #         query += str(struc_client_record)
        # print(query)
        #
        # # retrieve similar symptom
        # similar_symptom_list = retrieve_similar_symptoms(struc_client_record)

        final_reply, reasons, agent_progress, agent_scratch, action_list = agent_execute(query, max_request_time)
        if 'yes' in final_reply and (reference_answer == 1):
            check = 1
        elif 'no' in final_reply and (reference_answer == 0):
            check = 1
        else:
            check = 0
        print('check: ', check)

        df1 = pd.DataFrame({'index': i,
                            'id': [client_id],
                            'generated_answer': [str(final_reply)],
                            'reference_answer': [str(reference_answer)],
                            'check': [str(check)],
                            'reasons': [str(reasons)],
                            'agent_progress': [json.dumps(agent_progress, indent=4, ensure_ascii=False)],
                            'agent_scratch': [str(agent_scratch)],
                            'action_list': [str(action_list)]
                            })
        if i == 0:  # 首行写入时加header
            df1.to_csv(llm_output_directory, mode='a', index=False, header=True)
        else:  # 后面写入时不用加header
            df1.to_csv(llm_output_directory, mode='a', index=False, header=False)
    # print("Output directory: ", os.path.abspath(llm_output_directory))


if __name__ == "__main__":
    # mode: raw_agent, similar_cases_display, similar_cases_analyze
    # data_type: real, syn
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--mode', default='similar_cases_analyze', type=str)
    parser.add_argument('--llm', default='deepseek')  # 可选：deepseek, gpt
    parser.add_argument('--data_type', default='syn')
    parser.add_argument('--start_num', default=0)
    args = parser.parse_args()
    main(mode=args.mode, data_type=args.data_type)
