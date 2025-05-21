from tools import *
from debate_prompt import *

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


class DebateAgent:
    def __init__(self, system_message):
        self.chat_history = [{"role": "system", "content": system_message}]
        self.agent_progress = []
        self.max_request_time = 5


def debate_execute(query, model, max_round=4):
    current_request_time = 0
    pos_agent = DebateAgent(
        system_message="你是一名精神诊断专家，正在进行多专家会诊，你认为当前来访者存在心境障碍(mood disorder)，其中心境障碍包含抑郁和双相情感障碍，表现为抑郁或狂躁症状。")
    neg_agent = DebateAgent(
        system_message="你是一名精神诊断专家，正在进行多专家会诊，你认为当前来访者不存在心境障碍(mood disorder)，其中心境障碍包含抑郁和双相情感障碍，表现为抑郁或狂躁症状。")
    judge_agent = DebateAgent(system_message="你是一名精神诊断专家，在多专家会诊中担任裁判。")
    round = 0
    debate_progress = []

    while round < max_round:
        round += 1
        print("Round", round)
        pos_prompt = gen_debate_prompt(query, 'positive')
        neg_prompt = gen_debate_prompt(query, 'negative')
        judge_prompt = gen_debate_prompt(query, 'judge')
        if round == 1:
            pos_agent.chat_history.append({"role": "user", "content": pos_prompt})
            neg_agent.chat_history.append({"role": "user", "content": neg_prompt})
            judge_agent.chat_history.append({"role": "user", "content": judge_prompt})

        # pos agent发言
        response = call_llm(model, pos_agent.chat_history, use_alternative=False)
        response_json = parse_llm_response(model, response)
        while response_json == '':
            print("重新调用llm...")
            response = call_llm(model, pos_agent.chat_history, use_alternative=False)
            response_json = parse_llm_response(model, response)
        """
        response: {
        "response": "你认为该来访者有心境障碍的理由以及对于反方发言的反驳。",
        "thoughts": {
            "plan": "简要概述短期和长期计划", 
            "criticism": "有建设性的自我批评"
        }
        """
        pos_response = str(response_json.get("response"))
        print("正方：", pos_response)
        pos_agent.chat_history.append({"role": "assistant", "content": pos_response})
        neg_agent.chat_history.append({"role": "user", "content": ("正方：" + pos_response)})
        judge_agent.chat_history.append({"role": "user", "content": ("正方：" + pos_response)})
        debate_progress.append("正方：" + pos_response)

        # 反方发言
        response = call_llm(model, neg_agent.chat_history, use_alternative=False)
        response_json = parse_llm_response(model, response)
        while response_json == '':
            print("重新调用llm...")
            response = call_llm(model, neg_agent.chat_history, use_alternative=False)
            response_json = parse_llm_response(model, response)
        """
        response: {
        "response": "你认为该来访者有心境障碍的理由以及对于反方发言的反驳。",
        "thoughts": {
            "plan": "简要概述短期和长期计划", 
            "criticism": "有建设性的自我批评"
        }
        """
        neg_response = str(response_json.get("response"))
        print("反方：", neg_response)
        neg_agent.chat_history.append({"role": "assistant", "content": neg_response})
        pos_agent.chat_history.append({"role": "user", "content": ("反方：" + neg_response)})
        judge_agent.chat_history.append({"role": "user", "content": ("反方：" + neg_response)})
        debate_progress.append("反方：" + neg_response)
        # 裁判发言
        if round == max_round:
            judge_agent.chat_history.append({"role": "user", "content": "当前是最后一轮，请结束辩论并做出诊断。"})
        response = call_llm(model, judge_agent.chat_history, use_alternative=False)
        response_json = parse_llm_response(model, response)
        while response_json == '':
            print("重新调用llm...")
            response = call_llm(model, judge_agent.chat_history, use_alternative=False)
            response_json = parse_llm_response(model, response)
        """
        response: {
            "judge": "你认为辩论是否结束，仅回答yes或no。",
            "diagnose": "你认为该来访者是否有心境障碍，仅回答yes或no。",
            "thoughts": {
                "plan": "简要概述短期和长期计划", 
                "criticism": "有建设性的自我批评",
                "judge_reasons": "你认为辩论结束或尚未结束的原因。",
                "reasoning": "你认为来访者是或不是心境障碍的原因。"}
        }
        """
        print("裁判：")
        judge = response_json.get("judge")
        judge_reasons = response_json['thoughts'].get("judge_reasons")
        judge_agent.chat_history.append({"role": "assistant", "content": judge_reasons})
        if judge == 'yes':
            diagnose = response_json.get("diagnose")
            print("diagnose: ", diagnose)
            reasons = response_json['thoughts'].get("reasoning")
            return diagnose, reasons, debate_progress
            break
        else:
            print("裁判认为辩论尚未结束：\n", judge_reasons)
    return "", "", []


def get_agent_choices(dir, id):
    data = pd.read_csv(dir)
    description = ""
    for item in data.itertuples():
        current_id = getattr(item, 'id')
        if current_id == id:
            choice = getattr(item, 'generated_answer')
            reason = getattr(item, 'reasons')
            if 'yes' in choice:
                diagnosis = "是"
            elif 'no' in choice:
                diagnosis = "不是"
            if 'raw_agent.csv' in dir:
                agent_type = "未检索以往案例"
            elif 'similar_cases_display.csv' in dir:
                agent_type = "检索以往案例但未分析"
            elif 'similar_cases_analyzed.csv' in dir:
                agent_type = "检索以往案例并分析差异"
            description = agent_type + "的agent认为，该来访者" + diagnosis + "心境障碍，理由如下：" + reason
    return choice, description


def main(model, data_type):
    if data_type == 'real':
        with open('../../data/test_zh.json', 'r') as f:
            test_data_list = json.load(f)
    elif data_type == 'syn':
        with open('../../dataset_synthesis/tabsyn/synthetic/mood_angels/syn_test.json', 'r') as f:
            test_data_list = json.load(f)

    for i in range(3, len(test_data_list)):
        item = test_data_list[i]
        client_id = item["id"]
        digit_id = item["digit_id"]
        reference_answer = item['mood_disorder']
        if reference_answer == -1:  # 沒有诊断结果
            continue
        print('\nToggling client ', i, ' in ', len(test_data_list))
        if data_type == 'syn':
            choice1, description1 = get_agent_choices('../../result/syn_result/[syn_' + model + ']raw_agent.csv',
                                                      client_id)
            choice2, description2 = get_agent_choices(
                '../../result/syn_result/[syn_' + model + ']similar_cases_display.csv', client_id)
            choice3, description3 = get_agent_choices(
                '../../result/syn_result/[syn_' + model + ']similar_cases_analyzed.csv', client_id)
            llm_output_directory = '../../result/syn_result/[syn_' + model + ']debate.csv'
        elif data_type == 'real':
            choice1, description1 = get_agent_choices('../../result/real_result/[' + model + ']raw_agent.csv',
                                                      client_id)
            choice2, description2 = get_agent_choices(
                '../../result/real_result/[' + model + ']similar_cases_display.csv', client_id)
            choice3, description3 = get_agent_choices(
                '../../result/real_result/[' + model + ']similar_cases_analyzed.csv', client_id)
            llm_output_directory = '../../result/real_result/[' + model + ']debate.csv'
        # 三个agent看法一致
        if choice1 == choice2 and choice2 == choice3:
            instruction = "三位专家agent都对于该来访者做出的相同的判断并分别给出了理由，请用一段话总结他们的判断理由，仅总结理由不用额外做出判断。"
            prompt = instruction + "\n" + description1 + "\n" + description2 + "\n" + description3
            messages = [
                {"role": "system", "content": "你是一个精神诊断专家。"},
                {"role": "user", "content": prompt}
            ]
            if model == 'gpt':
                reasons = GPT().generate(messages)
            elif model == 'deepseek':
                reasons = DeepSeek().generate(messages)
            print('诊断理由: ', reasons)
            if 'yes' in choice1 and (reference_answer == 1):
                check = 1
            elif 'no' in choice1 and (reference_answer == 0):
                check = 1
            else:
                check = 0
            print('check: ', check)
            df1 = pd.DataFrame({'index': i,
                                'id': [client_id],
                                'generated_answer': [str(choice1)],
                                'reference_answer': [str(reference_answer)],
                                'check': [str(check)],
                                'reasons': [str(reasons)],
                                'debate_progress': [""]
                                })

        else:  # to debate
            # support user input
            query = "以下是一个困难案例，此前三位agent已分别做出诊断，以下是他们的诊断结果：" + "\n" + description1 + "\n" + description2 + "\n" + description3 + "\n以下是原始的案例信息：\n"
            if data_type == 'real':
                if get_client_records(digit_id) == "":  # real data, 无病历信息
                    query += ("来访者量表表现：\n" + get_scale_performances(digit_id)
                              + "\n以往量表参考：\n" + load_existing_similar_scales_display(
                                digit_id) + previous_scales_analysis(digit_id))
                else:  # real data，有病历信息
                    query += ("来访者病历信息：\n" + get_client_records(digit_id)
                              + "\n来访者量表表现：\n" + get_scale_performances(digit_id)
                              + "\n以往病历参考：\n" + load_existing_similar_records_display(
                                digit_id) + load_existing_similar_records_analyzed(digit_id)
                              + "\n以往量表参考：\n" + load_existing_similar_scales_display(
                                digit_id) + previous_scales_analysis(digit_id))
            elif data_type == 'syn':
                query += ("来访者量表表现：\n" + get_scale_performances_syn(digit_id)
                          + "\n以往量表参考：\n"
                          + load_existing_similar_scales_display(digit_id, data_type='syn')
                          + load_existing_similar_scales_analyzed(digit_id, data_type='syn'))
            diagnose, reasons, debate_progress = debate_execute(query, model, max_round=4)
            if 'yes' in diagnose and (reference_answer == 1):
                check = 1
            elif 'no' in diagnose and (reference_answer == 0):
                check = 1
            else:
                check = 0
            print('check: ', check)
            df1 = pd.DataFrame({'index': i,
                                'id': [client_id],
                                'generated_answer': [str(diagnose)],
                                'reference_answer': [str(reference_answer)],
                                'check': [str(check)],
                                'reasons': [str(reasons)],
                                'debate_progress': [str(debate_progress)]
                                })

        if i == 0:  # 首行写入时加header
            df1.to_csv(llm_output_directory, mode='a', index=False, header=True)
        else:  # 后面写入时不用加header
            df1.to_csv(llm_output_directory, mode='a', index=False, header=False)
    # print("Output directory: ", os.path.abspath(llm_output_directory))


if __name__ == "__main__":
    main(model='deepseek', data_type='syn')
