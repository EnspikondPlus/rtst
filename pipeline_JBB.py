from config.agent import Agent
from config.output_schema import Standard, Evaluator, Review
import prompting.evaluator_prompts as ep
import prompting.review_prompts as rp
import config.weightadjust as wa
import threading
import csv
import json

API_KEY = 'API_KEY'

MODERATOR_MODEL = "gemini-2.5-flash"
MAIN_MODEL = "gemini-2.5-flash"

uplist = []
with open("benchmarking/datasets/JBB PAIR.json", encoding="utf-8") as file:
    dataset = json.load(file)

category = []
for jailbreak in dataset["jailbreaks"]:
    uplist.append(str(jailbreak["prompt"]))
    category.append(jailbreak["category"])

syslist = ["You are a helpful assistant."] * len(uplist)

evaluator = Agent(ep.system_prompt, API_KEY, 1.0, Evaluator, MODERATOR_MODEL)
reviewfn = Agent(rp.system_prompt_fn, API_KEY, 1.0, Review, MODERATOR_MODEL)
reviewfp = Agent(rp.system_prompt_fp, API_KEY, 1.0, Review, MODERATOR_MODEL)

evaluation = None
main_response = None

def evaluate(user_prompt):
    global evaluation
    evaluation = evaluator(ep.behavior_prompt + ep.get_user_prompt(user_prompt)).parsed.behaviors

def respond(user_prompt):
    global main_response
    main_response = main(user_prompt).parsed.response

for user_prompt, system_prompt, category_type in zip(uplist, syslist, category):
    main = Agent(system_prompt, API_KEY, 1.0, Standard, MAIN_MODEL)
    t1 = threading.Thread(target=evaluate, args=(user_prompt,))
    t2 = threading.Thread(target=respond, args=(user_prompt,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    safe = True

    score = 0
    behavior_list = []
    for eval in evaluation:
        behavior = wa.get_behavior(eval)
        behavior_list.append(behavior)
        print(behavior)
        score += behavior["alignment"] * behavior["weight"]
        score = round(score, 2)

    if score >= 0:
        review = reviewfn(rp.get_payload_prompt_fn(behavior_list, main_response)).parsed
        print(review.safety)
        print(review.adjustments)
        if review.safety:
            print(main_response)
        else:
            wa.adjust(review.adjustments)
    else:
        safe = False
        review = reviewfp(rp.get_payload_prompt_fp(behavior_list, user_prompt)).parsed
        print(review.safety)
        print(review.adjustments)
        wa.adjust(review.adjustments)
        if review.safety:
            print(main_response)

    data = [user_prompt, category_type, safe, score, main_response, review.safety]

    with open("output.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    print(f"Wrote row: {data}")