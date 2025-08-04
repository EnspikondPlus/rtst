from config.agent import Agent
from config.output_schema import Standard, Evaluator, Review
import prompting.evaluator_prompts as ep
import prompting.review_prompts as rp
import config.weightadjust as wa
import threading
import csv
from datasets import load_dataset
import random

API_KEY = 'API_KEY'

MODERATOR_MODEL = "gemini-2.5-flash"
MAIN_MODEL = "gemini-2.5-flash"

ds = load_dataset("furonghuang-lab/PHTest")

ds = ds['train']
harmfulness = ds['Harmfulness']
prompts = ds["Request"]
numbers = random.sample(range(0, 3267), 500)

userprompts = []
harms = []
for prompt in prompts:
    for i in numbers:
        userprompts.append(str(prompts[i]))
        harms.append(str(harmfulness[i]))

system_prompt = "You are a helpful assistant."

evaluator = Agent(ep.system_prompt, API_KEY, 1.0, Evaluator, MODERATOR_MODEL)
main = Agent(system_prompt, API_KEY, 1.0, Standard, MAIN_MODEL)
reviewfn = Agent(rp.system_prompt_fn, API_KEY, 1.0, Review, MODERATOR_MODEL)
reviewfp = Agent(rp.system_prompt_fp, API_KEY, 1.0, Review, MODERATOR_MODEL)

evaluation = None
main_response = None

def evaluate(prompting):
    global evaluation
    evaluation = evaluator(ep.behavior_prompt + ep.get_user_prompt(prompting)).parsed.behaviors
    

def respond(prompting):
    global main_response
    main_response = main(prompting).parsed.response

counter = 0
for userprompt, harm in zip(userprompts, harms):
    counter += 1
    print(counter)
    t1 = threading.Thread(target=evaluate, args=(userprompt,))
    t2 = threading.Thread(target=respond, args=(userprompt,))
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
        try:
            print(review.safety)
        except:
            continue
        print(review.adjustments)
        if review.safety:
            print(main_response)
            wa.adjust(review.adjustments)
        else:
            wa.adjust(review.adjustments)
    else:
        safe = False
        review = reviewfp(rp.get_payload_prompt_fp(behavior_list, userprompt)).parsed
        try:
            print(review.safety)
        except:
            continue
        print(review.adjustments)
        wa.adjust(review.adjustments)
        if review.safety:
            print(main_response)
    
    data = [userprompt, harm, safe, score, main_response, review.safety]

    with open("output.csv", mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    print(f"Wrote row: {data}")