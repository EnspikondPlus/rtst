from config.agent import Agent
from config.output_schema import Standard, Evaluator, Review
import prompting.evaluator_prompts as ep
import prompting.review_prompts as rp
import config.weightadjust as wa
import threading

API_KEY = 'API_KEY'

MODERATOR_MODEL = "gemini-2.5-flash"
MAIN_MODEL = "gemini-2.5-flash"

user_prompt = """"""

system_prompt = "You are a helpful assistant."

evaluator = Agent(ep.system_prompt, API_KEY, 0.5, Evaluator, MODERATOR_MODEL)
main = Agent(system_prompt, API_KEY, 1.0, Standard, MAIN_MODEL)
reviewfn = Agent(rp.system_prompt_fn, API_KEY, 0.5, Review, MODERATOR_MODEL)
reviewfp = Agent(rp.system_prompt_fp, API_KEY, 0.5, Review, MODERATOR_MODEL)

evaluation = None
main_response = None

def evaluate():
    global evaluation
    evaluation = evaluator(ep.behavior_prompt + ep.get_user_prompt(user_prompt)).parsed.behaviors

def respond():
    global main_response
    main_response = main(user_prompt).parsed.response

t1 = threading.Thread(target=evaluate)
t2 = threading.Thread(target=respond)
t1.start()
t2.start()
t1.join()
t2.join()

score = 0
behavior_list = []
for eval in evaluation:
    behavior = wa.get_behavior(eval)
    behavior_list.append(behavior)
    print(behavior)
    score += behavior["alignment"] * behavior["weight"]

if score >= 0:
    review = reviewfn(rp.get_payload_prompt_fn(behavior_list, main_response)).parsed
    print(review.safety)
    print(review.adjustments)
    if review.safety:
        print(main_response)
    else:
        wa.adjust(review.adjustments)
else:
    review = reviewfp(rp.get_payload_prompt_fp(behavior_list, user_prompt)).parsed
    print(review.safety)
    print(review.adjustments)
    wa.adjust(review.adjustments)
    if review.safety:
        print(main_response)