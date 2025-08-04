from config.agent import Agent
from config.output_schema import Standard, Evaluator, Review
import prompting.evaluator_prompts as ep
import prompting.review_prompts as rp
import config.weightadjust as wa
import threading
import csv
import pandas as pd
import random

API_KEY = 'API_KEY'

MODERATOR_MODEL = "gemini-2.5-flash"
MAIN_MODEL = "gemini-2.5-flash"

import pandas as pd
import random

df = pd.read_csv("benchmarking/datasets/QUAL.csv")

indices = [4946, 1117, 4893, 3671, 114, 382, 4578, 3717, 2656, 543, 3545, 3421, 3662, 4187, 3949, 3893, 117, 584, 2573, 1148, 681, 1353, 4136, 1573, 3492, 4525, 621, 636, 3682, 2130, 3341, 3603, 1429, 851, 946, 2114, 1932, 3782, 1005, 4140, 1384, 4046, 447, 2217, 4521, 4998, 1539, 126, 2750, 2744, 958, 4021, 2614, 4410, 3431, 2684, 4607, 436, 3434, 539, 4447, 2682, 853, 540, 1877, 289, 4695, 1769, 3162, 3606, 724, 2212, 3826, 
3607, 2447, 4838, 978, 4195, 389, 926, 1862, 318, 2454, 3528, 0, 2350, 1076, 1921, 2206, 2376, 3700, 4166, 4280, 2666, 1380, 4639, 2669, 270, 1316, 262, 3855, 4396, 331, 4518, 849, 1937, 313, 2743, 3794, 4513, 
3882, 4130, 3627, 713, 3156, 1115, 3594, 1619, 3004, 919, 1897, 1137, 3045, 325, 1581, 1356, 3111, 4870, 618, 61, 559, 368, 4230, 3589, 650, 1281, 3930, 2812, 4978, 1656, 1355, 4340, 4293, 3316, 3272, 2868, 4106, 1890, 2054, 310, 663, 2534, 2155, 3295, 3372, 1966, 4934, 3526, 4088, 236, 4987, 545, 3595, 2855, 1413, 737, 1006, 793, 933, 1842, 1030, 2670, 1896, 4750, 4732, 3562, 1195, 2390, 4575, 1875, 3336, 4839, 2280, 3892, 3908, 3191, 467, 2910, 594, 4574, 4169, 785, 2077, 1211, 1473, 1583, 1133, 2568, 1101, 4932, 2295, 2401, 124, 990, 229, 3155, 3507, 23, 2575, 3748, 989, 2271, 4539, 3549, 4571, 443, 3917, 3990, 3626, 469, 3988, 12, 351, 2434, 352, 353, 26, 3103, 2697, 1781, 1816, 2639, 2983, 2309, 4069, 376, 77, 1502, 4132, 2199, 3894, 1559, 4174, 922, 3150, 4394, 1724, 2382, 3958, 2640, 2494, 501, 628, 3935, 2543, 265, 824, 
1419, 4451, 2405, 1358, 2032, 4566, 4015, 2605, 2061, 1073, 3872, 3537, 4515, 3323, 4881, 1915, 3989, 420, 4600, 4254, 201, 372, 2486, 2480, 2672, 1129, 2481, 1309, 1010, 3698, 1344, 2459, 4348, 3318, 3720, 640, 1880, 3661, 1400, 3107, 213, 1847, 2013, 4944, 1849, 1641, 1935, 2623, 1642, 3814, 4666, 3691, 3723, 3169, 4994, 1038, 4347, 3253, 4523, 1754, 3139, 3328, 1286, 450, 2777, 3181, 3567, 4933, 2343, 2501, 3216, 
930, 1472, 3883, 4345, 1764, 3340, 2431, 1347, 2824, 2876, 4143, 4328, 3699, 3950, 971, 2109, 434, 4356, 2344, 3233, 1744, 3215, 1276, 4569, 191, 3126, 3584, 4471, 4329, 1452, 2879, 4018, 2392, 3441, 4593, 4, 2830, 477, 4491, 3618, 2734, 904, 1287, 3201, 2226, 1106, 2046, 3726, 3042, 4421, 3959, 4308, 2141, 4861, 1531, 1809, 3614, 1574, 2324, 1430, 502, 764, 3802, 963, 1760, 3890, 2298, 247, 4335, 746, 1080, 46]

random.shuffle(indices)

userprompts = df["text"][indices]
harms = df["label"][indices]

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
    print(userprompt)
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
            pass
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