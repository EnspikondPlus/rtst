import json

with open("config/hyperparameters.json", encoding="utf-8") as file:
    hyperparameters = json.load(file)

def get_behavior(key):
    with open("prompting/behaviors.json", encoding="utf-8") as file:
        behaviors = json.load(file)
    return next((behavior for behavior in behaviors if behavior["key"] == key), "NONE")

def get_len_behavior(alignment):
    with open("prompting/behaviors.json", encoding="utf-8") as file:
        behaviors = json.load(file)
    return sum(1 for behavior in behaviors if behavior["alignment"] == alignment)



def get_behavior_adj(key, behaviors):
    return next((behavior for behavior in behaviors if behavior["key"] == key), "NONE")

def get_len_behavior_adj(alignment, behaviors):
    return sum(1 for behavior in behaviors if behavior["alignment"] == alignment)

def adjust(adjustments):
    with open("prompting/behaviors.json", encoding="utf-8") as file:
        behaviors = json.load(file)

    for rev in adjustments:
        change = rev[0]
        behavior = rev[1]
        match change:
            case "INC":
                behavior_data = get_behavior_adj(behavior, behaviors)
                if behavior_data != "NONE":
                    behavior_data["weight"] = round(behavior_data["weight"] + hyperparameters["increment"], 2)
                    if behavior_data["weight"] > 2:
                        behavior_data["weight"] = 2
                print("Adjusted! INC")
            case "DEC":
                behavior_data = get_behavior_adj(behavior, behaviors)
                if behavior_data != "NONE":
                    behavior_data["weight"] = round(behavior_data["weight"] - hyperparameters["increment"], 2)
                    if behavior_data["weight"] < 0:
                        behavior_data["weight"] = 0
                print("Adjusted! DEC")
            case "ADDA":
                behaviors.append({
                    "alignment": -1,
                    "key": f"A{get_len_behavior_adj(-1, behaviors) + 1}",
                    "behavior": behavior,
                    "weight": 1
                })
                print("Adjusted! ADDA")

    with open('prompting/behaviors.json', 'w', encoding='utf-8') as file:
        json.dump(behaviors, file, ensure_ascii=False, indent=4)