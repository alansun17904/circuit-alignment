import re
import json
import random
from pathlib import Path


TASKS = [
    "abstract_narrative_understanding",
    "analytic_entailment",
    "arithmetic_1dig_add",
    "arithmetic_1dig_sub",
    "arithmetic_2dig_add",
    "arithmetic_2dig_sub",
    "causal_judgement",
    "context_definition_agreement",
    "english_proverbs",
    "implicatures",
    "reasoning_about_color_objects_neither_color",
    "reasoning_about_color_objects_what_color",
    "reasoning_about_color_objects_yes_no_color",
    "temporal_sequences",
]


ddir = Path("data/bbench")


def _safe_remove(l, item):
    try:
        l.remove(item)
    except ValueError:
        pass
    return l

def arithmetic_choices(correct, prompt):
    correcti = int(correct)
    decoy1 = _safe_remove(list(range(10)),correcti)
    decoy2 = _safe_remove(list(range(10, 99)), correcti)
    return [
        correct,
        str(random.choice(decoy1)),
        str(random.choice(decoy2)),
        str(random.randint(200, 999)),
        "banana",
        "house",
    ]


def temporal_sequences(correct, prompt):
    times = re.findall(r"((\d+(am|pm)) to (\d+(am|pm)))", prompt)
    times = [v[0] for v in times]
    # remove the correct answer
    times = [v for v in times if v != correct]
    return [correct, *times[:3]]


TASK_CHOICES = {
    "abstract_narrative_understanding": ["A", "B"],
    "analytic_entailment": ["Yes", "No"],
    "arithmetic_1dig_add": arithmetic_choices,
    "arithmetic_1dig_sub": arithmetic_choices,
    "arithmetic_2dig_add": arithmetic_choices,
    "arithmetic_2dig_sub": arithmetic_choices,
    "causal_judgement": ["Yes", "No"],
    "implicatures": ["Yes", "No"],
    "english_proverbs": ["A", "B"],
    "reasoning_about_color_objects_what_color": [
        "black",
        "blue",
        "brown",
        "burgundy",
        "cyan",
        "gold",
        "green",
        "grey",
        "magenta",
        "white",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "teal",
        "turquoise",
        "yellow",
    ],
    "reasoning_about_color_objects_neither_color": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ],
    "reasoning_about_color_objects_yes_no_color": ["yes", "no"],
    "temporal_sequences": temporal_sequences,
}


def abstract_narrative_understanding(fname: str, n: int):
    exs = json.load(open(ddir / f"{fname}.json", "r"))
    desc = "Given a narrative choose the most related proverb."
    choice_prefix = "choice (A or B): "
    # get all of the proverbs and organize them by answer
    proverbs = dict()

    for ex in exs["examples"]:
        p_text = list(filter(lambda x: x[1] == 1, ex["target_scores"].items()))[0][0]
        if p_text not in proverbs:
            proverbs[p_text] = [ex["input"]]
        else:
            proverbs[p_text].append(ex["input"])
    proverbs_keys = list(proverbs.keys())

    if n > len(proverbs_keys):
        raise RuntimeError(
            f"Cannot choose more examples than the total number of proverbs ({len(proverbs_keys)})."
        )

    factuals, counterfactuals, answers, canswers = [], [], [], []
    for i in range(n):
        p_text = proverbs_keys[i]
        p_narr = random.choice(proverbs[p_text])
        otherk = proverbs_keys[i - 5]  # arbitrary number
        o_narr = random.choice(proverbs[otherk])
        if random.random() > 0.5:
            prompt = f"{desc}\n{p_narr}\nA: {p_text}\nB: {otherk}\n{choice_prefix}"
            cprompt = f"{desc}\n{o_narr}\nA: {p_text}\nB: {otherk}\n{choice_prefix}"
            answers.append("A")
            canswers.append("B")
        else:
            prompt = f"{desc}\n{p_narr}\nA: {otherk}\nB: {p_text}\n{choice_prefix}"
            cprompt = f"{desc}\n{o_narr}\nA: {otherk}\nB: {p_text}\n{choice_prefix}"
            answers.append("B")
            canswers.append("A")
        factuals.append(prompt)
        counterfactuals.append(cprompt)

    return (
        factuals,
        counterfactuals,
        answers,
        canswers,
    )


def analytic_entailment(fname: str, n: int):
    exs = json.load(open(ddir / f"{fname}.json", "r"))
    desc = "Does one sentence entails the next."

    exs = list(filter(lambda x: "counter" in x.keys(), exs["examples"]))

    if n > len(exs):
        raise RuntimeError(
            f"Cannot choose more examples than the number of entailments: {len(exs)}"
        )

    factuals, factuals_answers = [], []
    counterfactuals, counterfactuals_answers = [], []
    for i in range(n):
        prompt = f"{desc}\n{exs[i]['input']}\nchoice (Yes or No): "
        cprompt = f"{desc}\n{exs[i]['counter']}\nchoice (Yes or No): "
        factuals.append(prompt)
        counterfactuals.append(cprompt)
        factuals_answers.append(
            "Yes" if exs[i]["target_scores"]["entailment"] else "No"
        )
        counterfactuals_answers.append(
            "No" if exs[i]["target_scores"]["entailment"] else "Yes"
        )
    return factuals, counterfactuals, factuals_answers, counterfactuals_answers


def arithmetic(fname: str, n: int):
    exs = json.load(open(ddir / f"{fname}.json", "r"))["examples"]

    if n > len(exs):
        raise RuntimeError(
            f"Cannot choose more examples than the number of problems: {len(exs)}"
        )

    factuals, counterfactuals = [], []
    factual_answers, counterfactual_answers = [], []
    for i in range(n):
        factuals.append(f"{exs[i]["input"]} ")
        factual_answers.append(exs[i]["target"])
        counterfactuals.append(f"{exs[i-10]["input"]} ")  # arbitrary index
        counterfactual_answers.append(exs[i - 10]["target"])

    return factuals, counterfactuals, factual_answers, counterfactual_answers


def causal_judgement(fname: str, n: int):
    exs = json.load(open(ddir / f"{fname}.json", "r"))["examples"]
    desc = "How would a typical person answer each of the following questions about causation?"

    # only consider examples that have antipodal-causal pairs
    exs = list(
        filter(
            lambda x: any(
                x["input"].split()[:4] == v["input"].split()[:4]
                and x["target_scores"]["Yes"] != v["target_scores"]["Yes"]
                for v in exs
            ),
            exs,
        )
    )

    if n > len(exs):
        raise RuntimeError(f"n > {len(exs)}")

    factuals, factual_answers = [], []
    counterfactuals, counterfactual_answers = [], []

    for i in range(n):
        fin = exs[i]["input"].split()[:4], exs[i]["target_scores"]["Yes"]
        # find examples where the set up is the same but there is a different response
        cin = list(
            filter(
                lambda x: x["input"].split()[:4] == fin[0]
                and x["target_scores"]["Yes"] != fin[1],
                exs,
            )
        )

        factuals.append(f"{desc}\n{exs[i]["input"]} ")
        factual_answers.append("Yes" if exs[i]["target_scores"]["Yes"] else "No")

        counterfactuals.append(f"{desc}\n{cin[0]["input"]} ")
        counterfactual_answers.append("Yes" if cin[0]["target_scores"]["Yes"] else "No")

    return factuals, counterfactuals, factual_answers, counterfactual_answers


def implicatures(fname: str, n: int):
    exs = json.load(open(ddir / f"{fname}.json", "r"))["examples"]
    desc = "Does Speaker 2's answer mean Yes or No? "

    assert n <= len(exs), f"n <= {len(exs)}"

    yes = list(filter(lambda x: x["target_scores"]["yes"] == 1, exs))
    nos = list(filter(lambda x: x["target_scores"]["yes"] == 0, exs))

    factuals, factual_answers = [], []
    counterfactuals, counterfactual_answers = [], []

    for i in range(n):
        prompt = f"{exs[i]['input']}\n{desc}"
        ans = "Yes" if exs[i]["target_scores"]["yes"] == 1 else "No"
        if ans == "Yes":
            cin = random.choice(nos)
        else:
            cin = random.choice(yes)
        cprompt = f"{cin['input']}\n{desc}"
        cans = "No" if ans == "Yes" else "Yes"
        factuals.append(prompt)
        factual_answers.append(ans)
        counterfactuals.append(cprompt)
        counterfactual_answers.append(cans)

    return factuals, counterfactuals, factual_answers, counterfactual_answers


def temporal_sequences(fname: str, n: int):
    exs = json.load(open(ddir / f"{fname}.json", "r"))["examples"]
    factuals, factual_answers = [], []
    counterfactuals, counterfactual_answers = [], []

    assert n <= len(exs), f"n <= {len(exs)}"

    for i in range(n):
        ptext = exs[i]["input"] + " "
        target = exs[i]["target"]
        start, end = re.findall(r"(\d+)(am|pm)", target)
        starthr, startam = int(start[0]), start[1]
        endhr, endam = int(end[0]), end[1]
        cstart = str(12 if starthr == 1 else starthr - 1) + (
            startam if not (starthr == 12 and startam == "pm") else "am"
        )
        cend = str(12 if endhr == 11 else (endhr + 1) % 12) + (
            endam if not (endhr == 11 and endam == "am") else "pm"
        )

        ctext = re.sub(f"{starthr}{startam}", f"{cstart}", ptext)
        ctext = re.sub(f"{endhr}{endam}", f"{cend}", ctext)

        factuals.append(ptext)
        factual_answers.append(target)
        counterfactuals.append(ctext)
        counterfactual_answers.append(f"{cstart} to {cend}")

    return factuals, counterfactuals, factual_answers, counterfactual_answers


def color_objects(fname: str, n: int, type: str):
    assert type in ["neither_color", "what_color", "yes_no_color"]

    exs = json.load(open(ddir / f"{fname}.json", "r"))["examples"]
    exs = list(filter(lambda x: x["comment"] == type, exs))

    assert n <= len(exs), f"n <= {len(exs)}"

    factuals, factual_answers = [], []
    counterfactuals, counterfactual_answers = [], []

    colors = TASK_CHOICES["reasoning_about_color_objects_what_color"]

    for i in range(n):
        ex = exs[i]
        ptext, target = (
            ex["input"] + " ",
            list(filter(lambda x: x[1] == 1, ex["target_scores"].items()))[0][0],
        )
        factuals.append(ptext)
        factual_answers.append(target)
        if type == "what_color":
            # randomly choose a new color
            ntarget = random.choice(
                list(filter(lambda x: x[1] == 0, ex["target_scores"].items()))
            )[0]
            ctext = re.sub(target, ntarget, ptext)

            counterfactuals.append(ctext)
            counterfactual_answers.append(ntarget)
        elif type == "neither_color":
            query = re.match(
                rf".*neither ({'|'.join(colors)}) nor ({'|'.join(colors)})\?",
                ptext,
            )
            c1, c2 = query.group(1), query.group(2)
            context = ptext.split(".")[0]
            color_notin_ptext = random.choice(
                list(filter(lambda x: x not in ptext, colors))
            )
            if (
                target == "0"
            ):  # replace the query color with something that is not there
                # find a valid color that is not in the sentence
                cnew = color_notin_ptext
                ctext = re.sub(f"neither {c1}", f"neither {cnew}", ptext)
                ntarget = str(int(target) + 1)
            else:
                if c1 in context and c2 in context:  # remove one of them, t=t+1
                    cnew = color_notin_ptext
                    ctext = re.sub(f"neither {c1}", f"neither {cnew}", ptext)
                    ntarget = str(int(target) + 1)
                elif c1 not in context and c2 not in context:  # add one of them, t=t-1
                    # find a color that is in the context
                    cnew = random.choice(re.findall(f"({'|'.join(colors)})", context))
                    ctext = re.sub(f"neither {c1}", f"neither {cnew}", ptext)
                    ntarget = str(int(target) - 1)
                else:  # remove the one that is in, t=t+1
                    cnew = color_notin_ptext
                    if c1 in context:
                        ctext = re.sub(f"neither {c1}", f"neither {cnew}", ptext)
                    else:
                        ctext = re.sub(f"nor {c2}", f"nor {cnew}", ptext)
                    ntarget = str(int(target) + 1)
        elif type == "yes_no_color":
            query = re.match(rf".*({'|'.join(colors)})\?", ptext).group(1)
            last_color = re.compile(rf"({'|'.join(colors)})\?")
            if target == "yes":  # replace query color w/ wrong color
                ntarget = random.choice(list(filter(lambda x: x != query, colors)))
                ctext = re.sub(last_color, rf"{ntarget}?", ptext)
                ntarget = "no"
            else:  # replace query color w/ actual color
                ntarget = random.choice(
                    re.findall(rf"({'|'.join(colors)}).+?\.", ptext)
                )
                ctext = re.sub(last_color, rf"{ntarget}?", ptext)
                ntarget = "yes"
        counterfactuals.append(ctext)
        counterfactual_answers.append(ntarget)

    return factuals, counterfactuals, factual_answers, counterfactual_answers


def make_few_shot_prompt(prompts, answers, shots):
    """Generate few-shot prompts from a list of prompts and answers. Note that
    this method should be used sparingly since it does not guarantee that there
    is no information leakage between the few-shotted prompts.
    """
    shotted_prompts = []
    for i, prompt in enumerate(prompts):
        idxs = list(range(len(prompts)))
        exidxs = random.sample(_safe_remove(idxs, i), shots)
        few_shot = ""
        for ex in exidxs:
            few_shot += prompts[ex] + answers[ex] + "\n"
        few_shot += prompt
        shotted_prompts.append(few_shot)
    return shotted_prompts


def answer_index(task, prompt, answer):
    choices = TASK_CHOICES[task]
    if "arithmetic" in task or task == "temporal_sequences":
        choices = choices(answer, prompt)
    answer_idx = choices.index(answer)
    return answer_idx, choices


def generate_choices(task, prompts, answers):
    aidx_choices = list(map(lambda x: answer_index(task, x[0], x[1]), zip(prompts, answers)))
    aidx = [v[0] for v in aidx_choices]
    choices = [v[1] for v in aidx_choices]
    return prompts, aidx, choices


def generate_prompts(task):
    if task == "abstract_narrative_understanding":
        return abstract_narrative_understanding("abstract_narrative_understanding", 100)
    elif task == "analytic_entailment":
        return analytic_entailment("analytic_entailment", 54)
    elif task == "arithmetic_1dig_add":
        return arithmetic("1_digit_addition", 100)
    elif task == "arithmetic_1dig_sub":
        return arithmetic("1_digit_subtraction", 100)
    elif task == "arithmetic_2dig_add":
        return arithmetic("2_digit_addition", 1000)
    elif task == "arithmetic_2dig_sub":
        return arithmetic("2_digit_subtraction", 1000)
    elif task == "causal_judgement":
        return causal_judgement("causal_judgement", 179)
    elif task == "implicatures":
        return implicatures("implicatures", 492)
    elif task == "english_proverbs":
        return abstract_narrative_understanding("english_proverbs", 34)
    elif task == "reasoning_about_color_objects_what_color":
        return color_objects("reasoning_about_color_objects", 200, "what_color")
    elif task == "reasoning_about_color_objects_yes_no_color":
        return color_objects("reasoning_about_color_objects", 200, "yes_no_color")
    elif task == "reasoning_about_color_objects_neither_color":
        return color_objects("reasoning_about_color_objects", 200, "neither_color")
    elif task == "temporal_sequences":
        return temporal_sequences("temporal_sequences", 1000)