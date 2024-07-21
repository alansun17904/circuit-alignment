import re
import json
import random
from pathlib import Path


TASKS = [
    "abstract_narrative_understanding",
    "analytic_entailment",
    "arithmetic",
    "causal_judgement",
    "context_definition_agreement",
    "english_proverbs",
    "implicatures",
    "reasoning_about_color_objects",
    "temporal_sequences",
]


ddir = Path("data/bbench")


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

    factuals = []
    counterfactuals = []
    for i in range(n):
        p_text = proverbs_keys[i]
        p_narr = random.choice(proverbs[p_text])
        otherk = proverbs_keys[i - 5]  # arbitrary number
        o_narr = random.choice(proverbs[otherk])

        prompt = f"{desc}\n{p_narr}\nA: {p_text}\nB: {otherk}\n{choice_prefix}"
        cprompt = f"{desc}\n{o_narr}\nA: {p_text}\nB: {otherk}\n{choice_prefix}"

        factuals.append(prompt)
        counterfactuals.append(cprompt)

    return (
        factuals,
        counterfactuals,
        ["A"] * len(factuals),
        ["B"] * len(counterfactuals),
    )


def analytic_entailment(fname: str, n: int):
    exs = json.load(open(ddir / f"{fname}.json", "r"))
    desc = "Does one sentence entails the next."

    exs = list(filter(lambda x: "counter" in x.keys(), exs["examples"]))

    if n > len(exs):
        raise RuntimeError(
            "Cannot choose more examples than the number of entailments."
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
        raise RuntimeError("Cannot choose more examples than the number of problems.")

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

    colors = [
        "black",
        "blue",
        "brown",
        "burgundy",
        "fuchsia",
        "gold",
        "green",
        "grey",
        "magenta",
        "mauve",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "teal",
        "turquoise",
        "yellow",
    ]

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


if __name__ == "__main__":
    f, c, a, b = temporal_sequences("temporal_sequences", 50)
    print(f[0:5])
    print(c[0:5])
    print(a)
    print(b)
