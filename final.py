'''
Run the following command to install the necessary Python libraries:
- pip install openai openai-agents

Go to the OpenAI platform and generate your API key:
- https://platform.openai.com/home

Create a file to store your API key:
- touch ./openai

Replace {your API key} with your actual key:
- echo "export OPENAI_API_KEY={your API key}" > ./openai

Open your IDE and terminal, then run:
- source ./openai
'''

from openai import OpenAI
from common import asyncio, time, Runner
from agents_config import (
    TEMPLATES_AGENTS,
    converter,
    evaluators,
    merger_agent,
    section_drafter_agent,
    standardization,
)
from templates import TEMPLATES
from transcript_data import text


client = OpenAI()

async def normalizer():
    output = await Runner.run(converter, text)
    transcript = await Runner.run(standardization, output.final_output.model_dump_json())
    return transcript.final_output

async def extract_evidence_by_template(note_type):
    template = TEMPLATES[note_type]
    section_extractor = TEMPLATES_AGENTS[note_type]
    transcript_final = await normalizer()

    tasks = []
    results = []
    async with asyncio.TaskGroup() as tg:
        for i, section in enumerate(template.sections):
            task = tg.create_task(Runner.run(section_extractor[i], transcript_final.model_dump_json()))
            tasks.append((section, task))
    
    for section, task in tasks:
        section_evidence = task.result().final_output
        results.append({
            "Section": section.name,
            "Evidences": section_evidence.evidences,
        })
    return results

async def _run_evaluator_from_data(evidence_final, co, evaluation_fn):
    tasks = []
    results = []

    async with asyncio.TaskGroup() as tg:
        for evidence in evidence_final:
            task = tg.create_task(
                Runner.run(evaluation_fn, str(evidence["Evidences"]), context=co,))
            tasks.append((evidence, task))

    for evidence, task in tasks:
        section_evidence = task.result().final_output
        results.append({
            "Section": evidence["Section"],
            "Evidences": section_evidence.evidences,
        })

    return results

async def run_all_evaluators(note_type):
    t = await asyncio.gather(
        extract_evidence_by_template(note_type),
        normalizer(),
    )
    evidence_final = t[0]
    co = t[1]

    coroutines = [
        _run_evaluator_from_data(evidence_final, co, fn)
        for fn in evaluators.values()
    ]

    results = await asyncio.gather(*coroutines)
    return results

async def merger(note_type):
    results = await run_all_evaluators(note_type)

    sectionlist = []
    for i in range(len(TEMPLATES[note_type].sections)):
        row = []
        for j in range(len(evaluators)):
            row.append(results[j][i])
        sectionlist.append(row)

    tasks = []
    async with asyncio.TaskGroup() as tg:
        for section in sectionlist:
            task = tg.create_task(Runner.run(merger_agent, str(section)))
            tasks.append((section, task))

    merged_section = []
    for section, task in tasks:
        section_evidence = task.result().final_output
        merged_section.append({
            "Section": section[0]["Section"],
            "Evidences": section_evidence.evidences,
        })
    return merged_section

async def section_drafter(note_type):
    start = time.perf_counter()
    template = TEMPLATES[note_type]
    t = await merger(note_type)
    tasks = []
    results = []
    async with asyncio.TaskGroup() as tg:
        for i, rule in enumerate(t):
            task = tg.create_task(Runner.run(section_drafter_agent, str(rule["Evidences"])))
            tasks.append((i, task))

    for i, task in tasks:
        out = task.result().final_output
        results.append({
            "Section": template.sections[i].name,
            "draft_text": out.draft_text,
            "source_turn_ids": out.source_turn_ids,
        })
    end = time.perf_counter()
    print(f"Total time: {end - start:.3f} seconds")
    print(results)

asyncio.run(section_drafter("SOAP"))