import asyncio
from pydantic import BaseModel
from agents import Agent, Runner
from openai import OpenAI
client = OpenAI()
class TranscriptTurn(BaseModel):
    turn_id: str
    speaker: str 
    start_ms: int
    end_ms: int
    text: str

class Transcript(BaseModel):
    turns: list[TranscriptTurn]

class SectionRule(BaseModel):
    name: str
    description: str
    allowed_content: str
    forbidden_content: str

class NoteTemplate(BaseModel):
    template_name: str
    sections: list[SectionRule]

class Evidence(BaseModel):
    evidence_id: str
    type: str
    value: str
    source_turn_ids: list[str]

class Evidencelist(BaseModel):
    evidences: list[Evidence]

SOAP_TEMPLATE = NoteTemplate(
    template_name="SOAP",
    sections=[
        SectionRule(
            name="Subjective",
            description="Patient-reported symptoms, history, concerns",
            allowed_content="Only patient-reported symptoms, history, concerns, symptom duration/severity, relevant negatives stated in transcript",
            forbidden_content="Do not include clinician interpretation, diagnosis, exam findings, or treatment decisions",
        ),
        SectionRule(
            name="Objective",
            description="Clinician observations and measurable findings",
            allowed_content="Only clinician-observed findings, exam findings, vitals, measurable data explicitly stated in transcript",
            forbidden_content="Do not include patient feelings, opinions, or inferred diagnoses",
        ),
        SectionRule(
            name="Assessment",
            description="Clinical impression",
            allowed_content="Only clinician-stated diagnoses/impressions or clearly supported clinical problems",
            forbidden_content="Do not invent diagnoses or add unsupported interpretation",
        ),
        SectionRule(
            name="Plan",
            description="Treatment and next steps",
            allowed_content="Treatments, medications, follow-up, referrals, instructions, next steps explicitly discussed",
            forbidden_content="Do not include unsupported diagnoses or actions not discussed",
        ),
    ]
)

HNP_TEMPLATE = NoteTemplate(
    template_name="H&P",
    sections=[
        SectionRule(
            name="Chief Complaint",
            description="Primary reason for visit in patient's own words",
            allowed_content="Main reason for visit, brief and concise, derived from patient statement",
            forbidden_content="Do not include history, diagnosis, or clinician interpretation",
        ),
        SectionRule(
            name="History of Present Illness",
            description="Detailed narrative of current problem",
            allowed_content="""
            Symptom onset, duration, severity, progression,
            associated symptoms, relevant negatives,
            patient-reported context
            """,
            forbidden_content="Do not include clinician interpretation or diagnosis",
        ),
        SectionRule(
            name="Past Medical History",
            description="Relevant previous conditions",
            allowed_content="Chronic diseases, previous diagnoses mentioned in transcript",
            forbidden_content="Do not include current symptoms or plan",
        ),
        SectionRule(
            name="Medications",
            description="Current medications",
            allowed_content="Medications explicitly mentioned, including refills",
            forbidden_content="Do not invent medications",
        ),
        SectionRule(
            name="Allergies",
            description="Drug or other allergies",
            allowed_content="Only allergies explicitly mentioned",
            forbidden_content="Do not assume no allergies if not stated",
        ),
        SectionRule(
            name="Family and Social History",
            description="Family conditions and lifestyle context",
            allowed_content="Family medical history, lifestyle info (smoking, alcohol, etc.)",
            forbidden_content="Do not mix with patient conditions",
        ),
        SectionRule(
            name="Physical Examination",
            description="Objective findings",
            allowed_content="Vitals, exam findings, measurable data explicitly stated",
            forbidden_content="Do not include patient-reported symptoms",
        ),
        SectionRule(
            name="Assessment",
            description="Clinical impression",
            allowed_content="Only clinician-stated diagnoses or clearly supported problems",
            forbidden_content="Do not invent diagnoses",
        ),
        SectionRule(
            name="Plan",
            description="Next steps and management",
            allowed_content="Medications, refills, follow-up, referrals, instructions",
            forbidden_content="Do not include unsupported diagnoses",
        ),
    ]
)

PHYSIO_TEMPLATE = NoteTemplate(
    template_name="Physiotherapy",
    sections=[
        SectionRule(
            name="Reason for Visit",
            description="Why the patient is attending physiotherapy",
            allowed_content=(
                "Main complaint, injury, pain area, functional limitation, referral reason, "
                "or treatment goal stated in the transcript"
            ),
            forbidden_content=(
                "Do not include clinician interpretation, treatment performed, or future plan"
            ),
        ),
        SectionRule(
            name="History",
            description="Relevant health and functional history related to the physio issue",
            allowed_content=(
                "Mechanism of injury, onset, duration, aggravating/relieving factors, prior episodes, "
                "relevant medical history, prior treatment, functional impact, pain description"
            ),
            forbidden_content=(
                "Do not include measured exam findings, treatment performed in session, or plan"
            ),
        ),
        SectionRule(
            name="Assessment",
            description="Objective physiotherapy findings and clinical assessment",
            allowed_content=(
                "Range of motion, strength, gait, balance, pain scores, functional test results, "
                "tolerance, clinician-observed movement findings, and clinician-stated assessment"
            ),
            forbidden_content=(
                "Do not include patient goals alone, future plan, or unsupported diagnosis"
            ),
        ),
        SectionRule(
            name="Treatment Plan",
            description="Planned physiotherapy approach and next steps",
            allowed_content=(
                "Home exercise plan, progression strategy, frequency, goals, follow-up, referrals, "
                "education, precautions, and planned interventions explicitly discussed"
            ),
            forbidden_content=(
                "Do not include treatments that were not actually discussed or unsupported claims"
            ),
        ),
        SectionRule(
            name="Treatment Provided",
            description="What was done during the current session",
            allowed_content=(
                "Exercises performed, manual therapy, modalities, repetitions, sets, cueing, "
                "education delivered, and interventions actually completed during the session"
            ),
            forbidden_content=(
                "Do not include future treatments or plans not performed in the current session"
            ),
        ),
        SectionRule(
            name="Patient Response",
            description="How the patient responded to treatment during the session",
            allowed_content=(
                "Tolerance, symptom response, pain change, fatigue, improvement, difficulty, "
                "engagement, adherence comments explicitly stated or observed"
            ),
            forbidden_content=(
                "Do not invent improvement or decline if not supported by the transcript"
            ),
        ),
        SectionRule(
            name="Progress / Reassessment",
            description="Changes over time and updated interpretation",
            allowed_content=(
                "Progress since prior visit, reassessment findings, change in function, "
                "change in pain, updated goals, need for modification of treatment"
            ),
            forbidden_content=(
                "Do not include unsupported conclusions or duplicate the treatment plan"
            ),
        ),
    ]
)

DAP_TEMPLATE = NoteTemplate(
    template_name="DAP",
    sections=[
        SectionRule(
            name="Data",
            description="Facts from the session: client-reported content and directly observed behavior",
            allowed_content=(
                "Client-reported thoughts, emotions, symptoms, stressors, recent events, relevant quotes, "
                "and clinician-observed behavior or mental status explicitly present in the transcript"
            ),
            forbidden_content=(
                "Do not include diagnostic conclusions, case formulation, future plan, or unsupported interpretation"
            ),
        ),
        SectionRule(
            name="Assessment",
            description="Clinical interpretation of the session",
            allowed_content=(
                "Clinician-stated interpretation, progress, barriers, symptom severity impression, risk impression, "
                "or response to interventions only if clearly supported by the session content"
            ),
            forbidden_content=(
                "Do not invent diagnoses, risk levels, or conclusions not supported by the transcript"
            ),
        ),
        SectionRule(
            name="Plan",
            description="Next steps after the session",
            allowed_content=(
                "Homework, coping strategies, follow-up timing, referrals, safety steps, treatment focus for next session, "
                "and agreed next actions explicitly discussed"
            ),
            forbidden_content=(
                "Do not include actions that were not discussed or unsupported treatment claims"
            ),
        ),
    ]
)

BIRP_TEMPLATE = NoteTemplate(
    template_name="BIRP",
    sections=[
        SectionRule(
            name="Behavior",
            description="Client presentation and session data at the start of the note",
            allowed_content=(
                "Client-reported symptoms, thoughts, emotions, stressors, recent events, risk-related statements, "
                "and directly observed behavior or mental status explicitly present in the transcript"
            ),
            forbidden_content=(
                "Do not include clinician interpretation, treatment plan, future actions, or unsupported conclusions"
            ),
        ),
        SectionRule(
            name="Intervention",
            description="What the clinician did during the session",
            allowed_content=(
                "Therapeutic techniques used, psychoeducation provided, questions asked with therapeutic purpose, "
                "coping strategies introduced, reframing, validation, grounding, CBT/DBT techniques, safety discussion, "
                "and other clinician actions explicitly reflected in the transcript"
            ),
            forbidden_content=(
                "Do not include client response as if it were the intervention. Do not include future plan."
            ),
        ),
        SectionRule(
            name="Response",
            description="How the client responded to the intervention",
            allowed_content=(
                "Client engagement, receptiveness, emotional response, insight, participation, symptom change during session, "
                "agreement/disagreement, ability to practice skills, and stated reaction to interventions explicitly supported by the transcript"
            ),
            forbidden_content=(
                "Do not invent improvement, deterioration, or insight if not clearly supported"
            ),
        ),
        SectionRule(
            name="Plan",
            description="Next steps after the session",
            allowed_content=(
                "Homework, coping practice, follow-up timing, referrals, safety plan steps, crisis instructions, "
                "treatment focus for next session, and agreed next actions explicitly discussed"
            ),
            forbidden_content=(
                "Do not include actions that were not discussed or unsupported risk conclusions"
            ),
        ),
    ]
)

TEMPLATES = {
    "SOAP": SOAP_TEMPLATE,
    "H&P": HNP_TEMPLATE,
    "physio": PHYSIO_TEMPLATE,
    "behav_SOAP": SOAP_TEMPLATE,
    "behav_DAP": DAP_TEMPLATE,
    "behav_BIRP": BIRP_TEMPLATE,
}

section_extractor = Agent[SectionRule](
    name="Section Evidence Extractor",
    instructions="""
                You are a clinical evidence extraction system.

                You will receive:
                1. A section rule
                2. A structured transcript with speaker labels and turn_ids

                Your task:
                Extract ONLY the evidence relevant to that ONE section.

                Rules:
                - Use only information explicitly supported by the transcript
                - Do not write the final note
                - Output atomic evidence items only
                - Attach source_turn_ids for every evidence item
                - Preserve negations exactly
                - Do not invent diagnoses, findings, treatments, or plans
                - Respect the section's allowed_content and forbidden_content
                - If no evidence belongs in the section, return an empty evidence list
                - Create concise evidence types such as:
                visit_reason, symptom, negated_symptom, history, medication, medication_request,
                diagnosis, observation, demographic, family_history, social_history, plan, intervention, response

                Return JSON with this exact shape:
                {
                "evidences": [
                    {
                    "evidence_id": "ev_1",
                    "type": "symptom",
                    "value": "Patient reports increased thirst.",
                    "source_turn_ids": ["t10"]
                    }
                ]
                }

                If no evidence exists, return:
                {
                "evidences": []
                }
                """,
    model="gpt-5.4",
    output_type=Evidencelist,
)

async def extract_evidence_by_template(note_type: str):
    if note_type not in TEMPLATES:
        print(f"Unsupported note_type: {note_type}")

    template = TEMPLATES[note_type]
    transcript = """{
  "turns": [
    {
      "turn_id": "t1",
      "speaker": "clinician",
      "start_ms": 0,
      "end_ms": 2499,
      "text": "Hi Mr. Patel, how have you been since your last visit?"
    },
    {
      "turn_id": "t2",
      "speaker": "patient",
      "start_ms": 2500,
      "end_ms": 5299,
      "text": "I've been okay, but my blood sugar has been a bit all over the place lately."
    },
    {
      "turn_id": "t3",
      "speaker": "clinician",
      "start_ms": 5300,
      "end_ms": 6899,
      "text": "Can you tell me more about that?"
    },
    {
      "turn_id": "t4",
      "speaker": "patient",
      "start_ms": 6900,
      "end_ms": 10599,
      "text": "Yeah, in the mornings it's usually around 180 or 190, even when I don't eat much the night before."
    },
    {
      "turn_id": "t5",
      "speaker": "clinician",
      "start_ms": 10600,
      "end_ms": 13199,
      "text": "Have you been taking your medications as prescribed?"
    },
    {
      "turn_id": "t6",
      "speaker": "patient",
      "start_ms": 13200,
      "end_ms": 17199,
      "text": "Mostly, yeah. I take the metformin twice a day, but I missed a few doses last week when I was traveling."
    },
    {
      "turn_id": "t7",
      "speaker": "clinician",
      "start_ms": 17200,
      "end_ms": 19499,
      "text": "Any changes in your diet or activity levels?"
    },
    {
      "turn_id": "t8",
      "speaker": "patient",
      "start_ms": 19500,
      "end_ms": 22799,
      "text": "I've been eating out more, especially during that trip. Probably more carbs than usual."
    },
    {
      "turn_id": "t9",
      "speaker": "clinician",
      "start_ms": 22800,
      "end_ms": 25799,
      "text": "Any symptoms like dizziness, excessive thirst, or frequent urination?"
    },
    {
      "turn_id": "t10",
      "speaker": "patient",
      "start_ms": 25800,
      "end_ms": 29099,
      "text": "I do feel thirsty a lot, and I've been going to the bathroom more often at night."
    },
    {
      "turn_id": "t11",
      "speaker": "clinician",
      "start_ms": 29100,
      "end_ms": 31599,
      "text": "How about your blood pressure readings at home?"
    },
    {
      "turn_id": "t12",
      "speaker": "patient",
      "start_ms": 31600,
      "end_ms": 34099,
      "text": "Last few times I checked, it was around 145 over 90."
    },
    {
      "turn_id": "t13",
      "speaker": "clinician",
      "start_ms": 34100,
      "end_ms": 37499,
      "text": "That's a bit higher than we'd like. Are you still taking your lisinopril?"
    },
    {
      "turn_id": "t14",
      "speaker": "patient",
      "start_ms": 37500,
      "end_ms": 39699,
      "text": "Yes, every morning. Haven't missed that one."
    },
    {
      "turn_id": "t15",
      "speaker": "clinician",
      "start_ms": 39700,
      "end_ms": 42099,
      "text": "Any side effects from your medications?"
    },
    {
      "turn_id": "t16",
      "speaker": "patient",
      "start_ms": 42100,
      "end_ms": 43699,
      "text": "No, nothing noticeable."
    },
    {
      "turn_id": "t17",
      "speaker": "clinician",
      "start_ms": 43700,
      "end_ms": 44999,
      "text": "How about exercise?"
    },
    {
      "turn_id": "t18",
      "speaker": "patient",
      "start_ms": 45000,
      "end_ms": 47499,
      "text": "Not much lately, maybe just walking once or twice a week."
    },
    {
      "turn_id": "t19",
      "speaker": "clinician",
      "start_ms": 47500,
      "end_ms": 52899,
      "text": "Okay. Based on what you're telling me, your diabetes seems less controlled recently, likely due to missed doses and dietary changes."
    },
    {
      "turn_id": "t20",
      "speaker": "patient",
      "start_ms": 52900,
      "end_ms": 53999,
      "text": "Yeah, I figured."
    },
    {
      "turn_id": "t21",
      "speaker": "clinician",
      "start_ms": 54000,
      "end_ms": 58099,
      "text": "I'd like to increase your metformin dose slightly and also refer you to a dietitian. Would that be okay?"
    },
    {
      "turn_id": "t22",
      "speaker": "patient",
      "start_ms": 58100,
      "end_ms": 59399,
      "text": "Yes, that sounds good."
    },
    {
      "turn_id": "t23",
      "speaker": "clinician",
      "start_ms": 59400,
      "end_ms": 64899,
      "text": "For your blood pressure, we may also need to adjust your medication if it stays elevated. Let's monitor it closely for the next two weeks."
    },
    {
      "turn_id": "t24",
      "speaker": "patient",
      "start_ms": 64900,
      "end_ms": 65799,
      "text": "Okay."
    },
    {
      "turn_id": "t25",
      "speaker": "clinician",
      "start_ms": 65800,
      "end_ms": 69999,
      "text": "I'll also order some blood work, including HbA1c, kidney function, and lipids."
    },
    {
      "turn_id": "t26",
      "speaker": "patient",
      "start_ms": 70000,
      "end_ms": 70799,
      "text": "Got it."
    },
    {
      "turn_id": "t27",
      "speaker": "clinician",
      "start_ms": 70800,
      "end_ms": 73699,
      "text": "Try to increase your physical activity to at least 30 minutes, five times a week."
    },
    {
      "turn_id": "t28",
      "speaker": "patient",
      "start_ms": 73700,
      "end_ms": 74999,
      "text": "I'll try my best."
    },
    {
      "turn_id": "t29",
      "speaker": "clinician",
      "start_ms": 75000,
      "end_ms": 76799,
      "text": "Great. We'll follow up in two weeks."
    }
  ]
}"""

    final = []
    for rule in template.sections:
        result = await Runner.run(section_extractor, transcript, context=rule)
        section_evidence = result.final_output
        section = {"section": rule.name}
        section.update(section_evidence)
        final.append(section)
    print(final)

transcript = """{
  "turns": [
    {
      "turn_id": "t1",
      "speaker": "clinician",
      "start_ms": 0,
      "end_ms": 2499,
      "text": "Hi Mr. Patel, how have you been since your last visit?"
    },
    {
      "turn_id": "t2",
      "speaker": "patient",
      "start_ms": 2500,
      "end_ms": 5299,
      "text": "I've been okay, but my blood sugar has been a bit all over the place lately."
    },
    {
      "turn_id": "t3",
      "speaker": "clinician",
      "start_ms": 5300,
      "end_ms": 6899,
      "text": "Can you tell me more about that?"
    },
    {
      "turn_id": "t4",
      "speaker": "patient",
      "start_ms": 6900,
      "end_ms": 10599,
      "text": "Yeah, in the mornings it's usually around 180 or 190, even when I don't eat much the night before."
    },
    {
      "turn_id": "t5",
      "speaker": "clinician",
      "start_ms": 10600,
      "end_ms": 13199,
      "text": "Have you been taking your medications as prescribed?"
    },
    {
      "turn_id": "t6",
      "speaker": "patient",
      "start_ms": 13200,
      "end_ms": 17199,
      "text": "Mostly, yeah. I take the metformin twice a day, but I missed a few doses last week when I was traveling."
    },
    {
      "turn_id": "t7",
      "speaker": "clinician",
      "start_ms": 17200,
      "end_ms": 19499,
      "text": "Any changes in your diet or activity levels?"
    },
    {
      "turn_id": "t8",
      "speaker": "patient",
      "start_ms": 19500,
      "end_ms": 22799,
      "text": "I've been eating out more, especially during that trip. Probably more carbs than usual."
    },
    {
      "turn_id": "t9",
      "speaker": "clinician",
      "start_ms": 22800,
      "end_ms": 25799,
      "text": "Any symptoms like dizziness, excessive thirst, or frequent urination?"
    },
    {
      "turn_id": "t10",
      "speaker": "patient",
      "start_ms": 25800,
      "end_ms": 29099,
      "text": "I do feel thirsty a lot, and I've been going to the bathroom more often at night."
    },
    {
      "turn_id": "t11",
      "speaker": "clinician",
      "start_ms": 29100,
      "end_ms": 31599,
      "text": "How about your blood pressure readings at home?"
    },
    {
      "turn_id": "t12",
      "speaker": "patient",
      "start_ms": 31600,
      "end_ms": 34099,
      "text": "Last few times I checked, it was around 145 over 90."
    },
    {
      "turn_id": "t13",
      "speaker": "clinician",
      "start_ms": 34100,
      "end_ms": 37499,
      "text": "That's a bit higher than we'd like. Are you still taking your lisinopril?"
    },
    {
      "turn_id": "t14",
      "speaker": "patient",
      "start_ms": 37500,
      "end_ms": 39699,
      "text": "Yes, every morning. Haven't missed that one."
    },
    {
      "turn_id": "t15",
      "speaker": "clinician",
      "start_ms": 39700,
      "end_ms": 42099,
      "text": "Any side effects from your medications?"
    },
    {
      "turn_id": "t16",
      "speaker": "patient",
      "start_ms": 42100,
      "end_ms": 43699,
      "text": "No, nothing noticeable."
    },
    {
      "turn_id": "t17",
      "speaker": "clinician",
      "start_ms": 43700,
      "end_ms": 44999,
      "text": "How about exercise?"
    },
    {
      "turn_id": "t18",
      "speaker": "patient",
      "start_ms": 45000,
      "end_ms": 47499,
      "text": "Not much lately, maybe just walking once or twice a week."
    },
    {
      "turn_id": "t19",
      "speaker": "clinician",
      "start_ms": 47500,
      "end_ms": 52899,
      "text": "Okay. Based on what you're telling me, your diabetes seems less controlled recently, likely due to missed doses and dietary changes."
    },
    {
      "turn_id": "t20",
      "speaker": "patient",
      "start_ms": 52900,
      "end_ms": 53999,
      "text": "Yeah, I figured."
    },
    {
      "turn_id": "t21",
      "speaker": "clinician",
      "start_ms": 54000,
      "end_ms": 58099,
      "text": "I'd like to increase your metformin dose slightly and also refer you to a dietitian. Would that be okay?"
    },
    {
      "turn_id": "t22",
      "speaker": "patient",
      "start_ms": 58100,
      "end_ms": 59399,
      "text": "Yes, that sounds good."
    },
    {
      "turn_id": "t23",
      "speaker": "clinician",
      "start_ms": 59400,
      "end_ms": 64899,
      "text": "For your blood pressure, we may also need to adjust your medication if it stays elevated. Let's monitor it closely for the next two weeks."
    },
    {
      "turn_id": "t24",
      "speaker": "patient",
      "start_ms": 64900,
      "end_ms": 65799,
      "text": "Okay."
    },
    {
      "turn_id": "t25",
      "speaker": "clinician",
      "start_ms": 65800,
      "end_ms": 69999,
      "text": "I'll also order some blood work, including HbA1c, kidney function, and lipids."
    },
    {
      "turn_id": "t26",
      "speaker": "patient",
      "start_ms": 70000,
      "end_ms": 70799,
      "text": "Got it."
    },
    {
      "turn_id": "t27",
      "speaker": "clinician",
      "start_ms": 70800,
      "end_ms": 73699,
      "text": "Try to increase your physical activity to at least 30 minutes, five times a week."
    },
    {
      "turn_id": "t28",
      "speaker": "patient",
      "start_ms": 73700,
      "end_ms": 74999,
      "text": "I'll try my best."
    },
    {
      "turn_id": "t29",
      "speaker": "clinician",
      "start_ms": 75000,
      "end_ms": 76799,
      "text": "Great. We'll follow up in two weeks."
    }
  ]
}"""

rule = SectionRule(
            name="Objective",
            description="Clinician observations and measurable findings",
            allowed_content="Only clinician-observed findings, exam findings, vitals, measurable data explicitly stated in transcript",
            forbidden_content="Do not include patient feelings, opinions, or inferred diagnoses",
        )


async def main():
    result = await Runner.run(section_extractor, input=transcript, context=rule)
    print(result.final_output)

asyncio.run(main())