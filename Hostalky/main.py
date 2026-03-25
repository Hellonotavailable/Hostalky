import asyncio
from pydantic import BaseModel
from agents import Agent, Runner
from openai import OpenAI

client = OpenAI()
# Example conversaiton
text = """
Clinician: Hi Mr. Patel, how have you been since your last visit?
Patient: I've been okay, but my blood sugar has been a bit all over the place lately.

Clinician: Can you tell me more about that?
Patient: Yeah, in the mornings it's usually around 180 or 190, even when I don't eat much the night before.

Clinician: Have you been taking your medications as prescribed?
Patient: Mostly, yeah. I take the metformin twice a day, but I missed a few doses last week when I was traveling.

Clinician: Any changes in your diet or activity levels?
Patient: I've been eating out more, especially during that trip. Probably more carbs than usual.

Clinician: Any symptoms like dizziness, excessive thirst, or frequent urination?
Patient: I do feel thirsty a lot, and I've been going to the bathroom more often at night.

Clinician: How about your blood pressure readings at home?
Patient: Last few times I checked, it was around 145 over 90.

Clinician: That's a bit higher than we'd like. Are you still taking your lisinopril?
Patient: Yes, every morning. Haven't missed that one.

Clinician: Any side effects from your medications?
Patient: No, nothing noticeable.

Clinician: How about exercise?
Patient: Not much lately, maybe just walking once or twice a week.

Clinician: Okay. Based on what you're telling me, your diabetes seems less controlled recently, likely due to missed doses and dietary changes.

Patient: Yeah, I figured.

Clinician: I'd like to increase your metformin dose slightly and also refer you to a dietitian. Would that be okay?
Patient: Yes, that sounds good.

Clinician: For your blood pressure, we may also need to adjust your medication if it stays elevated. Let's monitor it closely for the next two weeks.

Patient: Okay.

Clinician: I'll also order some blood work, including HbA1c, kidney function, and lipids.

Patient: Got it.

Clinician: Try to increase your physical activity to at least 30 minutes, five times a week.

Patient: I'll try my best.

Clinician: Great. We'll follow up in two weeks.
"""


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
    Evidences: list[Evidence]

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

converter = Agent(
    name="Converter to JSON",
    instructions="""
                You are a clinical transcription processor.

                Convert the given dialogue into structured speaker turns.

                Ignore metadata or summary text that is not part of the spoken dialogue, such as:
                - dataset headers
                - codes like 0,GENHX
                - narrative summaries before the actual conversation

                Only extract the actual conversation turns that begin with Doctor: or Patient:.

                For each turn:
                - Assign a unique turn_id in order: t1, t2, t3, ...
                - Map Doctor to "clinician"
                - Map Patient to "patient"
                - Estimate start_ms and end_ms in increasing sequential order
                - Preserve the spoken text exactly, but remove the speaker prefix itself

                Return all turns in the Transcript schema as {"turns": [...]}.
                """,
    model="gpt-5.4",
    output_type=Transcript
)

standardization = Agent(
    name="Transcript Normalizer",
    instructions="""
                You are a clinical transcript normalizer.

                Your task is to clean and standardize a medical transcript WITHOUT changing its meaning.

                Rules:
                - Do NOT summarize or remove medical information
                - Preserve all clinical meaning exactly
                - Remove filler words only if they do not affect meaning
                - Fix grammar and sentence structure
                - Merge broken sentences
                - Standardize units (mg, ml, etc.)
                - Do NOT add new information

                return ONLY the cleaned sentence.
                """
)

section_extractor = Agent[SectionRule](
    name="Section Evidence Extractor",
    instructions="""
                You are a clinical transcript normalizer.

                Your task is to clean and standardize a medical transcript WITHOUT changing its meaning.

                Rules:
                - Do NOT summarize or remove medical information
                - Preserve all clinical meaning exactly
                - Remove filler words only if they do not affect meaning
                - Fix grammar and sentence structure
                - Merge broken sentences
                - Standardize units (mg, ml, etc.)
                - Do NOT add new information

                return ONLY the cleaned sentence.
                """,
    output_type=Evidencelist,
)

async def normalizer():
    result = await Runner.run(converter, text)
    normalized_turns = []

    for turn in result.final_output.turns:
        line = turn.text

        cleaned_result = await Runner.run(standardization, line)

        normalized_turns.append({
            "turn_id": turn.turn_id,
            "speaker": turn.speaker,
            "start_ms": turn.start_ms,
            "end_ms": turn.end_ms,
            "text": cleaned_result.final_output,
        })

    return normalized_turns

async def extract_evidence_by_template(note_type: str) -> list:
    if note_type not in TEMPLATES:
        print(f"Unsupported note_type: {note_type}")

    template = TEMPLATES[note_type]
    transcript = await normalizer()

    final = []
    for rule in template.sections:
        result = await Runner.run(section_extractor, transcript, context=rule)
        section_evidence = result.final_output
        section = {"section": rule.name}
        section.update(section_evidence)
        final.append(section)
    print(final)

asyncio.run(extract_evidence_by_template("SOAP"))
