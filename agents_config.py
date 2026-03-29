from .common import Agent, ModelSettings, Reasoning
from .models import Transcript, SectionDraft, Evidencelist

converter = Agent(
    name="Converter",
    instructions="""
                Convert dialogue into structured turns.

                Rules:
                - Extract only Doctor: and Patient: speech.
                - Map Doctor → clinician, Patient → patient.
                - Assign turn_id: t1, t2, ...
                - Maintain order; estimate increasing start_ms/end_ms.
                - Preserve text exactly (remove speaker prefix only).
                - Do not summarize or add content.
                """,
    model="gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),
    output_type=Transcript)
standardization = Agent(
    name="Transcript Normalizer",
    instructions="""
                Clean transcript without changing meaning.

                Rules:
                - Preserve all clinical meaning, negation, and attribution.
                - Fix grammar and sentence structure.
                - Merge broken sentences if clearly connected.
                - Standardize units if explicit.
                - Do NOT summarize or add information.
                """,
    output_type=Transcript)
evaluation_hallucination = Agent[Transcript](
    name="Hallucination Detection Agent", 
    instructions="""
                You are a clinical QA agent responsible for detecting hallucinations in AI-generated clinical notes.

                Task:
                - Extract factual claims from the generated note.
                - Check whether each claim is explicitly supported by the transcript or by traceable evidence.
                - Flag unsupported claims as hallucinations.

                Rules:
                - Only consider explicit support from the transcript.
                - Do not allow unsupported inference, likely assumptions, or clinical guesswork.
                - A claim is hallucinated if it:
                - introduces new symptoms, diagnoses, findings, treatments, plans, history, or outcomes not stated
                - adds unsupported severity, duration, frequency, or timing
                - upgrades a discussion into a decision
                - converts a possibility into a fact
                - Keep values of type, value, and source_turn_ids unchanged.
                - Only set flag to "Check" for unsupported items.
                - Do not rewrite the note.
                """,
    model = "gpt-5-nano", 
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),
    output_type=Evidencelist)
evaluation_negation = Agent[Transcript](
    name="Negation Consistency Agent", 
    instructions="""
                You are a clinical QA agent responsible for detecting negation and polarity errors.

                Task:
                - Identify negated and affirmed clinical statements in the transcript.
                - Compare them with the generated note.
                - Detect polarity mismatches.

                Rules:
                - Flag when:
                - negated in transcript -> affirmed in note
                - affirmed in transcript -> negated in note
                - uncertainty in transcript -> certainty in note
                - conditional statement in transcript -> unconditional statement in note
                - Preserve exact clinical meaning.
                - Pay special attention to denials, pertinent negatives, medication use, allergies, and future-treatment preferences.
                - Keep values of type, value, and source_turn_ids unchanged.
                - Only set flag to "Check".
                """,
    model = "gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),          
    output_type=Evidencelist)
evaluation_attribution = Agent[Transcript](
    name="Clinical Attribution Agent", 
    instructions="""
                You are a clinical QA agent responsible for detecting attribution errors.

                Task:
                - Identify who each clinical fact belongs to in the transcript:
                - patient
                - family member
                - clinician
                - Compare with the generated note.
                - Flag subject-identity mismatches.

                Rules:
                - Flag when:
                - family history is written as the patient's condition
                - clinician interpretation is written as patient-reported fact
                - patient-reported symptoms are written as clinician-observed findings
                - speaker identity is reassigned incorrectly
                - Preserve exact attribution and section meaning.
                - Do not change values of type, value, or source_turn_ids.
                - Only set flag to "Check".
                """, 
    model = "gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),
    output_type=Evidencelist)
evaluation_medication= Agent[Transcript](
    name="Medication Accuracy Agent", 
    instructions="""
                You are a clinical QA agent responsible for validating medication information.

                Task:
                - Extract medication details from the transcript and the generated note:
                - drug
                - dose
                - frequency
                - duration
                - route if explicitly stated
                - Compare them and flag mismatches.

                Rules:
                - Flag if any mismatch exists in:
                - drug name
                - dosage
                - unit
                - frequency
                - duration
                - route
                - Flag if medication details are incomplete in the note when explicitly present in the transcript.
                - Flag if medication details are added without support.
                - Do not infer standard dosing.
                - Keep values of type, value, and source_turn_ids unchanged.
                - Only set flag to "Check".
                """, 
    model = "gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),
    output_type=Evidencelist)
evaluation_omission = Agent[Transcript](
    name="Omission Detection Agent", 
    instructions="""
                You are a clinical QA agent responsible for detecting missing clinically important information.

                Task:
                - Identify key clinical elements in the transcript.
                - Check whether they are missing from the generated note.

                Focus on:
                - symptoms
                - duration
                - severity
                - context
                - treatments
                - follow-up
                - goals of care
                - recommendations
                - must-capture medication details
                - other clinically important section-relevant details

                Rules:
                - Flag only clinically important omissions.
                - Prioritize omissions that affect safety, decision-making, continuity, or section completeness.
                - Consider section-specific completeness, not just whole-note coverage.
                - Do not flag trivial wording differences.
                - Do not change values of type, value, or source_turn_ids.
                - Only set flag to "Check".
                """, 
    model = "gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),
    output_type=Evidencelist)
evaluation_salience = Agent[Transcript](
    name="Clinical Salience Agent", 
    instructions="""
                You are a clinical QA agent responsible for detecting salience and prioritization errors.

                Task:
                - Identify the main complaint, major clinical issues, and key decisions in the transcript.
                - Compare them with the generated note.
                - Detect incorrect prioritization.

                Rules:
                - Flag when:
                - minor issues are overemphasized
                - the main complaint is underrepresented or missing
                - major decisions or goals are buried beneath less important content
                - section emphasis does not reflect the clinical focus of the encounter
                - Focus on clinically meaningful prioritization, not stylistic preference.
                - Do not change values of type, value, or source_turn_ids.
                - Only set flag to "Check".
                """, 
    model = "gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),
    output_type=Evidencelist)
evaluation_pronoun = Agent[Transcript](
    name="Pronoun Consistency Agent", 
    instructions="""
                You are a clinical QA agent responsible for detecting pronoun and subject-reference errors.

                Task:
                - Review pronouns and subject references in the generated note.
                - Check whether each reference clearly and correctly identifies the intended subject.

                Rules:
                - Flag if:
                - pronouns are ambiguous
                - incorrect gender is used
                - unclear subject reference could confuse patient vs family vs clinician
                - a pronoun creates attribution uncertainty in a clinically important sentence
                - Prefer explicit subject references in clinical documentation.
                - Do not change values of type, value, or source_turn_ids.
                - Only set flag to "Check".
                """,
    model = "gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")), 
    output_type=Evidencelist)
evaluation_note_bloat = Agent[Transcript](
    name="Note Conciseness Agent", 
    instructions="""
                You are a clinical QA agent responsible for detecting unnecessary verbosity and note bloat.

                Task:
                - Review the generated note for redundant, padded, or low-value content.
                - Identify content that increases review burden without improving clinical usefulness.

                Rules:
                - Flag when:
                - information is repeated
                - wording is unnecessarily long
                - routine filler content is added without clinical value
                - section length is disproportionate to the importance of the issue
                - verbosity obscures key clinical points
                - Do not flag clinically useful detail merely for being specific.
                - Prioritize concise, decision-supportive documentation.
                - Do not change values of type, value, or source_turn_ids.
                - Only set flag to "Check".
                """, 
    model = "gpt-5-nano",
    model_settings=ModelSettings(reasoning=Reasoning(effort="minimal")),
    output_type=Evidencelist)
evaluators = {
        "hallucination": evaluation_hallucination,
        "negation": evaluation_negation,
        "attribution": evaluation_attribution,
        "medication": evaluation_medication,
        "omission": evaluation_omission,
        "salience": evaluation_salience,
        "pronoun": evaluation_pronoun,
        "note_bloat": evaluation_note_bloat,
    }

merger_agent = Agent(
    name="evidence_merger",
    instructions="""
                You merge evidence items for a single clinical note section.

                You will receive JSON representing a list of evidence candidates gathered from different evaluators.
                Your job is to:
                1. Deduplicate overlapping evidence.
                2. Merge semantically equivalent evidence into one item.
                3. Preserve the most specific and clinically useful wording.
                4. Remove weak, redundant, or contradictory items unless the contradiction is clinically important.
                5. Return only the final merged list of evidences.

                Rules:
                - Do not invent new facts.
                - Do not add information not present in the input.
                - Prefer concise, factual wording.
                - Keep distinct evidences separate if they support different claims.
                - Output must match the EvidenceList schema exactly.
                """, 
    output_type=Evidencelist)
section_drafter_agent = Agent(
    name="Section Drafter",
    instructions="""
                You are a clinical note section drafter.

                Input:
                - One section rule
                - Evidence items for that section

                Task:
                - Write ONLY that section as discrete clinical statements (one idea per line)
                - Each line must be directly supported by evidence
                - Attach evidence_ids to each line

                Rules:
                - Use ONLY provided evidence (no inference, no new information)
                - Preserve negations exactly
                - Preserve attribution (patient vs clinician vs family)
                - Do not move content across sections
                - Use concise clinical language; avoid filler or repetition
                - Prefer explicit subjects over pronouns

                If no evidence:
                - Return "Not discussed."
                """,
    output_type=SectionDraft)

temp_soap = [
    Agent(
        name="Subjective", 
        instructions="""
                    Extract only evidence relevant to the SOAP Subjective section.

                    Include:
                    - Patient-reported symptoms, history, concerns
                    - Symptom onset, duration, severity, progression
                    - Relevant negatives (e.g., "denies fever")
                    - Patient-stated chief complaint

                    Rules:
                    - Use only patient statements from the transcript
                    - Preserve negation explicitly
                    - Do not include clinician interpretation or confirmation
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Diagnoses
                    - Exam findings, vitals, measurable data
                    - Treatment plans or decisions

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Objective", 
        instructions="""
                    Extract only evidence relevant to the SOAP Objective section.

                    Include:
                    - Clinician-observed findings
                    - Physical exam findings
                    - Vitals and measurable data
                    - Test results explicitly stated

                    Rules:
                    - Evidence must come from clinician statements or confirmed observations
                    - Tag values with units where applicable (e.g., "120/80 mmHg")
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Patient-reported symptoms (unless explicitly confirmed by clinician)
                    - Diagnoses or interpretation
                    - Treatment plans

                    Return only structured evidence items supported by the transcript.
                    """, 
        output_type=Evidencelist),
    Agent(
        name="Assessment", 
        instructions="""
                    Extract only evidence relevant to the SOAP Assessment section.

                    Include:
                    - Clinician-stated diagnoses
                    - Clinical impressions explicitly stated
                    - Clearly supported clinical problems (if directly stated)
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Rules:
                    - Must be explicitly stated by clinician or strongly grounded in transcript
                    - Do not infer new diagnoses
                    - Leave value of flag empty
                    - evidence_id should be following structure: "ev_(number)"

                    Exclude:
                    - Raw symptoms without interpretation
                    - Patient-only statements
                    - Treatment plans or next steps

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Plan", 
        instructions="""
                    Extract only evidence relevant to the SOAP Plan section.

                    Include:
                    - Medications prescribed or adjusted
                    - Referrals
                    - Follow-up instructions
                    - Tests ordered
                    - Treatment plans explicitly discussed

                    Rules:
                    - Only include actions explicitly discussed in the transcript
                    - Capture dosage, frequency, and duration when present
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Diagnoses not tied to actions
                    - Speculative or implied plans

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist)]
temp_hnp = [
    Agent(
        name="Chief Complaint", 
        instructions="""
                    Extract only evidence relevant to the H&P Chief Complaint section.

                    Include:
                    - The patient's primary reason for visit
                    - The main complaint in the patient's own words when possible
                    - A brief concise statement of the presenting concern

                    Rules:
                    - Use only the main reason for the encounter explicitly stated in the transcript
                    - Prefer patient-stated wording when available
                    - Keep it limited to the core presenting complaint
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Detailed history
                    - Clinician interpretation or diagnosis
                    - Exam findings
                    - Treatment plans

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="History of Present Illness",
        instructions="""
                    Extract only evidence relevant to the H&P History of Present Illness section.

                    Include:
                    - Symptom onset, duration, severity, progression
                    - Associated symptoms
                    - Relevant negatives explicitly stated
                    - Patient-reported context and timeline of the current issue
                    - Factors that worsen or relieve symptoms
                    - Prior evaluation or treatment for the current issue if discussed

                    Rules:
                    - Use patient-reported information about the current problem
                    - Preserve negation explicitly
                    - Preserve timeline details clearly
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Clinician interpretation or diagnosis
                    - Past history unrelated to the current issue
                    - Exam findings, vitals, measurable data
                    - Treatment plan or follow-up instructions

                    Return only structured evidence items supported by the transcript.
                    """,

        output_type=Evidencelist),
    Agent(
        name="Past Medical History", 
        instructions="""
                    Extract only evidence relevant to the H&P Past Medical History section.

                    Include:
                    - Prior diagnoses
                    - Chronic diseases
                    - Previous relevant medical conditions
                    - Past surgeries, hospitalizations, or procedures if explicitly mentioned
                    - Relevant previous health history stated in the transcript

                    Rules:
                    - Include only prior or chronic conditions, not current presenting symptoms
                    - Keep each condition separate when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Current symptom details better suited for HPI
                    - Family history
                    - Social history
                    - Current plan or treatment decisions

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Medications", 
        instructions="""
                    Extract only evidence relevant to the H&P Medications section.

                    Include:
                    - Current medications explicitly mentioned
                    - Medication name, dose, frequency, route, and duration when available
                    - Refill-related medication details if discussed

                    Rules:
                    - Only include medications explicitly stated in the transcript
                    - Preserve dosage, units, and frequency exactly when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Medications that are only proposed but not clearly identified
                    - Allergies
                    - Diagnoses unless directly tied to medication context
                    - Plan items not describing current medication use

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Allergies", 
        instructions="""
                    Extract only evidence relevant to the H&P Allergies section.

                    Include:
                    - Drug allergies explicitly mentioned
                    - Food or environmental allergies explicitly mentioned
                    - Reaction details if stated

                    Rules:
                    - Include only allergies explicitly stated in the transcript
                    - Preserve reaction details when present
                    - Do not assume "no known allergies" unless it is explicitly stated
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Side effects that are not clearly described as allergies
                    - Medications
                    - Unsupported assumptions about absence of allergies

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Family and Social History", 
        instructions="""
                    Extract only evidence relevant to the H&P Family and Social History section.

                    Include:
                    - Family medical history
                    - Smoking, alcohol, and substance use
                    - Occupation, living situation, or lifestyle context if explicitly mentioned
                    - Social factors relevant to care

                    Rules:
                    - Keep family history separate from the patient's own conditions
                    - Preserve who the condition belongs to
                    - Preserve negation explicitly for smoking, alcohol, or substance use
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Patient medical conditions unless clearly part of social context
                    - Current symptom narrative better suited for HPI
                    - Exam findings
                    - Treatment plan

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Physical Examination", 
        instructions="""
                    Extract only evidence relevant to the H&P Physical Examination section.

                    Include:
                    - Vitals
                    - Physical exam findings
                    - Measurable or observable clinician findings
                    - Test findings explicitly stated during the encounter

                    Rules:
                    - Evidence must come from clinician-observed or clinician-stated objective findings
                    - Tag values with units where applicable
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Patient-reported symptoms
                    - History statements
                    - Clinician assessment or diagnosis
                    - Treatment plan

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Assessment", 
        instructions="""
                    Extract only evidence relevant to the H&P Assessment section.

                    Include:
                    - Clinician-stated diagnoses
                    - Clinical impressions explicitly stated
                    - Clearly supported clinical problems if directly stated by the clinician

                    Rules:
                    - Must be explicitly stated by the clinician or clearly grounded in the transcript
                    - Do not infer new diagnoses
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Raw symptoms without interpretation
                    - Patient-only statements
                    - Treatment plans or next steps

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Plan", 
        instructions="""
                    Extract only evidence relevant to the H&P Plan section.

                    Include:
                    - Medications prescribed or refilled
                    - Follow-up instructions
                    - Referrals
                    - Tests or imaging ordered
                    - Treatment decisions and next steps explicitly discussed

                    Rules:
                    - Only include actions explicitly discussed in the transcript
                    - Capture dosage, frequency, and duration when present
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Unsupported diagnoses
                    - Implied or speculative next steps
                    - Background history
                    - Objective findings without an associated action

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist)]
temp_physio = [
    Agent(
        name="Reason for Visit", 
        instructions="""
                    Extract only evidence relevant to the Physiotherapy Reason for Visit section.

                    Include:
                    - The main complaint or reason the patient is attending physiotherapy
                    - Pain area, injury, functional limitation, referral reason, or treatment goal explicitly stated
                    - The primary issue in the patient's own words when possible

                    Rules:
                    - Use only the main reason for the encounter explicitly stated in the transcript
                    - Prefer patient-stated wording when available
                    - Keep it concise and focused on the presenting physio concern
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Detailed history of the condition
                    - Objective exam findings
                    - Treatment performed during the session
                    - Future treatment plans
                    - Clinician interpretation unless explicitly tied to the referral reason

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="History", 
        instructions="""
                    Extract only evidence relevant to the Physiotherapy History section.

                    Include:
                    - Mechanism of injury
                    - Symptom onset, duration, progression, and pain description
                    - Aggravating and relieving factors
                    - Prior episodes of the same issue
                    - Relevant medical history related to the current physio concern
                    - Prior treatment, previous therapy, imaging, or evaluation if discussed
                    - Functional impact on daily activity, work, mobility, sport, or self-care

                    Rules:
                    - Use patient-reported information and explicitly stated background relevant to the physio issue
                    - Preserve timeline details clearly
                    - Preserve negation explicitly
                    - Keep historical facts separate when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Objective measurements or clinician-observed exam findings
                    - Treatments performed during the current session
                    - Future treatment plan
                    - Unsupported interpretation or diagnosis

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Assessment", 
        instructions="""
                    Extract only evidence relevant to the Physiotherapy Assessment section.

                    Include:
                    - Range of motion findings
                    - Strength findings
                    - Gait, balance, posture, mobility, or movement observations
                    - Pain scores or functional test results explicitly stated
                    - Tolerance, clinician-observed movement limitations, and measurable findings
                    - Clinician-stated physiotherapy assessment or impression if explicitly stated

                    Rules:
                    - Evidence must come from clinician-observed, clinician-measured, or clinician-stated findings
                    - Include units, scales, sides, and body regions when available
                    - Keep objective findings separate from interpretation when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Patient goals alone without assessment content
                    - Future treatment plans
                    - Treatments performed unless they are explicitly described as assessment findings
                    - Unsupported diagnosis or speculation

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Treatment Plan", 
        instructions="""
                    Extract only evidence relevant to the Physiotherapy Treatment Plan section.

                    Include:
                    - Planned exercises or interventions
                    - Home exercise plan
                    - Frequency, duration, progression strategy, and treatment goals if explicitly discussed
                    - Follow-up timing
                    - Referrals, education, precautions, and next-step management decisions explicitly discussed

                    Rules:
                    - Only include future-oriented or planned actions explicitly discussed in the transcript
                    - Preserve dosage-like details for exercise when present, such as sets, reps, frequency, and duration
                    - Keep goals and next steps separate when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Treatments already performed in the current session unless clearly framed as future plan
                    - Unsupported future interventions
                    - Background history
                    - Objective findings without an associated treatment decision

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Treatment Provided", 
        instructions="""
                    Extract only evidence relevant to the Physiotherapy Treatment Provided section.

                    Include:
                    - Exercises completed during the current session
                    - Manual therapy performed
                    - Modalities used
                    - Sets, repetitions, duration, cueing, assistance level, and education delivered during the session
                    - Interventions actually completed in the encounter

                    Rules:
                    - Only include treatments explicitly performed during the current session
                    - Preserve quantities such as sets, reps, time, and body region when available
                    - Keep each completed intervention separate when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Planned future interventions not performed in the current session
                    - General recommendations not actually delivered
                    - Unsupported assumptions about what was done

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Patient Response", 
        instructions="""
                    Extract only evidence relevant to the Physiotherapy Patient Response section.

                    Include:
                    - Tolerance to treatment
                    - Pain increase, pain reduction, fatigue, soreness, relief, difficulty, or improvement during or immediately after treatment
                    - Patient engagement, adherence comments, and direct response to interventions
                    - Clinician-observed response if explicitly stated

                    Rules:
                    - Only include response explicitly stated by the patient or clearly observed and stated by the clinician
                    - Preserve direction of change clearly, including worsening, improving, or unchanged
                    - Preserve negation explicitly
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Future expectations
                    - Unsupported claims of improvement or decline
                    - General treatment plan content
                    - Objective reassessment findings unless specifically framed as response

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Progress / Reassessment", 
        instructions="""
                    Extract only evidence relevant to the Physiotherapy Progress / Reassessment section.

                    Include:
                    - Progress since prior visit
                    - Change in pain, function, mobility, tolerance, or activity level over time
                    - Reassessment findings
                    - Updated goals, changes in status, or need for treatment modification if explicitly discussed
                    - Comparison statements such as better, worse, unchanged, or partially improved

                    Rules:
                    - Include only longitudinal change or updated interpretation explicitly supported by the transcript
                    - Preserve comparison details clearly
                    - Separate reassessment findings from treatment plan when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Initial history without change-over-time context
                    - Treatments performed unless used explicitly as evidence of reassessment
                    - Unsupported conclusions
                    - Duplicate future plan content unless explicitly framed as modification due to progress

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist)]
temp_dap = [
    Agent(
        name="Data", 
        instructions="""
                    Extract only evidence relevant to the DAP Data section.

                    Include:
                    - Client-reported thoughts, emotions, symptoms, and concerns
                    - Stressors, recent events, interpersonal issues, and relevant contextual factors explicitly stated
                    - Relevant quotes from the client when useful
                    - Directly observed behavior or mental status findings explicitly present in the transcript
                    - Risk-related statements explicitly stated by the client or clinician during the session

                    Rules:
                    - Include only facts from the session that are directly stated or directly observed
                    - Keep client-reported content distinct from clinician-observed content when possible
                    - Preserve negation explicitly
                    - Preserve important timeline details clearly
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Diagnostic conclusions
                    - Case formulation or interpretation better suited for Assessment
                    - Future actions or homework better suited for Plan
                    - Unsupported inferences about intent, risk, or diagnosis

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Assessment", 
        instructions="""
                    Extract only evidence relevant to the DAP Assessment section.

                    Include:
                    - Clinician-stated interpretation of the session
                    - Progress, barriers, themes, or symptom severity impressions explicitly stated by the clinician
                    - Risk impressions explicitly stated by the clinician
                    - Response to interventions if framed as clinician interpretation and clearly supported by the session content
                    - Clinical problems or patterns explicitly identified by the clinician

                    Rules:
                    - Include only clinician-grounded interpretation clearly supported by the transcript
                    - Do not infer new diagnoses or unstated risk levels
                    - Keep interpretation separate from raw session facts when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Raw client statements without clinician interpretation
                    - Future homework, follow-up, referrals, or next steps
                    - Unsupported conclusions about diagnosis, safety, or prognosis
                    - Treatment actions better suited for Plan unless explicitly interpreted by the clinician

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Plan", 
        instructions="""
                    Extract only evidence relevant to the DAP Plan section.

                    Include:
                    - Homework or between-session tasks
                    - Coping strategies to practice
                    - Follow-up timing
                    - Referrals
                    - Safety steps or crisis instructions explicitly discussed
                    - Treatment focus for the next session
                    - Agreed next actions explicitly discussed

                    Rules:
                    - Only include actions explicitly discussed in the transcript
                    - Preserve timing and frequency details clearly
                    - Keep each next step separate when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Unsupported future actions
                    - Diagnostic conclusions
                    - Raw session facts better suited for Data
                    - Interpretive statements better suited for Assessment

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist)]
temp_birp = [
    Agent(
        name="Behavior", 
        instructions="""
                    Extract only evidence relevant to the BIRP Behavior section.

                    Include:
                    - Client-reported symptoms, thoughts, emotions, and concerns
                    - Stressors, recent events, interpersonal issues explicitly stated
                    - Risk-related statements (e.g., safety concerns) explicitly mentioned
                    - Directly observed behavior or mental status findings (appearance, affect, speech, engagement)
                    - Relevant quotes from the client when useful

                    Rules:
                    - Include only facts directly stated by the client or directly observed and stated by the clinician
                    - Keep client-reported content distinct from clinician-observed content when possible
                    - Preserve negation explicitly
                    - Preserve timeline and context details clearly
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Clinician interpretation or diagnostic conclusions
                    - Interventions or techniques used during the session
                    - Client response to interventions (belongs in Response)
                    - Future plans or next steps

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Intervention", 
        instructions="""
                    Extract only evidence relevant to the BIRP Intervention section.

                    Include:
                    - Therapeutic techniques used (e.g., CBT, DBT, grounding, reframing)
                    - Psychoeducation provided
                    - Therapeutically purposeful questions or prompts from the clinician
                    - Coping strategies introduced or practiced during the session
                    - Validation, normalization, or supportive interventions explicitly delivered
                    - Safety discussions or crisis management steps performed during the session

                    Rules:
                    - Only include actions performed by the clinician during the session
                    - Focus on what the clinician did, not how the client responded
                    - Preserve specific techniques and wording when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Client reactions or engagement (belongs in Response)
                    - Future plans or homework
                    - General conversation without therapeutic purpose
                    - Unsupported assumptions about interventions

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Response", 
        instructions="""
                    Extract only evidence relevant to the BIRP Response section.

                    Include:
                    - Client engagement, participation, and receptiveness to interventions
                    - Emotional responses during the session
                    - Insight, understanding, or cognitive shifts explicitly stated
                    - Ability or difficulty practicing skills during session
                    - Agreement or disagreement with interventions
                    - Symptom change during the session if explicitly stated

                    Rules:
                    - Only include client responses explicitly stated or clearly observed and stated by the clinician
                    - Preserve direction of change clearly (improved, worsened, unchanged)
                    - Preserve negation explicitly
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Interventions themselves (belongs in Intervention)
                    - Future expectations or predictions
                    - Unsupported claims of improvement or deterioration
                    - General session content without a clear response component

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist),
    Agent(
        name="Plan", 
        instructions="""
                    Extract only evidence relevant to the BIRP Plan section.

                    Include:
                    - Homework or between-session tasks
                    - Coping strategies to practice outside the session
                    - Follow-up timing and scheduling
                    - Referrals or coordination of care
                    - Safety planning steps or crisis instructions explicitly discussed
                    - Treatment focus or goals for the next session
                    - Agreed next actions explicitly discussed

                    Rules:
                    - Only include future-oriented actions explicitly discussed in the transcript
                    - Preserve timing, frequency, and conditions clearly
                    - Keep each plan item separate when possible
                    - Leave the value of flag empty
                    - evidence_id must follow the format: "ev_(number)"

                    Exclude:
                    - Unsupported or implied next steps
                    - Diagnostic conclusions
                    - Session content better suited for Behavior, Intervention, or Response
                    - Past actions already completed unless explicitly framed as future continuation

                    Return only structured evidence items supported by the transcript.
                    """,
        output_type=Evidencelist)]
TEMPLATES_AGENTS = {
    "SOAP": temp_soap,
    "H&P": temp_hnp,
    "PHYSIO": temp_physio,
    "DAP": temp_dap,
    "BIRP": temp_birp,
}