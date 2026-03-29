from .models import NoteTemplate, Section


SOAP_TEMPLATE = NoteTemplate(
    template_name="SOAP",
    sections=[
        Section(
            name="Subjective"),
        Section(
            name="Objective"),
        Section(
            name="Assessment"),
        Section(
            name="Plan")])
HNP_TEMPLATE = NoteTemplate(
    template_name="H&P",
    sections=[
        Section(
            name="Chief Complaint"),
        Section(
            name="History of Present Illness"),
        Section(
            name="Past Medical History"),
        Section(
            name="Medications"),
        Section(
            name="Allergies"),
        Section(
            name="Family and Social History"),
        Section(
            name="Physical Examination"),
        Section(
            name="Assessment"),
        Section(
            name="Plan")])
PHYSIO_TEMPLATE = NoteTemplate(
    template_name="Physiotherapy",
    sections=[
        Section(
            name="Reason for Visit"),
        Section(
            name="History"),
        Section(
            name="Assessment"),
        Section(
            name="Treatment Plan"),
        Section(
            name="Treatment Provided"),
        Section(
            name="Patient Response"),
        Section(
            name="Progress / Reassessment")])
DAP_TEMPLATE = NoteTemplate(
    template_name="DAP",
    sections=[
        Section(
            name="Data"),
        Section(
            name="Assessment"),
        Section(
            name="Plan")])
BIRP_TEMPLATE = NoteTemplate(
    template_name="BIRP",
    sections=[
        Section(
            name="Behavior"),
        Section(
            name="Intervention"),
        Section(
            name="Response"),
        Section(
            name="Plan")])

TEMPLATES = {
    "SOAP": SOAP_TEMPLATE,
    "H&P": HNP_TEMPLATE,
    "PHYSIO": PHYSIO_TEMPLATE,
    "DAP": DAP_TEMPLATE,
    "BIRP": BIRP_TEMPLATE,
}