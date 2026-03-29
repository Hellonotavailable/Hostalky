from .common import BaseModel

class TranscriptTurn(BaseModel):
    turn_id: str
    speaker: str 
    text: str

class Transcript(BaseModel):
    turns: list[TranscriptTurn]

class Section(BaseModel):
    name: str

class NoteTemplate(BaseModel):
    template_name: str
    sections: list[Section]

class Evidence(BaseModel):
    evidence_id: str
    type: str
    value: str
    source_turn_ids: list[str]
    flag: str

class Evidencelist(BaseModel):
    evidences: list[Evidence]

class SectionDraft(BaseModel):
    section: str
    draft_text: str
    source_turn_ids: list[str]