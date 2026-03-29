import asyncio
import time

from pydantic import BaseModel
from openai import OpenAI
from openai.types.shared import Reasoning

from agents import Agent, ModelSettings, Runner
from agents.model_settings import _merge_retry_settings