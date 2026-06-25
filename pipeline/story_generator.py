import torch
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import DEVICE
from utils.logger import get_logger
from utils.exceptions import StoryGenerationError

logger = get_logger(__name__)

class StoryGenerator:
    def __init__(self, language: str = "Albanian"):
        self.language = language
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        logger.info(f"Initializing StoryGenerator inference layers targeting engine: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )

            self.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype,
                device_map="auto" if DEVICE == "cuda" else "cpu",
                trust_remote_code=True
            )
            
            logger.success(f"Story generation LLM subsystem successfully loaded onto compute target device: {DEVICE.upper()}")
            
        except Exception as e:
            raise StoryGenerationError(f"Critical execution fault loading fundamental LLM system architectures: {str(e)}")

    def generate(
        self, 
        name: str, 
        birthday: Any, 
        gender: Optional[str] = None, 
        preferences: Optional[Dict[str, Any]] = None
    ) -> dict:
        user_prefs = preferences if preferences else {}
        gender_str = gender if gender else "child"

        age = self._calculate_age(birthday)
        
        logger.info(f"Compiling synchronized text composition story for target: {name}, calculated age: {age}, gender: {gender_str}")

        prompt = self._build_prompt(name, age, gender_str, user_prefs, self.language)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
            if DEVICE == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=850,
                    temperature=0.75,
                    top_p=0.92,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            response_text = generated_text[len(prompt):].strip()

            return self._parse_response(response_text, name, age, gender_str)
            
        except Exception as e:
            logger.error(f"Textual runtime token processing exception trapped during step execution: {str(e)}")
            raise StoryGenerationError(f"Target narrative pipeline process sequence aborted: {str(e)}")

    def _calculate_age(self, birthday: Any) -> int:
        try:
            if isinstance(birthday, str):
                dt = datetime.fromisoformat(birthday.replace("Z", ""))
            elif isinstance(birthday, datetime):
                dt = birthday
            else:
                return 6
            
            now = datetime.now()
            return now.year - dt.year - ((now.month, now.day) < (dt.month, dt.day))
        except Exception:
            return 6

    def _build_prompt(self, name: str, age: int, gender: str, preferences: dict, language: str) -> str:
        theme = preferences.get("theme", "magical adventure")
        favorite_animal = preferences.get("favorite_animal", "friendly creature")
        character_trait = preferences.get("trait", "brave")

        return f"""<|system|>
You are a brilliant children's book author. You always output data STRICTLY as a single, structurally perfect JSON object fitting the requested layout schema exactly. Do not include any introductory text, background talk, conversational greetings, or notes before or after the JSON payload.
<|user|>
Write a beautiful, engaging children's story for a {age}-year-old {gender} named {name}.
The narrative theme is: {theme}. The story must include a {favorite_animal} and highlight that {name} is very {character_trait}.
The story text must be composed entirely in the {language} language.

Your output must be a single valid JSON object containing exactly 3 chronological sequential scenes matching this exact structural schema blueprint:
{{
  "title": "A short beautiful title of the overall book",
  "scenes": [
    {{
      "scene_number": 1,
      "title": "Title for scene 1",
      "narration": "Story narrative text for scene 1 (exactly 2 engaging sentences written entirely in {language}).",
      "visual_prompt": "Detailed Stable Diffusion image prompt in English specifying character descriptions, setting environment, cartoon cinematic style, high quality."
    }},
    {{
      "scene_number": 2,
      "title": "Title for scene 2",
      "narration": "Story narrative text for scene 2 (exactly 2 engaging sentences written entirely in {language}).",
      "visual_prompt": "Detailed Stable Diffusion image prompt in English maintaining character consistency, showing action, vibrant details."
    }},
    {{
      "scene_number": 3,
      "title": "Title for scene 3",
      "narration": "Story narrative text for scene 3 (exactly 2 engaging sentences written entirely in {language}).",
      "visual_prompt": "Detailed Stable Diffusion image prompt in English resolving the story happily, colorful cartoon composition."
    }}
  ]
}}
<|assistant|>
"""

    def _parse_response(self, text: str, name: str, age: int, gender: str) -> dict:
        try:
            json_match = re.search(r"({.*})", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text
                
            parsed_data = json.loads(json_str)

            if "title" in parsed_data and "scenes" in parsed_data and len(parsed_data["scenes"]) >= 3:
                return parsed_data
            else:
                raise ValueError("Generated structural matrix lacks the required multi-scene layout metrics.")
                
        except Exception as json_err:
            logger.warning(f"LLM produced non-compliant format output schema. Activating fallback matrix routines: {json_err}")

            return {
                "title": f"Aventurat e Mrekullueshme të {name}",
                "scenes": [
                    {
                        "scene_number": 1,
                        "title": "Fillimi i një Udhëtimi",
                        "narration": f"Na ishte një herë një fëmijë shumë i guximshëm i quajtur {name}, i cili sapo kishte filluar një moshë të re prej {age} vjeç. Një ditë e bukur solli një ftesë sekrete për të eksploruar një botë plot mistere dritash dhe magjie.",
                        "visual_prompt": f"A happy brave {age} year old child named {name} discovering a magical glowing portal in a vibrant room, stylized cartoon animation art style, highly detailed, 8k resolution."
                    },
                    {
                        "scene_number": 2,
                        "title": "Miku i Ri",
                        "narration": f"Rrugës për në kështjellën fluturuese, {name} takoi një krijesë fantastike dhe shumë miqësore që po kërkonte ndihmë. Së bashku, ata vendosën të bashkonin forcat për të kapërcyer çdo sfidë me buzëqeshje.",
                        "visual_prompt": f"A brave {age} year old child named {name} walking alongside a magical friendly animal companion through an enchanted forest path, cinematic lighting, colorful whimsical animation style."
                    },
                    {
                        "scene_number": 3,
                        "title": "Festimi i Fitores",
                        "narration": f"Falë guximit të madh të treguar, e gjithë mbretëria organizoi një festë të madhe me drita dhe fishekzjarre për nder të tyre. {name} kuptoi se aventura më e madhe ishte miqësia e vërtetë.",
                        "visual_prompt": f"A triumphant celebration scene with a joyful child named {name} and a magical creature looking up at beautiful colorful fireworks over a castle, epic happy ending cinematic composition."
                    }
                ]
            }
