import torch
import json
import re
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import LLM_CONFIG
from utils.logger import get_logger
from utils.exceptions import StoryGenerationError

logger = get_logger(__name__)

class StoryGenerator:
    def __init__(self, language: str = "Albanian"):
        self.language = language
        self.model_name = LLM_CONFIG.get("model_name", "microsoft/phi-2")
        
        logger.info(f"Duke inicializuar StoryGenerator me modelin: {self.model_name}")
        
        try:
            # 1. Ngarkojmë Tokenizer-in
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Sigurohemi që ekziston pad_token (Phi-2 nuk e ka të definuar si parazgjedhje)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # 2. Ngarkojmë modelin DIREKT në GPU (CUDA) për të shmangur crash-in e RAM-it
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,        # Përgjysmon peshën në VRAM (fp16)
                low_cpu_mem_usage=True,           # Parandalon "spike" të RAM-it në CPU
                device_map="cuda",                # Detyron alokimin e plotë në GPU T4
                trust_remote_code=True
            )
            logger.success("Modeli Phi-2 u ngarkua me sukses në GPU CUDA")
            
        except Exception as e:
            raise StoryGenerationError(f"Dështoi ngarkimi i modelit LLM: {str(e)}")

    def generate(self, name: str, age: int, birthday: datetime, language: str = "Albanian") -> dict:
        """Gjeneron një histori të strukturuar për fëmijën bazuar në të dhënat e tij."""
        logger.info(f"Duke gjeneruar histori për {name}, moshë: {age}")
        
        # Formatimi i një prompt-i të strukturuar (Instruct-style për Phi-2)
        prompt = self._build_prompt(name, age, language)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=LLM_CONFIG.get("max_new_tokens", 800),
                    temperature=LLM_CONFIG.get("temperature", 0.7),
                    top_p=LLM_CONFIG.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Izolojmë vetëm përgjigjen e modelit pas prompt-it
            response_text = generated_text[len(prompt):].strip()
            
            # Parsimi i strukturës JSON nga përgjigja e gjeneruar
            return self._parse_response(response_text, name, age)
            
        except Exception as e:
            logger.error(f"Gabim gjatë gjenerimit të tekstit: {str(e)}")
            raise StoryGenerationError(f"Dështoi gjenerimi i historisë: {str(e)}")

    def _build_prompt(self, name: str, age: int, language: str) -> str:
        """Krijon një prompt të saktë që e detyron modelin të kthejë një strukturë të pastër JSON."""
        return f"""Instruct: Create a magical children's adventure story for a child named {name} who is {age} years old.
The story must be written in {language}.
The output MUST be a valid JSON object exactly following this schema, with no additional conversational text or markers before or after the JSON:
{{
  "title": "Title of the story",
  "scenes": [
    {{
      "scene_number": 1,
      "title": "Title of scene 1",
      "narration": "The story text for scene 1 to be read aloud (2-3 sentences in {language}).",
      "visual_prompt": "Detailed Stable Diffusion image prompt describing the characters, action, and magical background for this scene (in English)."
    }},
    {{
      "scene_number": 2,
      "title": "Title of scene 2",
      "narration": "The story text for scene 2 to be read aloud (2-3 sentences in {language}).",
      "visual_prompt": "Detailed Stable Diffusion image prompt for this scene (in English)."
    }},
    {{
      "scene_number": 3,
      "title": "Title of scene 3",
      "narration": "The story text for scene 3 to be read aloud (2-3 sentences in {language}).",
      "visual_prompt": "Detailed Stable Diffusion image prompt for this scene (in English)."
    }}
  ]
}}
Output:"""

    def _parse_response(self, text: str, name: str, age: int) -> dict:
        """Pastrohet teksti dhe parsohet si JSON objekti final."""
        try:
            # Përdorim regex për të gjetur bllokun JSON nëse modeli ka shtuar tekst rrethues
            json_match = re.search(r"({.*})", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text
                
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            logger.warning("Modeli nuk ktheu një JSON plotësisht valid. Duke aplikuar fallback...")
            # Fallback strukture në rast se dështon parsimi (siguron që pipeline mos të bëjë crash)
            return {
                "title": f"Aventurat e {name}",
                "scenes": [
                    {
                        "scene_number": 1,
                        "title": "Fillimi i udhëtimit",
                        "narration": f"Na ishte një herë një fëmijë i guximshëm i quajtur {name}, i cili sapo kishte mbushur {age} vjeç dhe ishte gati për një aventurë të madhe.",
                        "visual_prompt": f"A brave {age} year old child named {name} looking at a magical forest, cartoon animation style, high quality, colorful."
                    }
                ]
            }
