import torch
import json
import re
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import get_logger
from utils.exceptions import StoryGenerationError

logger = get_logger(__name__)

class StoryGenerator:
    def __init__(self, language: str = "Albanian"):
        self.language = language
        # Kalojmë te TinyLlama që është ultra i lehtë për CPU dhe nuk bën crash RAM-in
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        logger.info(f"Duke inicializuar StoryGenerator në CPU me modelin: {self.model_name}")
        
        try:
            # 1. Ngarkojmë Tokenizer-in
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # 2. Ngarkojmë modelin në CPU me saktësi float32 (standarde për CPU)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,           # Parandalon mbingarkesën e RAM-it gjatë leximit
                device_map="cpu",                 # Detyron ekzekutimin në CPU
                trust_remote_code=True
            )
            logger.success("Modeli TinyLlama u ngarkua me sukses në CPU!")
            
        except Exception as e:
            raise StoryGenerationError(f"Dështoi ngarkimi i modelit LLM në CPU: {str(e)}")

    def generate(self, name: str, age: int, birthday: datetime, language: str = "Albanian") -> dict:
        """Gjeneron një histori të strukturuar për fëmijën duke përdorur CPU."""
        logger.info(f"Duke gjeneruar histori në CPU për {name}, moshë: {age}")
        
        # Formatimi i prompt-it sipas strukturës chat të TinyLlama (<|system|>, <|user|>)
        prompt = self._build_prompt(name, age, language)
        
        try:
            # Tokenizimi i inputit direkt në CPU
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,            # Kufizojmë pak tokenat për shpejtësi në CPU
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Izolojmë përgjigjen duke hequr prompt-in fillestar
            response_text = generated_text[len(prompt):].strip()
            
            # Parsimi i strukturës JSON
            return self._parse_response(response_text, name, age)
            
        except Exception as e:
            logger.error(f"Gabim gjatë gjenerimit të tekstit në CPU: {str(e)}")
            raise StoryGenerationError(f"Dështoi gjenerimi i historisë: {str(e)}")

    def _build_prompt(self, name: str, age: int, language: str) -> str:
        """Krijon një prompt të saktë sipas template-it të TinyLlama."""
        return f"""<|system|>
You are a creative children's book author. You always respond strictly with a valid JSON object following the requested schema. No conversational text before or after JSON.
<|user|>
Create a magical story for a child named {name} who is {age} years old. The story must be in {language}.
Output MUST be a single valid JSON object exactly like this:
{{
  "title": "Title of the story",
  "scenes": [
    {{
      "scene_number": 1,
      "title": "Title of scene 1",
      "narration": "Story text for scene 1 (2 sentences in {language}).",
      "visual_prompt": "Stable Diffusion prompt in English describing characters and scene."
    }},
    {{
      "scene_number": 2,
      "title": "Title of scene 2",
      "narration": "Story text for scene 2 (2 sentences in {language}).",
      "visual_prompt": "Stable Diffusion prompt in English describing characters and scene."
    }},
    {{
      "scene_number": 3,
      "title": "Title of scene 3",
      "narration": "Story text for scene 3 (2 sentences in {language}).",
      "visual_prompt": "Stable Diffusion prompt in English describing characters and scene."
    }}
  ]
}}
<|assistant|>
"""

    def _parse_response(self, text: str, name: str, age: int) -> dict:
        """Pastrohet teksti dhe parsohet si JSON."""
        try:
            # Përdorim regex për të izoluar bllokun e parë JSON në rast se modeli ka shtuar zhurmë
            json_match = re.search(r"({.*})", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text
                
            return json.loads(json_str)
            
        except json.JSONDecodeError:
            logger.warning("TinyLlama nuk ktheu një JSON plotësisht valid. Duke aktivizuar fallback...")
            return {
                "title": f"Aventurat e {name}",
                "scenes": [
                    {
                        "scene_number": 1,
                        "title": "Fillimi i Udhëtimit",
                        "narration": f"Na ishte një herë një fëmijë i guximshëm i quajtur {name}, i cili sapo kishte mbushur {age} vjeç dhe ishte gati për një aventurë të madhe.",
                        "visual_prompt": f"A brave {age} year old child named {name} looking at a magical forest, cartoon animation style, high quality, colorful."
                    }
                ]
            }
