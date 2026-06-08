import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

from config.settings import LLM_CONFIG, DEVICE
from utils.logger import get_logger
from utils.exceptions import StoryGenerationError, ModelLoadError

logger = get_logger(__name__)

THEMES_BY_AGE = {
    range(1, 4) : ["kafshë të miqësueshme", "lodra magjike", "kopshti i ngjyrave"],
    range(4, 7) : ["superhero i vogël", "eksplorues hapësinor", "pirat i mirë", "magjistar"],
    range(7, 10): ["aventurë në xhungël", "detektiv i vogël", "udhëtim nëpër kohë", "dragua mik"],
    range(10, 13): ["hero i shkollës", "sportist kampion", "shpikës gjeniu", "ekspeditë arktike"],
    range(13, 17): ["lider i ekipit", "artistë i talentuar", "shkencëtar i ri", "alpinist"],
}

MOODS = ["magjik", "aventuroz", "gëzues", "emocionues", "misteroz", "heroik"]

SETTINGS = {
    "Albanian": ["Shqipëri", "Alpet Shqiptare", "Deti Adriatik", "Berat", "Gjirokastrës"],
    "English" : ["enchanted forest", "outer space", "underwater kingdom", "magic castle"],
    "Italian" : ["foresta incantata", "castello magico", "isola misteriosa", "giardino segreto"],
    "German"  : ["Zauberwald", "Weltraum", "Unterwasserwelt", "Märchenburg"],
    "French"  : ["forêt enchantée", "château magique", "île mystérieuse", "jardin secret"],
}


class StoryGenerator:
    def __init__(self, language: str = "Albanian"):
        self.language   = language
        self.model      = None
        self.tokenizer  = None
        self.pipe       = None
        self._load_model()
    
    def _load_model(self) -> None:
        model_name = "microsoft/phi-2" 
        cache_dir  = LLM_CONFIG["model_cache_dir"]

        logger.debug(f"Duke ngarkuar LLM: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            dtype = torch.float16 if DEVICE == "cuda" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                device_map="auto" if DEVICE == "cuda" else "cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            logger.success(f"LLM u ngarkua: {model_name}")

        except OSError as e:
            raise ModelLoadError(model_name, f"Model nuk u gjet: {e}")
        except Exception as e:
            raise ModelLoadError(model_name, str(e))

    def generate(
        self,
        name: str,
        age: int,
        birthday: datetime,
       language: str = "Albanian",
    ) -> dict:
        logger.debug(f"Duke gjeneruar histori | emër: {name} | moshë: {age} | gjuhë: {language}")

        theme = self._pick_theme(age)
        logger.debug(f"Tema e zgjedhur: {theme}")

        prompt = self._build_prompt(
            name=name,
            age=age,
            birthday=birthday,
            language=language,
            theme=theme,
        )

        raw_output = self._call_llm(prompt)

        story = self._parse_story(raw_output, name, age, language, theme)

        story = self._enrich_with_visual_prompts(story, name, age)

        self._validate_story(story)

        logger.success(
            f"Historia u gjenerua | titull: '{story['title']}' | "
            f"{len(story['scenes'])} skena | "
            f"{story['total_duration_sec']}s total"
        )

        return story

    def _build_prompt(
        self,
        name: str,
        age: int,
        birthday: datetime,
        language: str,
        theme: str,
    ) -> str:
        num_scenes   = LLM_CONFIG["num_scenes"]
        scene_dur    = LLM_CONFIG["scene_duration_sec"]
        bday_str     = birthday.strftime("%B %d")
        settings_opt = ", ".join(SETTINGS.get(language, SETTINGS["English"])[:3])

        system_prompt = (
            "You are a creative children's movie scriptwriter specializing in "
            "personalized stories. You write age-appropriate, positive, and "
            "emotionally engaging stories where the child is the hero. "
            "Always return valid JSON only, with no additional text or markdown."
        )

        user_prompt = f"""
Create a personalized children's movie script for a child with these details:
- Name: {name}
- Age: {age} years old
- Birthday: {bday_str}
- Story theme: {theme}
- Language for the story: {language}
- Number of scenes: {num_scenes}
- Each scene duration: {scene_dur} seconds

Requirements:
1. The child ({name}) is the HERO of the story
2. Story must be age-appropriate for a {age}-year-old
3. ALL narration and dialogue must be written in {language}
4. Story should be positive, adventurous, and emotionally engaging
5. Each scene must have a clear visual description for animation
6. Possible settings to use: {settings_opt}

CRITICAL:
Return ONLY valid parsable JSON.
Do not write explanations.
Do not use markdown.
Do not use ```json.
Do not add notes before or after JSON.
Return a single valid JSON object.
Do not wrap in markdown.
{{
  "title": "Movie title in {language}",
  "description": "2-3 sentence summary in {language}",
  "theme": "{theme}",
  "language": "{language}",
  "total_duration_sec": {num_scenes * scene_dur},
  "scenes": [
    {{
      "index": 1,
      "title": "Scene title in {language}",
      "narration": "Narrator text spoken aloud, 2-4 sentences in {language}",
      "setting": "Where the scene takes place",
      "mood": "one of: happy, adventure, magical, exciting, mysterious, heroic",
      "duration_sec": {scene_dur},
      "visual_description": "Detailed visual description for animation (in English)"
    }}
  ]
}}

Make sure:
- narration is engaging and mentions {name} by name
- visual_description is always in English (for the image generator)
- mood varies across scenes for better storytelling
- the story has a clear beginning, middle, and happy ending
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"

        return formatted

    def _call_llm(self, prompt: str) -> str:
        logger.debug("Duke thirrur LLM...")

        try:
            outputs = self.pipe(
                prompt,
                max_new_tokens     = LLM_CONFIG["max_new_tokens"],
                temperature        = LLM_CONFIG["temperature"],
                top_p              = LLM_CONFIG["top_p"],
                repetition_penalty = LLM_CONFIG["repetition_penalty"],
                do_sample          = LLM_CONFIG["do_sample"],
                return_full_text   = False,
                pad_token_id       = self.tokenizer.eos_token_id,
            )

            generated = outputs[0]["generated_text"].strip()
            print("\n========== RAW LLM OUTPUT ==========\n")
            print(generated)
            print("\n===================================\n")
            logger.debug(f"LLM gjeneroi {len(generated)} karaktere")
            return generated

        except torch.cuda.OutOfMemoryError:
            raise StoryGenerationError(
                "GPU memory e pamjaftueshme për LLM. "
                "Provo të ulësh max_new_tokens në settings.py"
            )
        except Exception as e:
            raise StoryGenerationError(f"Gabim gjatë gjenerimit të historisë: {e}")

    def _parse_story(
        self,
        raw_output: str,
        name: str,
        age: int,
        language: str,
        theme: str,
    ) -> dict:
        logger.debug("Duke parsuar output-in e LLM...")

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        cleaned = re.sub(r'```(?:json)?\s*', '', raw_output)
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        logger.warning("JSON parsimi deshtoi, duke gjeneruar histori fallback...")
        if '"title"' in raw_output and '"scenes"' not in raw_output:
            logger.warning("LLM output missing scenes, using fallback.")

        return self._generate_fallback_story(name, age, language, theme)

    def _enrich_with_visual_prompts(self, story: dict, name: str, age: int) -> dict:
        style_prefix = (
            "children's movie animation style, pixar quality, "
            "vibrant colors, soft warm lighting, child-friendly, "
            "high detail, cinematic composition, "
        )

        negative_base = (
            "ugly, blurry, dark, scary, violent, adult, "
            "realistic photo, dark theme, disturbing"
        )

        age_style = self._get_age_style(age)

        for scene in story.get("scenes", []):
            visual_desc = scene.get("visual_description", scene.get("title", ""))
            mood        = scene.get("mood", "happy")

            mood_suffix   = self._get_mood_suffix(mood)
            scene_prompt  = (
                f"{style_prefix}"
                f"{age_style}, "
                f"{visual_desc}, "
                f"main character is a {age}-year-old child hero, "
                f"{mood_suffix}"
            )

            scene["visual_prompt"]    = scene_prompt
            scene["negative_prompt"]  = negative_base
            scene.setdefault("duration_sec", LLM_CONFIG["scene_duration_sec"])

        return story

    def _get_age_style(self, age: int) -> str:
        if age <= 4:
            return "cute rounded characters, pastel colors, very simple friendly shapes"
        elif age <= 8:
            return "colorful cartoon style, expressive characters, magical elements"
        elif age <= 12:
            return "detailed animation, dynamic action poses, adventure atmosphere"
        else:
            return "stylized animation, dramatic lighting, epic atmosphere"

    def _get_mood_suffix(self, mood: str) -> str:
        mood_map = {
            "happy"     : "bright sunny atmosphere, warm colors, smiling faces",
            "adventure" : "dynamic composition, exciting action, bold colors",
            "magical"   : "sparkles and magic particles, glowing lights, wonder",
            "exciting"  : "dramatic angle, motion blur, intense colors",
            "mysterious": "soft fog, moonlight, subtle shadows, curious atmosphere",
            "heroic"    : "epic angle, golden lighting, triumphant pose",
        }
        return mood_map.get(mood, "warm and friendly atmosphere")

    def _pick_theme(self, age: int) -> str:
        """Zgjidh temën e historisë bazuar në moshën e fëmijës."""
        import random
        for age_range, themes in THEMES_BY_AGE.items():
            if age in age_range:
                return random.choice(themes)
        return random.choice(THEMES_BY_AGE[range(7, 10)])

    def _validate_story(self, story: dict) -> None:
        """
        Valido strukturën e historisë dhe plotëso fushat që mungojnë.

        Raises:
            StoryGenerationError: Nëse struktura është e pavlefshme
        """
        required_fields = ["title", "scenes", "language"]
        for field in required_fields:
            if field not in story:
                raise StoryGenerationError(
                    f"Historia nuk ka fushën e kërkuar: '{field}'"
                )

        scenes = story.get("scenes", [])
        if not scenes:
            raise StoryGenerationError("Historia nuk ka asnjë skenë")

        if len(scenes) < 2:
            raise StoryGenerationError(
                f"Historia ka vetëm {len(scenes)} skenë, kërkohen të paktën 2"
            )

        # Valido dhe plotëso çdo skenë
        for i, scene in enumerate(scenes):
            scene.setdefault("index", i + 1)
            scene.setdefault("title", f"Skena {i + 1}")
            scene.setdefault("narration", "")
            scene.setdefault("mood", "happy")
            scene.setdefault("setting", "")
            scene.setdefault("duration_sec", LLM_CONFIG["scene_duration_sec"])
            scene.setdefault("visual_prompt", "children's movie scene, colorful animation")
            scene.setdefault("negative_prompt", "ugly, blurry, scary")

        story["total_duration_sec"] = sum(
            s.get("duration_sec", LLM_CONFIG["scene_duration_sec"])
            for s in scenes
        )

    def _generate_fallback_story(
        self,
        name: str,
        age: int,
        language: str,
        theme: str,
    ) -> dict:
        """
        Gjenero histori bazike kur LLM dështon të kthejë JSON të vlefshëm.
        Garanton që pipeline të vazhdojë edhe pa output të mirë nga LLM.
        """
        logger.warning(f"Duke përdorur fallback story për: {name}")

        first_name = name.split()[0]

        fallback_narrations = {
            "Albanian": [
                f"Ishte njëherë {first_name}, një fëmijë i guximshëm dhe i mrekullueshëm.",
                f"{first_name} zbuloi një sekret magjik që ndryshoi gjithçka.",
                f"Sfida ishte e madhe, por {first_name} nuk u dorëzua kurrë.",
                f"Me guxim dhe zemër të mirë, {first_name} gjeti rrugën drejt fitores.",
                f"Kështu {first_name} u bë heroi i historisë së tij të bukur.",
            ],
            "English": [
                f"Once upon a time, {first_name} was a brave and wonderful child.",
                f"{first_name} discovered a magical secret that changed everything.",
                f"The challenge was great, but {first_name} never gave up.",
                f"With courage and a kind heart, {first_name} found the way to victory.",
                f"And so {first_name} became the hero of their beautiful story.",
            ],
        }

        narrations = fallback_narrations.get(language, fallback_narrations["English"])
        settings_list = SETTINGS.get(language, SETTINGS["English"])

        scenes = []
        moods  = ["magical", "adventure", "exciting", "heroic", "happy"]

        for i in range(min(5, len(narrations))):
            scenes.append({
                "index"        : i + 1,
                "title"        : f"Chapter {i + 1}",
                "narration"    : narrations[i],
                "setting"      : settings_list[i % len(settings_list)],
                "mood"         : moods[i],
                "duration_sec" : LLM_CONFIG["scene_duration_sec"],
                "visual_description": (
                    f"A {age}-year-old child hero in a {theme} adventure, "
                    f"scene {i+1}, colorful and magical"
                ),
            })

        return {
            "title"              : f"{first_name} dhe Aventura e Madhe",
            "description"        : f"Historia e personalizuar e {name}.",
            "theme"              : theme,
            "language"           : language,
            "total_duration_sec" : len(scenes) * LLM_CONFIG["scene_duration_sec"],
            "scenes"             : scenes,
            "_fallback"          : True,
        }
