import gc
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import DEVICE, LLM_CONFIG
from utils.logger import get_logger
from utils.exceptions import StoryGenerationError

logger = get_logger(__name__)

ALLOWED_MOODS = ("happy", "adventure", "magical", "exciting", "mysterious", "heroic")

ALLOWED_SPEAKERS = ("child", "creature", "both", "narrator_only")

UNSAFE_PATTERNS = [
    r"\bkill(ed|ing)?\b", r"\bmurder\w*\b", r"\bblood\w*\b", r"\bgun\w*\b",
    r"\bknife\b", r"\bsuicide\b", r"\bsex\w*\b", r"\bnaked\b", r"\bdrunk\w*\b",
    r"\bdrug\w*\b", r"\bhate\b", r"\bdie\b", r"\bdeath\b",
]
UNSAFE_RE = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)

class StoryGenerator:
    def __init__(
        self,
        language: str = "Albanian",
        num_scenes: Optional[int] = None,
        max_retries: int = 3,
    ):
        self.language = language
        self.num_scenes = num_scenes or LLM_CONFIG.get("num_scenes", 3)
        self.max_retries = max(1, max_retries)

        self.model_name = LLM_CONFIG["model_name"]
        self.cache_dir = LLM_CONFIG.get("model_cache_dir")
        self.max_new_tokens = LLM_CONFIG.get("max_new_tokens", 850)
        self.base_temperature = LLM_CONFIG.get("temperature", 0.75)
        self.top_p = LLM_CONFIG.get("top_p", 0.92)
        self.repetition_penalty = LLM_CONFIG.get("repetition_penalty", 1.1)

        logger.info(f"Initializing StoryGenerator inference layers targeting engine: {self.model_name}")

        self.tokenizer = None
        self.model = None
        self.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self._supports_chat_template = False

        self._load_model()

    def _load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._supports_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

            quant_config = self._build_quant_config()

            load_kwargs: Dict[str, Any] = dict(
                cache_dir=self.cache_dir,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = self.dtype
                load_kwargs["device_map"] = "auto" if DEVICE == "cuda" else "cpu"

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            self.model.eval()

            logger.success(
                f"Story generation LLM subsystem successfully loaded onto compute target device: "
                f"{DEVICE.upper()} | 4-bit: {quant_config is not None} | chat_template: {self._supports_chat_template}"
            )

        except Exception as e:
            raise StoryGenerationError(f"Critical execution fault loading fundamental LLM system architectures: {str(e)}")

    def _build_quant_config(self):
        want_4bit = LLM_CONFIG.get("load_in_4bit", DEVICE == "cuda")
        if not want_4bit or DEVICE != "cuda":
            return None

        try:
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            logger.debug(f"4-bit quantization unavailable ({e}); loading model at full precision instead.")
            return None

    def generate(
        self,
        name: str,
        birthday: Any,
        gender: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> dict:
        user_prefs = preferences if preferences else {}
        gender_str = gender if gender else "child"
        age = self._calculate_age(birthday)

        logger.info(f"Compiling synchronized text composition story for target: {name}, calculated age: {age}, gender: {gender_str}")

        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            temperature = self._temperature_for_attempt(attempt)
            strict = attempt > 1

            try:
                prompt_text, uses_chat_template = self._build_prompt(
                    name, age, gender_str, user_prefs, self.language, strict=strict
                )

                start = time.time()
                raw_text = self._run_inference(prompt_text, temperature=temperature)
                elapsed = time.time() - start
                logger.debug(f"[attempt {attempt}/{self.max_retries}] LLM generation finished in {elapsed:.1f}s (temp={temperature:.2f})")

                story = self._extract_and_validate(raw_text, name, age)
                if story is not None:
                    logger.success(f"Story generated and validated successfully on attempt {attempt}/{self.max_retries}")
                    return story

                last_error = "Generated output failed JSON structure/content validation."
                logger.warning(f"[attempt {attempt}/{self.max_retries}] {last_error} Retrying with adjusted parameters...")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[attempt {attempt}/{self.max_retries}] Generation attempt raised an exception: {e}")
            finally:
                self._free_memory()

        logger.warning(
            f"All {self.max_retries} generation attempts failed to produce a valid story "
            f"(last error: {last_error}). Falling back to a safe template story."
        )
        return self._fallback_story(name, age)

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

    def _build_schema_example(self) -> str:
        scene_blocks = []
        for i in range(1, self.num_scenes + 1):
            role_hint = (
                "introducing the setting and the main character"
                if i == 1
                else "resolving the story happily, wrapping up warmly"
                if i == self.num_scenes
                else "developing the adventure further, raising the stakes slightly"
            )
            speaker_hint = (
                "child"
                if i == 1
                else "both"
                if i == self.num_scenes
                else "creature"
            )
            scene_blocks.append(f"""    {{
      "scene_number": {i},
      "title": "Title for scene {i}",
      "mood": "one of: {', '.join(ALLOWED_MOODS)}",
      "speaker": "one of: {', '.join(ALLOWED_SPEAKERS)} (e.g. \\"{speaker_hint}\\" for this scene) — who is speaking the narration out loud in this scene",
      "narration": "Story narrative text for scene {i} (exactly 2 engaging sentences, written entirely in {{language}}).",
      "visual_prompt": "Detailed Stable Diffusion image prompt in English, {role_hint}, cartoon cinematic storybook style, maintaining character consistency."
    }}""")
        return "{\n  \"title\": \"A short beautiful title of the overall book\",\n  \"scenes\": [\n" + ",\n".join(scene_blocks) + "\n  ]\n}"

    def _build_prompt(
        self,
        name: str,
        age: int,
        gender: str,
        preferences: dict,
        language: str,
        strict: bool = False,
    ) -> Tuple[str, bool]:
        theme = preferences.get("theme", "magical adventure")
        favorite_animal = preferences.get("favorite_animal", "friendly creature")
        character_trait = preferences.get("trait", "brave")

        schema_example = self._build_schema_example().replace("{language}", language)

        system_content = (
            "You are a brilliant, warm-hearted children's book author. "
            "You always output data STRICTLY as a single, structurally perfect JSON object "
            "matching the requested schema exactly. Never include introductory text, "
            "explanations, greetings, markdown code fences, or notes before or after the JSON payload. "
            "Content must always be gentle, age-appropriate, non-violent, and end happily."
        )

        user_content = (
            f"Write a beautiful, engaging children's story for a {age}-year-old {gender} named {name}.\n"
            f"The narrative theme is: {theme}. The story must include a {favorite_animal} and highlight "
            f"that {name} is very {character_trait}.\n"
            f"The story text must be composed entirely in the {language} language.\n\n"
            f"For every scene, also decide who is speaking the narration out loud on-screen: "
            f"\"child\" if {name} is the one speaking, \"creature\" if the {favorite_animal} is speaking, "
            f"\"both\" if they are speaking together or exchanging lines, or \"narrator_only\" if no "
            f"character's mouth should move (e.g. a pure scenery/establishing moment). Vary this naturally "
            f"across the {self.num_scenes} scenes rather than using the same value every time.\n\n"
            f"Your output must be a single valid JSON object containing exactly {self.num_scenes} "
            f"chronological sequential scenes matching this exact structural schema blueprint:\n"
            f"{schema_example}"
        )

        if strict:
            user_content += (
                "\n\nIMPORTANT: Your previous output was not valid JSON or did not match the schema. "
                "Output ONLY the raw JSON object this time — no commentary, no markdown fences, "
                "no text before the opening '{' or after the closing '}'."
            )

        if self._supports_chat_template:
            try:
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return prompt, True
            except Exception as e:
                logger.debug(f"Chat template application failed ({e}); falling back to manual prompt formatting.")

        manual_prompt = f"<|system|>\n{system_content}\n<|user|>\n{user_content}\n<|assistant|>\n"
        return manual_prompt, False

    def _temperature_for_attempt(self, attempt: int) -> float:
        decay = 0.1 * (attempt - 1)
        return max(0.3, self.base_temperature - decay)

    def _run_inference(self, prompt: str, temperature: float) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
            if DEVICE == "cuda":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()

            for marker in ("<|assistant|>", "[/INST]", "assistant\n"):
                if marker in generated_text:
                    return generated_text.split(marker)[-1].strip()

            return generated_text.strip()

        except Exception as e:
            raise StoryGenerationError(f"Target narrative pipeline process sequence aborted: {str(e)}")

    def _extract_and_validate(self, text: str, name: str, age: int) -> Optional[dict]:
        parsed = self._extract_json(text)
        if parsed is None:
            return None

        if not self._validate_story_structure(parsed):
            return None

        if self._flag_unsafe_content(parsed):
            logger.warning("Generated story tripped the child-safety content scan; discarding this attempt.")
            return None

        self._normalize_story(parsed)
        return parsed

    def _extract_json(self, text: str) -> Optional[dict]:
        candidates = []

        candidates.append(text.strip())

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for candidate in candidates:
            repaired = self._repair_json(candidate)
            try:
                return json.loads(repaired)
            except (json.JSONDecodeError, TypeError):
                continue

        return None

    def _repair_json(self, raw: str) -> str:
        s = raw.strip()

        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

        s = re.sub(r",\s*([}\]])", r"\1", s)

        open_braces, close_braces = s.count("{"), s.count("}")
        if open_braces > close_braces:
            s += "}" * (open_braces - close_braces)

        open_brackets, close_brackets = s.count("["), s.count("]")
        if open_brackets > close_brackets:
            s += "]" * (open_brackets - close_brackets)

        return s.strip()

    def _validate_story_structure(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        if "title" not in data or not str(data.get("title", "")).strip():
            return False

        scenes = data.get("scenes")
        if not isinstance(scenes, list) or len(scenes) < self.num_scenes:
            return False

        for i, scene in enumerate(scenes[: self.num_scenes], start=1):
            if not isinstance(scene, dict):
                return False
            narration = str(scene.get("narration", "")).strip()
            visual_prompt = str(scene.get("visual_prompt", "")).strip()
            if len(narration) < 5 or len(visual_prompt) < 5:
                return False

        return True

    def _flag_unsafe_content(self, data: dict) -> bool:
        for scene in data.get("scenes", []):
            narration = str(scene.get("narration", ""))
            title = str(scene.get("title", ""))
            if UNSAFE_RE.search(narration) or UNSAFE_RE.search(title):
                return True
        return False

    def _normalize_story(self, data: dict) -> None:
        data["scenes"] = data["scenes"][: self.num_scenes]
        for i, scene in enumerate(data["scenes"], start=1):
            scene["scene_number"] = i
            mood = str(scene.get("mood", "")).strip().lower()
            scene["mood"] = mood if mood in ALLOWED_MOODS else "happy"
            scene.setdefault("title", f"Scene {i}")

            speaker = str(scene.get("speaker", "")).strip().lower()
            scene["speaker"] = speaker if speaker in ALLOWED_SPEAKERS else self._default_speaker_for_index(i)

    def _default_speaker_for_index(self, scene_index: int) -> str:
        """Fallback speaker assignment when the LLM omits or misformats the
        'speaker' field. Cycles through child -> creature -> both so that,
        even in the worst case, both lip-sync paths (SadTalker for the child,
        procedural mouth-flap for the creature) get exercised across a story
        rather than defaulting everything to silence."""
        cycle = ("child", "creature", "both")
        return cycle[(scene_index - 1) % len(cycle)]

    def _fallback_story(self, name: str, age: int) -> dict:
        moods_cycle = ["happy", "adventure", "magical", "exciting", "heroic", "mysterious"]
        speakers_cycle = ["child", "creature", "both"]

        templates = [
            {
                "title": "The Beginning of a Journey",
                "speaker": "child",
                "narration": (
                    f"Once upon a time there was a very brave child named {name}, who had just "
                    f"started a new age of {age} years old. A beautiful day brought a secret invitation "
                    f"to explore a world full of mysteries of light and magic. "
                ),
                "visual_prompt": (
                    f"A happy brave {age} year old child named {name} discovering a magical glowing portal "
                    f"in a vibrant room, stylized cartoon animation art style, highly detailed, 8k resolution."
                ),
            },
            {
                "title": "New Friend",
                "speaker": "creature",
                "narration": (
                    f"On the way to the flying castle, {name} met a fantastic and very "
                    f"friendly person who was asking for help. Together, they decided to join forces to "
                    f"overcome every challenge with a smile."
                ),
                "visual_prompt": (
                    f"A brave {age} year old child named {name} walking alongside a magical friendly animal "
                    f"companion through an enchanted forest path, cinematic lighting, colorful whimsical animation style."
                ),
            },
            {
                "title": "Victory Celebration",
                "speaker": "both",
                "narration": (
                    f"Thanks to the great courage shown, the entire kingdom organized a great celebration with "
                    f"lights and fireworks in their honor. {name} realized that the greatest adventure was "
                    f"true friendship."
                ),
                "visual_prompt": (
                    f"A triumphant celebration scene with a joyful child named {name} and a magical creature "
                    f"looking up at beautiful colorful fireworks over a castle, epic happy ending cinematic composition."
                ),
            },
        ]

        scenes = []
        for i in range(1, self.num_scenes + 1):
            base = templates[(i - 1) % len(templates)]
            scenes.append({
                "scene_number": i,
                "title": f"{base['title']} ({i})" if i > len(templates) else base["title"],
                "mood": moods_cycle[(i - 1) % len(moods_cycle)],
                "speaker": base.get("speaker", speakers_cycle[(i - 1) % len(speakers_cycle)]),
                "narration": base["narration"],
                "visual_prompt": base["visual_prompt"],
            })

        return {
            "title": f"The Amazing Adventures of {name}",
            "scenes": scenes,
        }

    def _free_memory(self) -> None:
        gc.collect()
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
