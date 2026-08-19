import gc
import json
import re
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import DEVICE, LLM_CONFIG
from utils.logger import get_logger
from utils.exceptions import StoryGenerationError

logger = get_logger(__name__)


ALLOWED_MOODS = (
    "happy",
    "adventure",
    "magical",
    "exciting",
    "mysterious",
    "heroic",
)

ALLOWED_SPEAKERS = (
    "child",
    "creature",
    "both",
    "narrator_only",
)


# Safety filter intentionally targets narration/title content.
# Visual prompts are also checked separately because generated visual
# descriptions can otherwise introduce unsuitable concepts.
UNSAFE_PATTERNS = [
    r"\bkill(?:s|ed|ing)?\b",
    r"\bmurder\w*\b",
    r"\bblood\w*\b",
    r"\bgun\w*\b",
    r"\brifle\w*\b",
    r"\bpistol\w*\b",
    r"\bweapon\w*\b",
    r"\bknife\b",
    r"\bsuicide\b",
    r"\bsex\w*\b",
    r"\bnaked\b",
    r"\bnudity\b",
    r"\bdrunk\w*\b",
    r"\bdrug\w*\b",
    r"\bhate\w*\b",
    r"\bdie\b",
    r"\bdied\b",
    r"\bdeath\b",
    r"\bdead\b",
]

UNSAFE_RE = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)


class StoryGenerator:
    """
    Generates and validates structured children's stories.

    Design goals:
      - JSON-only model output.
      - Strong structural validation before downstream pipeline stages.
      - Safe fallback when the LLM produces malformed/unsafe content.
      - CPU-friendly loading when CUDA is unavailable.
      - Optional 4-bit CUDA quantization.
      - Stable scene numbering and speaker assignment.
      - Clean downstream contract for audio/frame generation.

    Expected output:

        {
            "title": "...",
            "scenes": [
                {
                    "scene_number": 1,
                    "title": "...",
                    "mood": "happy",
                    "speaker": "child",
                    "narration": "...",
                    "visual_prompt": "..."
                }
            ]
        }
    """

    def __init__(
        self,
        language: str = "Albanian",
        num_scenes: Optional[int] = None,
        max_retries: int = 3,
    ):
        self.language = str(language or "Albanian").strip() or "Albanian"

        configured_scenes = LLM_CONFIG.get("num_scenes", 3)

        try:
            configured_scenes = int(configured_scenes)
        except (TypeError, ValueError):
            configured_scenes = 3

        self.num_scenes = max(
            1,
            int(num_scenes) if num_scenes is not None else configured_scenes,
        )

        self.max_retries = max(1, int(max_retries))

        self.model_name = LLM_CONFIG["model_name"]
        self.cache_dir = LLM_CONFIG.get("model_cache_dir")

        self.max_new_tokens = max(
            128,
            int(LLM_CONFIG.get("max_new_tokens", 850)),
        )

        self.base_temperature = float(
            LLM_CONFIG.get("temperature", 0.75)
        )

        self.top_p = float(
            LLM_CONFIG.get("top_p", 0.92)
        )

        self.repetition_penalty = float(
            LLM_CONFIG.get("repetition_penalty", 1.1)
        )

        # Avoid accidental invalid sampling parameters.
        self.base_temperature = max(0.1, min(self.base_temperature, 2.0))
        self.top_p = max(0.05, min(self.top_p, 1.0))
        self.repetition_penalty = max(1.0, self.repetition_penalty)

        self.tokenizer = None
        self.model = None

        self.dtype = (
            torch.float16
            if DEVICE == "cuda"
            else torch.float32
        )

        self._supports_chat_template = False

        logger.info(
            f"Initializing StoryGenerator | "
            f"model={self.model_name} | "
            f"device={DEVICE.upper()} | "
            f"language={self.language} | "
            f"scenes={self.num_scenes}"
        )

        self._load_model()

    # ------------------------------------------------------------------
    # MODEL INITIALIZATION
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    raise StoryGenerationError(
                        "Tokenizer has neither pad_token nor eos_token."
                    )

            self._supports_chat_template = (
                getattr(self.tokenizer, "chat_template", None) is not None
            )

            quant_config = self._build_quant_config()

            load_kwargs: Dict[str, Any] = {
                "cache_dir": self.cache_dir,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }

            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = self.dtype

                if DEVICE == "cuda":
                    load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["device_map"] = "cpu"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

            self.model.eval()

            logger.success(
                f"Story LLM loaded successfully | "
                f"device={DEVICE.upper()} | "
                f"4bit={quant_config is not None} | "
                f"chat_template={self._supports_chat_template}"
            )

        except StoryGenerationError:
            raise

        except Exception as e:
            raise StoryGenerationError(
                f"Failed to load story generation model: {e}"
            )

    def _build_quant_config(self):
        want_4bit = bool(
            LLM_CONFIG.get(
                "load_in_4bit",
                DEVICE == "cuda",
            )
        )

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
            logger.warning(
                f"4-bit quantization unavailable ({e}); "
                f"falling back to standard precision."
            )
            return None

    # ------------------------------------------------------------------
    # PUBLIC GENERATION API
    # ------------------------------------------------------------------

    def generate(
        self,
        name: str,
        birthday: Any,
        gender: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> dict:

        name = self._sanitize_name(name)
        gender_str = self._sanitize_gender(gender)
        user_prefs = self._sanitize_preferences(preferences)

        age = self._calculate_age(birthday)

        logger.info(
            f"Generating story | "
            f"name={name} | age={age} | gender={gender_str}"
        )

        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            temperature = self._temperature_for_attempt(attempt)
            strict = attempt > 1

            try:
                prompt_text, _ = self._build_prompt(
                    name=name,
                    age=age,
                    gender=gender_str,
                    preferences=user_prefs,
                    language=self.language,
                    strict=strict,
                )

                start_time = time.time()

                raw_text = self._run_inference(
                    prompt_text,
                    temperature=temperature,
                )

                elapsed = time.time() - start_time

                logger.debug(
                    f"[attempt {attempt}/{self.max_retries}] "
                    f"LLM generation completed in {elapsed:.1f}s "
                    f"(temperature={temperature:.2f})"
                )

                story = self._extract_and_validate(
                    raw_text,
                    name=name,
                    age=age,
                )

                if story is not None:
                    logger.success(
                        f"Story generated successfully on "
                        f"attempt {attempt}/{self.max_retries}"
                    )
                    return story

                last_error = (
                    "Generated output failed JSON, "
                    "structure, safety, or content validation."
                )

                logger.warning(
                    f"[attempt {attempt}/{self.max_retries}] "
                    f"{last_error}"
                )

            except Exception as e:
                last_error = str(e)

                logger.warning(
                    f"[attempt {attempt}/{self.max_retries}] "
                    f"Generation failed: {e}"
                )

            finally:
                self._free_memory()

        logger.warning(
            f"All {self.max_retries} story-generation attempts failed. "
            f"Using deterministic safe fallback. "
            f"Last error: {last_error}"
        )

        return self._fallback_story(name, age, user_prefs)

    # ------------------------------------------------------------------
    # INPUT NORMALIZATION
    # ------------------------------------------------------------------

    def _sanitize_name(self, name: Any) -> str:
        value = str(name or "").strip()

        if not value:
            return "Alex"

        # Keep the name human-readable but prevent prompt injection-like
        # formatting from becoming an enormous prompt.
        value = re.sub(r"[\r\n\t]+", " ", value)
        value = value[:80].strip()

        return value or "Alex"

    def _sanitize_gender(self, gender: Optional[str]) -> str:
        if not gender:
            return "child"

        value = str(gender).strip().lower()

        allowed = {
            "boy",
            "girl",
            "child",
            "male",
            "female",
        }

        return value if value in allowed else "child"

    def _sanitize_preferences(
        self,
        preferences: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:

        if not isinstance(preferences, dict):
            preferences = {}

        def clean(value: Any, default: str, max_length: int = 120) -> str:
            if value is None:
                return default

            text = re.sub(
                r"[\r\n\t]+",
                " ",
                str(value),
            ).strip()

            text = text[:max_length]

            if not text:
                return default

            return text

        return {
            "theme": clean(
                preferences.get("theme"),
                "magical adventure",
            ),
            "favorite_animal": clean(
                preferences.get("favorite_animal"),
                "friendly creature",
            ),
            "trait": clean(
                preferences.get("trait"),
                "brave",
            ),
        }

    # ------------------------------------------------------------------
    # AGE
    # ------------------------------------------------------------------

    def _calculate_age(self, birthday: Any) -> int:
        try:
            if isinstance(birthday, datetime):
                dt = birthday

            elif isinstance(birthday, date):
                dt = datetime.combine(
                    birthday,
                    datetime.min.time(),
                )

            elif isinstance(birthday, str):
                value = birthday.strip()

                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"

                dt = datetime.fromisoformat(value)

            else:
                return 6

            now = datetime.now(
                dt.tzinfo
            ) if dt.tzinfo else datetime.now()

            age = (
                now.year
                - dt.year
                - ((now.month, now.day) < (dt.month, dt.day))
            )

            # Keep the generated content in the intended children's range.
            return max(1, min(age, 17))

        except Exception:
            logger.debug(
                f"Unable to parse birthday '{birthday}', "
                f"using default age 6."
            )
            return 6

    # ------------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------------

    def _build_schema_example(self) -> str:
        scene_blocks = []

        for i in range(1, self.num_scenes + 1):
            if i == 1:
                role_hint = (
                    "introducing the setting, the main character, "
                    "and the adventure"
                )
                speaker_hint = "child"

            elif i == self.num_scenes:
                role_hint = (
                    "bringing the adventure to a warm, "
                    "happy and satisfying conclusion"
                )
                speaker_hint = "both"

            else:
                role_hint = (
                    "developing the adventure with a gentle "
                    "new discovery or challenge"
                )
                speaker_hint = "creature"

            scene_blocks.append(
                f'''    {{
      "scene_number": {i},
      "title": "Short scene title",
      "mood": "happy",
      "speaker": "{speaker_hint}",
      "narration": "Exactly two engaging sentences written entirely in {{language}}.",
      "visual_prompt": "Detailed English image-generation prompt, {role_hint}, colorful cinematic children's storybook illustration, consistent characters."
    }}'''
            )

        return (
            '{\n'
            '  "title": "Short overall story title",\n'
            '  "scenes": [\n'
            + ",\n".join(scene_blocks)
            + "\n  ]\n}"
        )

    def _build_prompt(
        self,
        name: str,
        age: int,
        gender: str,
        preferences: dict,
        language: str,
        strict: bool = False,
    ) -> Tuple[str, bool]:

        theme = preferences.get(
            "theme",
            "magical adventure",
        )

        favorite_animal = preferences.get(
            "favorite_animal",
            "friendly creature",
        )

        character_trait = preferences.get(
            "trait",
            "brave",
        )

        schema_example = (
            self._build_schema_example()
            .replace("{language}", language)
        )

        system_content = (
            "You are an expert children's story author and structured-data "
            "generator. "
            "Generate gentle, imaginative, age-appropriate stories. "
            "Return ONLY one valid JSON object. "
            "Never use Markdown. "
            "Never use code fences. "
            "Never add explanations before or after the JSON. "
            "Never include unsafe, frightening, violent, sexual, "
            "or otherwise inappropriate content. "
            "Follow every requested field and scene-count constraint exactly."
        )

        user_content = (
            f"Create a beautiful children's story for a "
            f"{age}-year-old {gender} named {name}.\n\n"

            f"Theme: {theme}\n"
            f"Favorite animal/creature: {favorite_animal}\n"
            f"Main character trait: {character_trait}\n"
            f"Language for narration/title text: {language}\n\n"

            f"The story must contain exactly {self.num_scenes} scenes.\n"
            f"Scenes must be chronological and numbered 1 through "
            f"{self.num_scenes}.\n\n"

            "SPEAKER RULES:\n"
            '- "child": the child is speaking.\n'
            '- "creature": the creature is speaking.\n'
            '- "both": both characters are speaking in the scene.\n'
            '- "narrator_only": no character is speaking.\n\n'

            "Use speaker values naturally. Do not automatically make "
            "every scene use the same speaker.\n\n"

            "NARRATION RULES:\n"
            "- Exactly two complete sentences per scene.\n"
            "- Write narration entirely in the requested language.\n"
            "- Keep the language simple and engaging for children.\n"
            "- Keep the story positive and emotionally safe.\n"
            "- Avoid violence and unsafe subjects.\n\n"

            "VISUAL PROMPT RULES:\n"
            "- Write visual_prompt in English.\n"
            "- Describe the scene visually, not as dialogue.\n"
            "- Use a colorful cinematic children's storybook style.\n"
            "- Maintain consistent character appearance across scenes.\n"
            "- Do not include text, logos, watermarks, weapons, violence, "
            "or inappropriate imagery.\n\n"

            "Return JSON matching this exact structural blueprint:\n\n"
            f"{schema_example}"
        )

        if strict:
            user_content += (
                "\n\nSTRICT RETRY MODE:\n"
                "Your previous answer failed validation. "
                "This attempt MUST contain only the raw JSON object. "
                "Do not output Markdown, commentary, or additional text. "
                "Ensure the JSON parses correctly."
            )

        if self._supports_chat_template:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ]

                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                return prompt, True

            except Exception as e:
                logger.debug(
                    f"Chat template failed ({e}); "
                    f"using manual prompt."
                )

        manual_prompt = (
            "<|system|>\n"
            f"{system_content}\n"
            "<|user|>\n"
            f"{user_content}\n"
            "<|assistant|>\n"
        )

        return manual_prompt, False

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def _temperature_for_attempt(self, attempt: int) -> float:
        if attempt <= 1:
            return self.base_temperature

        # Gradually reduce randomness after invalid generations.
        decay = 0.12 * (attempt - 1)

        return max(
            0.3,
            self.base_temperature - decay,
        )

    def _run_inference(
        self,
        prompt: str,
        temperature: float,
    ) -> str:

        if self.model is None or self.tokenizer is None:
            raise StoryGenerationError(
                "Story generation model is not initialized."
            )

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )

            if DEVICE == "cuda":
                model_device = getattr(
                    self.model,
                    "device",
                    torch.device("cuda"),
                )

                inputs = {
                    key: value.to(model_device)
                    for key, value in inputs.items()
                }

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=(
                        self.tokenizer.pad_token_id
                        or self.tokenizer.eos_token_id
                    ),
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[-1]

            # Decode only newly generated tokens. This is substantially more
            # reliable than trying to strip the original prompt from decoded
            # text, especially with chat templates.
            generated_tokens = outputs[0][input_length:]

            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            return self._clean_generated_text(generated_text)

        except Exception as e:
            raise StoryGenerationError(
                f"Story LLM inference failed: {e}"
            )

    def _clean_generated_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # Remove common accidental wrappers without modifying legitimate JSON.
        text = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```\s*$",
            "",
            text,
        )

        for marker in (
            "<|assistant|>",
            "<|endoftext|>",
            "[/INST]",
        ):
            if marker in text:
                text = text.split(marker)[-1].strip()

        return text.strip()

    # ------------------------------------------------------------------
    # EXTRACTION + VALIDATION
    # ------------------------------------------------------------------

    def _extract_and_validate(
        self,
        text: str,
        name: str,
        age: int,
    ) -> Optional[dict]:

        parsed = self._extract_json(text)

        if parsed is None:
            logger.debug("LLM output did not contain valid JSON.")
            return None

        if not self._validate_story_structure(parsed):
            logger.debug("Story structure validation failed.")
            return None

        if self._flag_unsafe_content(parsed):
            logger.warning(
                "Generated story failed child-safety screening."
            )
            return None

        self._normalize_story(parsed)

        if not self._validate_normalized_story(parsed):
            logger.debug(
                "Story failed validation after normalization."
            )
            return None

        return parsed

    def _extract_json(self, text: str) -> Optional[dict]:
        if not text:
            return None

        text = text.strip()

        candidates = [text]

        # Find the first plausible JSON object and then progressively test
        # shorter closing boundaries. This is more robust than a greedy
        # {.*} expression when the model adds trailing text.
        first_brace = text.find("{")

        if first_brace >= 0:
            candidate_text = text[first_brace:]

            decoder = json.JSONDecoder()

            try:
                parsed, _ = decoder.raw_decode(candidate_text)

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                pass

        # Fallback candidate extraction.
        matches = re.findall(
            r"\{[\s\S]*?\}",
            text,
        )

        candidates.extend(matches)

        for candidate in candidates:
            repaired = self._repair_json(candidate)

            try:
                parsed = json.loads(repaired)

                if isinstance(parsed, dict):
                    return parsed

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):
                continue

        return None

    def _repair_json(self, raw: str) -> str:
        s = str(raw).strip()

        s = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            s,
            flags=re.IGNORECASE,
        )

        s = re.sub(
            r"\s*```\s*$",
            "",
            s,
        )

        # Remove trailing commas.
        s = re.sub(
            r",\s*([}\]])",
            r"\1",
            s,
        )

        # Only perform simple balance repair. We deliberately do not attempt
        # aggressive quote repair because doing so can silently corrupt
        # otherwise valid story text.
        open_braces = s.count("{")
        close_braces = s.count("}")

        if open_braces > close_braces:
            s += "}" * (open_braces - close_braces)

        open_brackets = s.count("[")
        close_brackets = s.count("]")

        if open_brackets > close_brackets:
            s += "]" * (open_brackets - close_brackets)

        return s.strip()

    def _validate_story_structure(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False

        title = data.get("title")

        if not isinstance(title, str) or not title.strip():
            return False

        scenes = data.get("scenes")

        if not isinstance(scenes, list):
            return False

        if len(scenes) != self.num_scenes:
            return False

        required_fields = {
            "scene_number",
            "title",
            "mood",
            "speaker",
            "narration",
            "visual_prompt",
        }

        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                return False

            if not required_fields.issubset(scene.keys()):
                return False

            scene_number = scene.get("scene_number")

            try:
                if int(scene_number) != index:
                    return False
            except (TypeError, ValueError):
                return False

            title = str(scene.get("title", "")).strip()
            mood = str(scene.get("mood", "")).strip().lower()
            speaker = str(scene.get("speaker", "")).strip().lower()
            narration = str(scene.get("narration", "")).strip()
            visual_prompt = str(
                scene.get("visual_prompt", "")
            ).strip()

            if not title:
                return False

            if mood not in ALLOWED_MOODS:
                return False

            if speaker not in ALLOWED_SPEAKERS:
                return False

            if len(narration) < 5:
                return False

            if len(visual_prompt) < 10:
                return False

            # Prevent absurdly large model output from reaching later stages.
            if len(title) > 200:
                return False

            if len(narration) > 3000:
                return False

            if len(visual_prompt) > 2500:
                return False

        return True

    def _validate_normalized_story(self, data: dict) -> bool:
        return self._validate_story_structure(data)

    # ------------------------------------------------------------------
    # SAFETY
    # ------------------------------------------------------------------

    def _flag_unsafe_content(self, data: dict) -> bool:
        title = str(data.get("title", ""))

        if UNSAFE_RE.search(title):
            return True

        for scene in data.get("scenes", []):
            fields_to_check = (
                scene.get("title", ""),
                scene.get("narration", ""),
                scene.get("visual_prompt", ""),
            )

            for value in fields_to_check:
                if UNSAFE_RE.search(str(value)):
                    return True

        return False

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_story(self, data: dict) -> None:
        title = str(
            data.get("title", "")
        ).strip()

        data["title"] = title[:200]

        scenes = data.get("scenes", [])

        data["scenes"] = scenes[:self.num_scenes]

        for i, scene in enumerate(
            data["scenes"],
            start=1,
        ):
            scene["scene_number"] = i

            scene["title"] = str(
                scene.get("title", f"Scene {i}")
            ).strip()[:200]

            mood = str(
                scene.get("mood", "")
            ).strip().lower()

            scene["mood"] = (
                mood
                if mood in ALLOWED_MOODS
                else "happy"
            )

            speaker = str(
                scene.get("speaker", "")
            ).strip().lower()

            scene["speaker"] = (
                speaker
                if speaker in ALLOWED_SPEAKERS
                else self._default_speaker_for_index(i)
            )

            scene["narration"] = str(
                scene.get("narration", "")
            ).strip()

            scene["visual_prompt"] = str(
                scene.get("visual_prompt", "")
            ).strip()

    def _default_speaker_for_index(
        self,
        scene_index: int,
    ) -> str:

        cycle = (
            "child",
            "creature",
            "both",
        )

        return cycle[
            (scene_index - 1) % len(cycle)
        ]

    # ------------------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------------------

    def _fallback_story(
        self,
        name: str,
        age: int,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> dict:

        preferences = preferences or {}

        animal = preferences.get(
            "favorite_animal",
            "friendly creature",
        )

        trait = preferences.get(
            "trait",
            "brave",
        )

        theme = preferences.get(
            "theme",
            "magical adventure",
        )

        templates = [
            {
                "title": "A Magical Beginning",
                "mood": "magical",
                "speaker": "child",
                "narration": (
                    f"{name} was a {trait} child who discovered a "
                    f"shimmering path leading into a wonderful new world. "
                    f"With a curious smile, {name} decided to follow the "
                    f"path and discover where the magical adventure would lead."
                ),
                "visual_prompt": (
                    f"A {age}-year-old child named {name}, friendly and "
                    f"confident, discovering a glowing magical path in a "
                    f"beautiful storybook landscape, colorful cinematic "
                    f"children's animation, whimsical atmosphere, "
                    f"consistent character design, no text."
                ),
            },
            {
                "title": "A Wonderful Friend",
                "mood": "adventure",
                "speaker": "creature",
                "narration": (
                    f"Along the path, {name} met a friendly {animal} who "
                    f"was excited to explore the magical world together. "
                    f"They smiled, shared ideas, and began a cheerful journey "
                    f"through the sparkling forest."
                ),
                "visual_prompt": (
                    f"A cheerful {age}-year-old child named {name} walking "
                    f"through an enchanted forest with a friendly {animal}, "
                    f"warm cinematic lighting, colorful children's storybook "
                    f"illustration, whimsical magical environment, "
                    f"consistent characters, no text."
                ),
            },
            {
                "title": "The Happy Discovery",
                "mood": "exciting",
                "speaker": "both",
                "narration": (
                    f"Together, {name} and the friendly {animal} discovered "
                    f"a beautiful place filled with glowing flowers and "
                    f"twinkling lights. They celebrated their friendship "
                    f"and happily promised to return for another adventure."
                ),
                "visual_prompt": (
                    f"A joyful child named {name} and a friendly magical "
                    f"{animal} discovering a glowing garden beside a "
                    f"storybook castle, sparkling lights, colorful flowers, "
                    f"cinematic children's animation, happy magical ending, "
                    f"consistent characters, no text."
                ),
            },
        ]

        scenes = []

        for i in range(1, self.num_scenes + 1):
            base = templates[
                (i - 1) % len(templates)
            ]

            scenes.append(
                {
                    "scene_number": i,
                    "title": (
                        base["title"]
                        if i <= len(templates)
                        else f"{base['title']} {i}"
                    ),
                    "mood": base["mood"],
                    "speaker": base["speaker"],
                    "narration": base["narration"],
                    "visual_prompt": base["visual_prompt"],
                }
            )

        return {
            "title": (
                f"{name}'s Magical {theme.title()} Adventure"
            ),
            "scenes": scenes,
        }

    # ------------------------------------------------------------------
    # MEMORY
    # ------------------------------------------------------------------

    def _free_memory(self) -> None:
        gc.collect()

        if DEVICE == "cuda":
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug(
                    f"CUDA memory cleanup failed: {e}"
                )
