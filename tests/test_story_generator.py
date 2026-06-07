import sys
from pathlib import Path
from datetime import datetime
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class TestStoryGeneratorFallback:
    def _get_generator(self):
        from pipeline.story_generator import StoryGenerator
        gen = object.__new__(StoryGenerator)
        gen.language = "Albanian"
        return gen

    def test_fallback_story_structure(self):
        gen   = self._get_generator()
        story = gen._generate_fallback_story("Erind", 6, "Albanian", "superhero")

        assert "title"   in story
        assert "scenes"  in story
        assert len(story["scenes"]) >= 3
        assert story["language"] == "Albanian"

    def test_fallback_has_narration(self):
        gen   = self._get_generator()
        story = gen._generate_fallback_story("Erind", 6, "Albanian", "superhero")

        for scene in story["scenes"]:
            assert "narration" in scene
            assert len(scene["narration"]) > 0

    def test_theme_by_age(self):
        gen = self._get_generator()
        theme_3  = gen._pick_theme(3)
        theme_8  = gen._pick_theme(8)
        theme_12 = gen._pick_theme(12)

        assert isinstance(theme_3, str) and len(theme_3) > 0
        assert isinstance(theme_8, str) and len(theme_8) > 0
        assert isinstance(theme_12, str) and len(theme_12) > 0

    def test_parse_valid_json(self):
        import json
        gen = self._get_generator()

        story_dict = {
            "title": "Test Film",
            "scenes": [
                {"index": 1, "title": "S1", "narration": "Tekst.", "mood": "happy"}
            ],
            "language": "Albanian",
        }
        raw = json.dumps(story_dict)
        result = gen._parse_story(raw, "Erind", 6, "Albanian", "superhero")
        assert result["title"] == "Test Film"

    def test_parse_json_with_extra_text(self):
        import json
        gen = self._get_generator()

        story_dict = {
            "title": "Test Film",
            "scenes": [{"index": 1, "title": "S1", "narration": "T.", "mood": "happy"}],
            "language": "Albanian",
        }
        raw = f"Ja historia:\n```json\n{json.dumps(story_dict)}\n```"
        result = gen._parse_story(raw, "Erind", 6, "Albanian", "adventure")
        assert result["title"] == "Test Film"

    def test_validate_story_fills_defaults(self):
        gen = self._get_generator()
        story = {
            "title"   : "Test",
            "language": "Albanian",
            "scenes"  : [
                {"index": 1, "narration": "Tekst 1."},
                {"index": 2, "narration": "Tekst 2."},
            ]
        }
        gen._validate_story(story)
        for scene in story["scenes"]:
            assert "mood"          in scene
            assert "duration_sec"  in scene
            assert "visual_prompt" in scene

    def test_enrich_with_visual_prompts(self):
        gen   = self._get_generator()
        story = {
            "title"   : "Test",
            "language": "Albanian",
            "scenes"  : [
                {"index": 1, "visual_description": "A child running", "mood": "happy"},
                {"index": 2, "visual_description": "A castle",        "mood": "magical"},
            ]
        }
        enriched = gen._enrich_with_visual_prompts(story, "Erind", 7)
        for scene in enriched["scenes"]:
            assert "visual_prompt"   in scene
            assert "negative_prompt" in scene
            assert "pixar" in scene["visual_prompt"].lower()