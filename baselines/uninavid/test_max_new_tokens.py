#!/usr/bin/env python
"""
Verify that lowering max_new_tokens (1024 -> 32) does not change UniNaVid's
predicted action sequence.

Approach: run the upstream offline test case (test_cases/vln_1) twice with
identical seeding — once with max_new_tokens=1024 and once with 32 — and
compare the decoded outputs step by step. The output for navigation is at most
~4 tokens followed by EOS, so 32 should be plenty.

Usage:
    cd ${NAVVERSE_ROOT}
    conda activate uninavid
    python baselines/uninavid/test_max_new_tokens.py
"""
import os
import sys
import time

import cv2
import numpy as np
import torch

_BASELINES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_DIR = os.path.dirname(_BASELINES_DIR)
sys.path.insert(0, _BASELINES_DIR)

from uninavid.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from uninavid.model.builder import load_pretrained_model


MODEL_PATH = os.path.join(
    _BASELINES_DIR,
    "submodules/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2",
)
TEST_CASE_DIR = os.path.join(
    _BASELINES_DIR,
    "submodules/Uni-NaVid/test_cases/vln_1",
)
SEED = 30
NUM_TEST_STEPS = 12  # subset of frames to keep runtime modest


def load_sequence(test_case_dir):
    img_dir = os.path.join(test_case_dir, "images")
    files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(".jpg")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    images = [cv2.imread(os.path.join(img_dir, f)) for f in files]
    import json

    with open(os.path.join(test_case_dir, "instruction.json")) as fh:
        instruction = json.load(fh)["instruction"]
    return images, instruction


class TestAgent:
    """Minimal copy of UniNaVidAgent.predict_inference for testing."""

    def __init__(self, tokenizer, model, image_processor):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_mode = "vicuna_v1"
        self.prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. You have been "
            "given a video of historical observations and an image of the current "
            "observation <image>. Your assigned task is: '{}'. Analyze this series "
            "of images to determine your next four actions. The predicted action "
            "should be one of the following: forward, left, right, or stop."
        )
        self.rgb_list = []

    def reset(self):
        self.rgb_list = []
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

    def process_images(self, rgb_list):
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = (
            self.image_processor.preprocess(batch_image, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
        return [video]

    def predict_inference(self, prompt, max_new_tokens):
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IMAGE_SEPARATOR = "<image_sep>"
        image_start = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IMAGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs.replace("<image>", "")
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        token_prompt = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end)
            new_list.append(image_start)
            new_list.append(image_end)
            new_list.append(navigation_special)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        with torch.inference_mode():
            self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.5,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        new_token_count = int(output_ids.shape[1] - input_token_len)
        return outputs.strip(), new_token_count


def run_sequence(agent, images, instruction, max_new_tokens, seed):
    """Run through `images` once and return (outputs, token_counts, wall_time)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    agent.reset()
    outputs = []
    counts = []
    t0 = time.time()
    for img in images:
        agent.rgb_list.append(img)
        prompt = agent.prompt_template.format(instruction)
        out, n = agent.predict_inference(prompt, max_new_tokens)
        outputs.append(out)
        counts.append(n)
    return outputs, counts, time.time() - t0


def main():
    print(f"Loading model from: {MODEL_PATH}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, get_model_name_from_path(MODEL_PATH)
    )
    assert image_processor is not None
    agent = TestAgent(tokenizer, model, image_processor)

    images, instruction = load_sequence(TEST_CASE_DIR)
    images = images[:NUM_TEST_STEPS]
    print(f"Loaded {len(images)} frames; instruction: '{instruction}'")
    print()

    print("=" * 70)
    print(f"Pass A: max_new_tokens=1024, seed={SEED}")
    print("=" * 70)
    outs_1024, counts_1024, t_1024 = run_sequence(
        agent, images, instruction, max_new_tokens=1024, seed=SEED
    )
    for i, (o, n) in enumerate(zip(outs_1024, counts_1024)):
        print(f"  step {i:2d} | tokens={n:3d} | '{o}'")
    print(f"Pass A total time: {t_1024:.2f}s\n")

    print("=" * 70)
    print(f"Pass B: max_new_tokens=32, seed={SEED}")
    print("=" * 70)
    outs_32, counts_32, t_32 = run_sequence(
        agent, images, instruction, max_new_tokens=32, seed=SEED
    )
    for i, (o, n) in enumerate(zip(outs_32, counts_32)):
        print(f"  step {i:2d} | tokens={n:3d} | '{o}'")
    print(f"Pass B total time: {t_32:.2f}s\n")

    print("=" * 70)
    print("Diff")
    print("=" * 70)
    n_match = 0
    n_diff = 0
    for i, (a, b) in enumerate(zip(outs_1024, outs_32)):
        if a == b:
            n_match += 1
        else:
            n_diff += 1
            print(f"  step {i:2d} DIFF: '{a}'  !=  '{b}'")
    print(f"\nMatched {n_match}/{len(outs_1024)}  Mismatched {n_diff}/{len(outs_1024)}")
    print(f"Max tokens generated in Pass A (max=1024): {max(counts_1024)}")
    print(f"Max tokens generated in Pass B (max=32):   {max(counts_32)}")
    print(f"Speedup: {t_1024 / max(t_32, 1e-6):.2f}x")

    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
