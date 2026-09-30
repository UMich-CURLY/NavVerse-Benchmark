from baselines.base.base_agent import BaseAgent

import os

import numpy as np
import torch

from uninavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from uninavid.conversation import conv_templates, SeparatorStyle
from uninavid.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from uninavid.model.builder import load_pretrained_model


seed = 30
torch.manual_seed(seed)
np.random.seed(seed)
_BASELINES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UniNaVidAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.l_step = 0
        print("Initializing UniNaVid")
        model_path = os.environ.get("UNINAVID_MODEL_PATH") or os.path.join(
            _BASELINES_DIR, "submodules/Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
        )

        self.conv_mode = "vicuna_v1"

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))

        assert self.image_processor is not None

        print("Initialization Complete")

        self.rgb_list = []
        self.count_id = 0
        self.vis_frames = {}
        self.action_type = args.action_type
        self.reset()
        


    def reset(self):
        self.l_step = 0
        print("UniNaVidAgent reset")

        self.transformation_list = []
        self.rgb_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.first_forward = False
        self.executed_steps = 0
        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0

        self.vis_frames = {}


    def act(self, obs, step_data_queue=None):

        rgb = obs["rgb"]
        
        self.rgb_list.append(rgb)

        navigation_qs = self._navigation_prompt(obs)
        
        navigation = self.predict_inference(navigation_qs)
                
        action_list = navigation.split(" ")
                        
        if len(action_list)==0:
            raise ValueError("No action found in the output")

        self.l_step += 1
        print(f"UniNaVidAgent act: step={self.l_step}")

        # Send a list of actions
        if self.action_type == "multi":
            actions = []
            for action in action_list:
                if action == "stop":
                    actions.append("STOP")
                elif action == "forward":
                    actions.append("FORWARD")
                elif action == "right":
                    actions.append("TURN_RIGHT")
                elif action == "left":
                    actions.append("TURN_LEFT")
            print("Current action list:", actions)
            return {"action": actions}
        # Send single action
        else:
            if action_list[0] == "stop":
                return {"action": "STOP"}
            elif action_list[0] == "forward":
                return {"action": "FORWARD"}
            elif action_list[0] == "right":
                return {"action": "TURN_RIGHT"}
            elif action_list[0] == "left":
                return {"action": "TURN_LEFT"}

    """ UniNaVid Helper Functions """
    @staticmethod
    def _navigation_prompt(obs):
        target_type = obs.get("target_type", "obj")
        instruction_type = obs.get("instruction_type", "target")
        if target_type not in {"obj", "place"} or instruction_type not in {"target", "vln"}:
            raise ValueError(f"Unsupported navigation instruction: {instruction_type}/{target_type}")
        goal = obs["objectgoal"]
        if instruction_type == "vln":
            destination = "target object" if target_type == "obj" else "destination place"
            task = f"Follow this route to the {destination} and stop there: {goal}"
        elif target_type == "place":
            task = f"Navigate to the place described as '{goal}', and stop when you reach it"
        else:
            task = f"Search for a '{goal}', and stop by it"
        return (
            "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical "
            "observations and an image of the current observation <image>. Your assigned task is: "
            f"{task}. Analyze this series of images to determine your next four actions. "
            "The predicted action should be one of the following: forward, left, right, or stop."
        )

    def process_images(self, rgb_list):

        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(batch_image, return_tensors='pt')['pixel_values'].half().cuda()

        return [video]

    def predict_inference(self, prompt):
        
        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IMAGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IMAGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.5,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs
