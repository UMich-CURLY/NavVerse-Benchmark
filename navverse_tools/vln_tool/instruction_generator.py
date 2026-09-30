import os
import glob
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from litellm import completion
import json
import base64
import sys
sys.path.append("../../")
from utils.path_following_utils import calc_yaw, wrap_to_pi, world_to_body
from utils.episode import VLNEpisode

class InstructionGenerator:
    def __init__(self, episode_folder, episode_data_folder):
        self.episode_folder = episode_folder
        self.episode_data_folder = episode_data_folder
        self.episode_list = VLNEpisode.from_json_folder(episode_folder)
    
    def get_data_path(self, episode, *children):
        scene_label = "_".join(episode.episode_label.split("_")[:-1])
        episode_name = "episode_" + str(episode.episode_id)
        return os.path.join(self.episode_data_folder, scene_label, episode_name, *children)

    def get_episode_status(self, scene_id):
        scene_episodes = [x for x in self.episode_list if x.scene_id == scene_id]
        status = []
        for episode in scene_episodes:
            save_path = self.get_data_path(episode, "generated_instructions.json")
            has_result = os.path.exists(save_path)
            is_checked = False
            is_bad = False
            if has_result:
                try:
                    with open(save_path, "r") as f:
                        data = json.load(f)
                        is_checked = data.get("checked", False)
                        is_bad = data.get("is_bad", False)
                except:
                    pass
            status.append({
                "episode_id": episode.episode_id,
                "has_result": has_result,
                "is_checked": is_checked,
                "is_bad": is_bad
            })
        return status

    def generate_instruction(
        self,
        scene_id,
        episode_id=None,
        template_based=True,
        llm_based=True,
        video=True,
        force=False
    ):
        # load episodes
        scene_episodes = [x for x in self.episode_list if x.scene_id == scene_id]
        if episode_id is not None:
            scene_episodes = [x for x in scene_episodes if x.episode_id == episode_id]
        # template based generation
        result = []
        for episode in scene_episodes:
            # Check if result already exists
            save_path = self.get_data_path(episode, "generated_instructions.json")
            loaded_data = None
            
            if os.path.exists(save_path) and not force:
                print(f"Loading existing result from {save_path}")
                try:
                    with open(save_path, "r") as f:
                        loaded_data = json.load(f)
                except Exception as e:
                    print(f"Failed to load json: {e}")
                    loaded_data = {}
                print("Found keys:", [*loaded_data.keys()])


            # Validate loaded data
            has_template = False
            has_vlm = False
            
            if loaded_data:
                # Check template part
                if loaded_data.get("aligned_instructions") and loaded_data.get("template_instruction") and loaded_data.get("full_instruction"):
                    full_instruction = loaded_data.get("full_instruction")
                    aligned_instructions = loaded_data.get("aligned_instructions")
                    aligned_instruction_str = " ".join([instruction for _, instruction in aligned_instructions])
                    if aligned_instruction_str != full_instruction:
                        print(f"Instruction mismatch: {aligned_instruction_str} not in {full_instruction}, regenerating template")
                        has_template = False
                    else:
                        has_template = True
                
                # Check VLM part if requested
                if llm_based:
                    if loaded_data.get("improved_instructions") and loaded_data.get("prompt") and loaded_data.get("used_images"):
                        has_vlm = True
                else:
                    # If not requesting LLM, we don't care about VLM part
                    has_vlm = True
            
                if not has_template and not has_vlm:
                    print(f"No template or VLM found for episode {episode.episode_id}")
                    os.remove(save_path)

            # Decide whether to regenerate
            need_regen_template = not has_template
            need_regen_vlm = llm_based and not has_vlm

            def generate_video(episode, aligned_instructions, view_type="pov"):
                print(f"Regenerating video for episode {episode.episode_id}")
                find_instruction = lambda x: min(aligned_instructions, key=lambda y: abs(y[0]-x) if y[0]>=x else float('inf'))[1]
                self.generate_video(episode, info_func=lambda x: [f"Frame {x}: {find_instruction(x)}"], view_type=view_type)
            
            if not need_regen_template and not need_regen_vlm:
                # Everything we need is loaded
                full_instruction = loaded_data.get("full_instruction", "")
                if "template_instruction" in loaded_data:
                    full_instruction = loaded_data["template_instruction"]
                aligned_instructions = loaded_data.get("aligned_instructions", [])
                
                # ... check video/image (omitted for brevity, keep existing checks) ...
                # Check if video exists
                video_path = self.get_data_path(episode, f"video_instruction_pov.mp4")
                if video and not os.path.exists(video_path):
                    generate_video(episode, aligned_instructions, view_type="pov")
                video_path = self.get_data_path(episode, f"video_instruction_third_person.mp4")
                if video and not os.path.exists(video_path):
                    generate_video(episode, aligned_instructions, view_type="third_person")
                path_image = self.get_data_path(episode, "path_simplified_path.png")
                # ... check path image ...

                result.append({
                    "episode": episode,
                    "template_instruction": full_instruction,
                    "full_instruction": full_instruction, 
                    "aligned_instructions": aligned_instructions,
                    "improved_instructions": loaded_data.get("improved_instructions", ""),
                    "prompt": loaded_data.get("prompt", ""),
                    "used_images": loaded_data.get("used_images", []),
                    "is_bad": loaded_data.get("is_bad", False),
                    "checked": loaded_data.get("checked", False)
                })
                continue
            
            # If we are here, we need to regenerate something
            print(f"Regenerating for episode {episode.episode_id}. Template: {need_regen_template}, VLM: {need_regen_vlm}")
            is_bad = False
            if need_regen_template:
                full_instruction, aligned_instructions, is_bad = self.template_based_generation(episode)
                if video and not is_bad:
                    generate_video(episode, aligned_instructions, view_type="pov")
                    generate_video(episode, aligned_instructions, view_type="third_person")
            else:
                # Use loaded template
                full_instruction = loaded_data.get("template_instruction", loaded_data.get("full_instruction"))
                aligned_instructions = loaded_data["aligned_instructions"]
                # Still check video if we are reusing template
                video_path = self.get_data_path(episode, f"video_instruction_pov.mp4")
                if video and not os.path.exists(video_path):
                    print(f"Regenerating video for episode {episode.episode_id}")
                    find_instruction = lambda x: min(aligned_instructions, key=lambda y: abs(y[0]-x) if y[0]>=x else float('inf'))[1]
                    generate_video(episode, aligned_instructions, view_type="pov")
                    generate_video(episode, aligned_instructions, view_type="third_person")

            # Initialize VLM vars
            improved_instructions = loaded_data.get("improved_instructions", "") if loaded_data else ""
            prompt = loaded_data.get("prompt", "") if loaded_data else ""
            used_images = loaded_data.get("used_images", []) if loaded_data else []
            if not is_bad:
                is_bad = loaded_data.get("is_bad", False) if loaded_data else False
            is_checked = loaded_data.get("checked", False) if loaded_data else False

            if need_regen_vlm:
                print(f"Running VLM generation for episode {episode.episode_id}")
                if is_bad and force:
                    vlm_res = self.vlm_based_generation(episode, None, full_instruction=full_instruction)
                    improved_instructions = vlm_res["improved_instructions"]
                    prompt = vlm_res["prompt"]
                    used_images = vlm_res["used_images"]
                elif not is_bad:
                    vlm_res = self.vlm_based_generation(episode, aligned_instructions)
                    improved_instructions = vlm_res["improved_instructions"]
                    prompt = vlm_res["prompt"]
                    used_images = vlm_res["used_images"]

            # Save results (if we changed anything)
            data_to_save = {
                "aligned_instructions": aligned_instructions,
                "full_instruction": full_instruction,
                "template_instruction": full_instruction,
                "improved_instructions": improved_instructions,
                "prompt": prompt,
                "used_images": used_images,
                "is_bad": is_bad,
                "checked": is_checked
            }
            with open(save_path, "w") as f:
                json.dump(data_to_save, f, indent=2)
            print(f"Saved results to {save_path}")

            result.append({
                "episode": episode,
                "template_instruction": full_instruction,
                "full_instruction": full_instruction, # legacy support
                "aligned_instructions": aligned_instructions,
                "improved_instructions": improved_instructions,
                "prompt": prompt,
                "used_images": used_images,
                "is_bad": is_bad,
                "checked": is_checked
            })
        return result

    def template_based_generation(self, episode):
        robot_pos = episode["start_position"]
        robot_quat = episode["start_rotation"] #wxyz
        robot_quat = R.from_quat(list(robot_quat[1:])+[robot_quat[0]])
        robot_pose = np.eye(4)
        robot_pose[:3, 3] = robot_pos
        robot_pose[:3, :3] = robot_quat.as_matrix()
        closest_goal_idx = episode["closest_goal_idx"]
        closest_goal = episode["goals"][closest_goal_idx]
        reference_path = closest_goal["reference_path"]

        # simplify path
        self.generate_path_image(episode, reference_path, robot_pose, "reference_path")
        pose_diff = self._calc_diff(reference_path, robot_pose)
        to_remove = [] # indices to remove if diff is too small
        for i, diff in enumerate(pose_diff):
            # print(f"pose_diff {i}: ex_b={diff[0]:.2f}, yaw_diff={np.rad2deg(diff[2]):.2f}")
            if abs(diff[2])<np.deg2rad(10) or abs(diff[0])<0.5:
                if i==len(reference_path)-1:
                    to_remove.append(i-1)
                else:
                    to_remove.append(i)
        # print(f"to_remove: {to_remove}")
        simplified_path = [point for i, point in enumerate(reference_path) if i not in to_remove]
        self.generate_path_image(episode, simplified_path, robot_pose, "simplified_path")
        pose_diff = self._calc_diff(simplified_path, robot_pose)
        # for i, diff in enumerate(pose_diff):
        #     print(f"pose_diff {i}: ex_b={diff[0]:.2f}, yaw_diff={np.rad2deg(diff[2]):.2f}")

        # generate instruction
        verb_table = {
            (-45, 0): "Slightly turn right",
            (0, 45): "Slightly turn left",
            (-150,-45): "Turn right",
            (45,150): "Turn left",
            (150,-150): "Turn around",
        }
        instruction_list = [] # each diff corresponds to an instruction, arriving message at the end
        for ex_b, ey_b, yaw_diff in pose_diff:
            yaw_diff = np.rad2deg(yaw_diff)
            instruction = ""
            if abs(yaw_diff)<10:
                instruction = f"Move forward {ex_b:.2f} m."
            else:
                for k,v in verb_table.items():
                    if yaw_diff>k[0] and yaw_diff<k[1]:
                        instruction = f"{v}, then move forward {ex_b:.2f} m."
                        break
            instruction_list.append(instruction)
        instruction_list.append(f"You will arrive at the {episode['objnav']}.")
        template_instruction = " ".join(instruction_list)
        print(f"Template instruction: {template_instruction}")

        # read poses from recorded trajectory
        pose_file = self.get_data_path(episode, "pose.txt")
        with open(pose_file, 'r') as f:
            lines = [line[1:-2].split(",") for line in f.readlines()]
        pose_list = [np.array([float(x) for x in line]) for line in lines]
        # some times the idx is not unique, fix it here
        idx = 1
        unique_pose_list = []
        for i, pose in enumerate(pose_list):
            if idx!=pose[0]:
                idx = pose[0]
                unique_pose_list.append(pose_list[i-1][1:])
        unique_pose_list.append(pose_list[-1])
        pose_list = unique_pose_list

        # align instruction with poses
        aligned_instructions = []
        start_idx = 0
        for i, point in enumerate(simplified_path):
            for j in range(start_idx, len(pose_list)):
                pose = pose_list[j]
                if np.linalg.norm(pose[:2] - np.array(point[:2])) < 1.5:
                    aligned_instructions.append((j, instruction_list[i]))
                    start_idx = j+1
                    break
        aligned_instructions.append((len(pose_list)-1, instruction_list[-1]))
        print(f"Aligned instructions: {len(aligned_instructions)}/{len(instruction_list)}")
        
        is_bad = len(aligned_instructions) != len(instruction_list)
        if is_bad:
            print(f"Alignment failed, episode is labeled as bad: {len(aligned_instructions)}/{len(instruction_list)}")

        return template_instruction, aligned_instructions, is_bad
    
    def _get_SE2_pose(self, xyz_quat):
        yaw = R.from_quat(xyz_quat[3:]).as_euler('ZYX')[0]
        return [xyz_quat[0], xyz_quat[1], yaw]

    
    def _calc_diff(self, points, pose):
        current_pos = pose[:3, 3]
        current_yaw = R.from_matrix(pose[:3, :3]).as_euler('ZYX')[0]
        pose_diff = []
        for target_pos in points:
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            pos_diff = np.linalg.norm([dx, dy])
            if pos_diff < 0.1:
                # very close to next point, angle is not stable
                target_yaw = current_yaw
            else:
                target_yaw = calc_yaw(current_pos[:2], target_pos[:2])
            yaw_diff = wrap_to_pi(target_yaw - current_yaw)
            ex_b, ey_b = world_to_body(dx, dy, target_yaw)
            pose_diff.append([ex_b, ey_b, yaw_diff])
            current_pos = target_pos
            current_yaw = target_yaw
        return pose_diff
    
    def generate_path_image(self, episode, points, pose, path_name="test"):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        points = [pose[:3, 3]] + points
        ax.plot([point[0] for point in points], [point[1] for point in points])
        ax.plot(pose[:3, 3][0], pose[:3, 3][1], 'ro')
        image_path = self.get_data_path(episode, f"path_{path_name}.png")
        fig.savefig(image_path)
        plt.close(fig)
        print(f"Path image generated and saved to {image_path}")

    def generate_video(self, episode, info_func=None, view_type="pov"):
        """
        Generate video with info. info_func is a function func(frame_id) -> text_list
        """
        if info_func is None:
            info_func = lambda x: [f"Frame {x}"]
        # load image folder
        image_folder = self.get_data_path(episode, f"{view_type}_rgb")
        image_files = glob.glob(os.path.join(image_folder, "*.png"))
        image_files = sorted(image_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        images = [cv2.imread(x) for x in image_files]
        for frame_id in range(len(images)):
            text_list = info_func(frame_id)
            text_height = 100
            for text in text_list:
                cv2.putText(images[int(frame_id)], text, (10, text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
                text_height += 30
        # generate video
        video_path = self.get_data_path(episode, f"video_instruction_{view_type}.mp4")
        
        # Try avc1 (H.264) first
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10, (images[0].shape[1], images[0].shape[0]))
        
        if not video_writer.isOpened():
             print("Failed to open video writer with avc1, trying H264")
             fourcc = cv2.VideoWriter_fourcc(*'H264')
             video_writer = cv2.VideoWriter(video_path, fourcc, 10, (images[0].shape[1], images[0].shape[0]))

        if not video_writer.isOpened():
             print("Failed to open video writer with H264, trying vp09 (VP9)")
             fourcc = cv2.VideoWriter_fourcc(*'vp09')
             video_writer = cv2.VideoWriter(video_path, fourcc, 10, (images[0].shape[1], images[0].shape[0]))

        if not video_writer.isOpened():
             print("Failed to open video writer with vp09, falling back to mp4v")
             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
             video_writer = cv2.VideoWriter(video_path, fourcc, 10, (images[0].shape[1], images[0].shape[0]))

        
        for image in images:
            video_writer.write(image)
        video_writer.release()
        print(f"Video generated and saved to {video_path}")
    
    def vlm_based_generation(self, episode, aligned_instructions, prompt=None, full_instruction=None):
        message_content = []
        add_text = lambda text: message_content.append({"type": "text","text": text})
        add_image = lambda image_path: message_content.append({"type": "image_url","image_url": {"url": "data:image/png;base64,"+base64.b64encode(open(image_path, "rb").read()).decode()}})
        add_video = lambda video_path: message_content.append({"type": "video_url","video_url": {"url": "data:video/mp4;base64,"+base64.b64encode(open(video_path, "rb").read()).decode()}})
        # change the frame_id to when the instruction begins
        if aligned_instructions is not None:
            shifted_instructions = [(1, "You start at this position. "+aligned_instructions[0][1])]
            for i in range(1, len(aligned_instructions)):
                shifted_instructions.append((aligned_instructions[i-1][0], aligned_instructions[i][1]))
            image_interval = max(10, aligned_instructions[-1][0]//15)
            image_list = list(range(1, aligned_instructions[-1][0]+1, image_interval))
            print(f"Using {len(image_list)} images (interval={image_interval})")
            # add image idx to the prompt
            prompt_instruction = []
            for i, (keyframe_id, instruction) in enumerate(shifted_instructions):
                print(f"Add keyframe {keyframe_id//image_interval+1} at frame {keyframe_id}: {instruction}")
                prompt_instruction.append(f"(img {keyframe_id//image_interval+1}) {instruction}")
                # format_time = lambda x: f"{x//60:02d}:{x%60:02d}"
                # prompt_instruction.append(f"(video {format_time(keyframe_id//10)}) {instruction}")
            prompt_instruction = " ".join(prompt_instruction)
            # print(prompt_instruction)
        else:
            prompt_instruction = full_instruction
            #check image number
            image_folder = self.get_data_path(episode, "pov_rgb")
            image_files = glob.glob(os.path.join(image_folder, "*.png"))
            image_interval = max(10, len(image_files)//15)
            image_list = list(range(1, len(image_files)+1, image_interval))
            print(f"Using {len(image_list)} images")
        used_images = []
        for image_idx in image_list:
            image_url = self.get_data_path(episode, "pov_rgb", f"{image_idx}.png")
            # print(f"added image: {image_url}")
            add_image(image_url)
            used_images.append(image_url)
            
        # video_path = instruction_generator.get_data_path(episode, "video_instruction.mp4")
        # add_video(video_path)
        # print(f"added video: {video_path}")
        if prompt is None:
            text_prompt = f"""
Describe the trajectory of the robot in the images. Improve the given instruction to:
1) include unique landmarks you see in the images for guidance, describe color, shape, etc. briefly. Only mention landmarks that can be clearly observed.
2) make the expression more diverse.
3) replace the exact distance value with more general description.
4) do not mention (img x) in the output.
5) in some cases, if you don’t directly see the target in the image, just say “you will arrive at [target name]“.
6) do not mention the red lines in the images, those are pointing to the waypoints.
7) if the robot is following the road, mention it in the output.
8) make the instruction compact, only include the key landmarks.
9) return the output only, no other text.
10) make it within 100 words.
Improve this instruction: {prompt_instruction}
Example output: Go forward until you see the red building, then turn right and go until you see the blue car. You should be able to see the target mailbox on your right.
            """
        else:
            # If user provided a prompt, we assume it contains prompt_instruction or we just append it
            if "{prompt_instruction}" in prompt:
                text_prompt = prompt.format(prompt_instruction=prompt_instruction)
            elif "Improve this instruction" in prompt:
                text_prompt = prompt
            else:
                text_prompt = prompt + f"\nImprove this instruction: {prompt_instruction}"
        
        add_text(text_prompt)
        # print(text_prompt)
        resp = completion(
            model="gemini/gemini-3-pro-preview",
            messages=[{"role": "user", "content": message_content}],
            reasoning_effort="low",
        )
        improved_instructions = resp.choices[0].message.content
        print(f"Improved instructions: {improved_instructions}")
        return {
            "improved_instructions": improved_instructions,
            "prompt": text_prompt,
            "used_images": used_images
        }