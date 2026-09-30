import sys
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from dotenv import load_dotenv

# Add robot_env to sys.path
current_file = Path(__file__).resolve()
robot_env_path = current_file.parent.parent.parent
sys.path.append(str(robot_env_path))

try:
    from instruction_generator import InstructionGenerator
    from utils.episode import VLNEpisode
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    # Handle the case where imports fail, maybe mock for testing UI
    InstructionGenerator = None

app = FastAPI()

# Load environment variables
load_dotenv(robot_env_path / ".env")
load_dotenv() # Load from current directory as well

# Mount static files if needed (e.g., for serving images/videos)
# We will mount the episode_data_folder dynamically or on a specific path
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

templates = Jinja2Templates(directory=str(current_file.parent / "templates"))

# Global state
generator_instance: Optional[InstructionGenerator] = None
current_results: Dict[str, Any] = {}
episode_data_root: str = ""

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Get token from env
    token = os.getenv("GEMINI_API_KEY", "")
    
    # Default paths
    default_episodes = str(robot_env_path / "episodes")
    # Try to guess episode_data_folder or use a default
    default_data = str(robot_env_path / "data" / "episode_data")
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "token": token,
        "default_episodes_path": default_episodes,
        "default_data_path": default_data
    })

@app.post("/initialize")
async def initialize(
    episodes_path: str = Body(...),
    data_path: str = Body(...),
    token: str = Body(None)
):
    global generator_instance, episode_data_root
    
    if token:
        os.environ["GEMINI_API_KEY"] = token
    
    try:
        episode_data_root = data_path
        generator_instance = InstructionGenerator(episodes_path, data_path)
        
        # Mount the data path to serve images
        app.mount("/data", StaticFiles(directory=data_path), name="data")
        
        scenes = sorted(list(set([e.scene_id for e in generator_instance.episode_list])))
        return {"status": "success", "scenes": scenes, "message": "Initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/episodes/{scene_id}")
async def get_episodes(scene_id: str):
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Generator not initialized")
    
    episodes = [e.episode_id for e in generator_instance.episode_list if e.scene_id == scene_id]
    return {"episodes": episodes}

@app.get("/episodes_status/{scene_id}")
async def get_episodes_status(scene_id: str):
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Generator not initialized")
    
    status = generator_instance.get_episode_status(scene_id)
    return {"status": status}

@app.post("/generate_instruction")
def generate_instruction(
    scene_id: str = Body(...),
    episode_id: Optional[int] = Body(None),
    template_based: bool = Body(True),
    llm_based: bool = Body(False),
    video: bool = Body(True),
    force: bool = Body(False)
):
    global current_results
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Generator not initialized")
    
    try:
        # The generate_instruction method returns a list of results
        # We process one episode at a time based on user selection
        # If episode_id is None, it generates for all episodes in the scene
        results = generator_instance.generate_instruction(
            scene_id=scene_id,
            episode_id=episode_id,
            template_based=template_based,
            llm_based=llm_based,
            video=video,
            force=force
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No episodes found")
            
        # Helper to convert abs path to url
        def to_url(abs_path):
            if abs_path.startswith(episode_data_root):
                rel_path = abs_path[len(episode_data_root):].lstrip("/")
                return f"/data/{rel_path}"
            return ""

        processed_results = []
        for result in results:
            episode = result["episode"]
            ep_id = episode.episode_id
            
            # Store for next step
            current_results[str(ep_id)] = result
            
            # Find generated files
            video_path_pov = generator_instance.get_data_path(episode, "video_instruction_pov.mp4")
            video_path_third_person = generator_instance.get_data_path(episode, "video_instruction_third_person.mp4")
            path_image = generator_instance.get_data_path(episode, "path_simplified_path.png")
            
            processed_results.append({
                "episode_id": ep_id,
                "full_instruction": result["full_instruction"], # legacy
                "template_instruction": result.get("template_instruction", result["full_instruction"]),
                "aligned_instructions": result["aligned_instructions"],
                "improved_instruction": result.get("improved_instructions"),
                "prompt": result.get("prompt"),
                "used_images": [to_url(p) for p in result.get("used_images", [])],
                "video_url_pov": to_url(video_path_pov) if os.path.exists(video_path_pov) else None,
                "video_url_third_person": to_url(video_path_third_person) if os.path.exists(video_path_third_person) else None,
                "path_image_url": to_url(path_image) if os.path.exists(path_image) else None,
                "is_bad": result.get("is_bad", False),
                "checked": result.get("checked", False)
            })
            
        response = {
            "status": "success",
            "results": processed_results
        }
        
        # Backward compatibility for single episode
        if episode_id is not None and len(processed_results) == 1:
            res = processed_results[0]
            response.update({
                "full_instruction": res["full_instruction"],
                "template_instruction": res["template_instruction"],
                "aligned_instructions": res["aligned_instructions"],
                "improved_instruction": res.get("improved_instruction"),
                "prompt": res.get("prompt"),
                "used_images": res.get("used_images"),
                "video_url_pov": res["video_url_pov"],
                "video_url_third_person": res["video_url_third_person"],
                "path_image_url": res["path_image_url"],
                "checked": res.get("checked", False)
            })
            
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vlm_generation")
def vlm_generation(
    episode_id: int = Body(...),
    aligned_instructions: List = Body(...), # Allow overriding instructions
    prompt: Optional[str] = Body(None)
):
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Generator not initialized")
        
    try:
        # Retrieve the episode object
        # We could use the stored one, or re-fetch from list
        episode_res = current_results.get(str(episode_id))
        if not episode_res:
             # Try to find it in the list
             scene_episodes = [x for x in generator_instance.episode_list if x.episode_id == episode_id]
             if not scene_episodes:
                 raise HTTPException(status_code=404, detail="Episode not found")
             episode = scene_episodes[0]
        else:
            episode = episode_res["episode"]

        # Call VLM generation
        vlm_result = generator_instance.vlm_based_generation(episode, aligned_instructions, prompt=prompt)
        
        # Handle new dictionary return type
        if isinstance(vlm_result, dict):
            improved_instruction = vlm_result["improved_instructions"]
            actual_prompt = vlm_result["prompt"]
            used_images = vlm_result["used_images"]
        else:
             # Fallback if old code
             improved_instruction = vlm_result
             actual_prompt = ""
             used_images = []
        
        # Update current results
        if str(episode_id) not in current_results:
            current_results[str(episode_id)] = {"episode": episode}
        current_results[str(episode_id)]["improved_instructions"] = improved_instruction
        current_results[str(episode_id)]["aligned_instructions"] = aligned_instructions
        current_results[str(episode_id)]["prompt"] = actual_prompt
        
        # Convert image paths to URLs
        def to_url(abs_path):
            if abs_path.startswith(episode_data_root):
                rel_path = abs_path[len(episode_data_root):].lstrip("/")
                return f"/data/{rel_path}"
            return ""
            
        used_images_urls = [to_url(p) for p in used_images]

        return {
            "status": "success",
            "improved_instruction": improved_instruction,
            "prompt": actual_prompt,
            "used_images": used_images_urls
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_results")
async def save_results(
    episode_id: int = Body(..., embed=True),
    improved_instructions: Optional[str] = Body(None),
    aligned_instructions: Optional[List] = Body(None),
    prompt: Optional[str] = Body(None),
    is_bad: Optional[bool] = Body(None),
    checked: Optional[bool] = Body(None)
):
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Generator not initialized")
        
    result = current_results.get(str(episode_id))
    if not result:
        # Try to find it in the list if not in current_results
        scene_episodes = [x for x in generator_instance.episode_list if x.episode_id == episode_id]
        if not scene_episodes:
            raise HTTPException(status_code=404, detail="No results to save for this episode")
        episode = scene_episodes[0]
        result = {"episode": episode}
        
        # Load existing data if available to avoid overwriting with None
        save_path = generator_instance.get_data_path(episode, "generated_instructions.json")
        if os.path.exists(save_path):
            try:
                with open(save_path, "r") as f:
                    existing_data = json.load(f)
                    result.update(existing_data)
            except:
                pass
        
        # Store back to cache
        current_results[str(episode_id)] = result
    else:
        episode = result["episode"]
        
    try:
        save_path = generator_instance.get_data_path(episode, "generated_instructions.json")
        
        # Update server state if new data provided
        if improved_instructions is not None:
            result["improved_instructions"] = improved_instructions
        if aligned_instructions is not None:
            result["aligned_instructions"] = aligned_instructions
        if prompt is not None:
            result["prompt"] = prompt
        if is_bad is not None:
            result["is_bad"] = is_bad
        if checked is not None:
            result["checked"] = checked

        data_to_save = {
            "aligned_instructions": result.get("aligned_instructions"),
            "improved_instructions": result.get("improved_instructions"),
            "full_instruction": result.get("full_instruction"),
            "template_instruction": result.get("template_instruction", result.get("full_instruction")),
            "prompt": result.get("prompt"),
            "used_images": result.get("used_images"),
            "is_bad": result.get("is_bad", False),
            "checked": result.get("checked", False)
        }
        
        with open(save_path, "w") as f:
            json.dump(data_to_save, f, indent=2)
            
        return {"status": "success", "path": save_path}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_results")
async def delete_results(
    episode_ids: List[int] = Body(..., embed=True)
):
    if not generator_instance:
        raise HTTPException(status_code=400, detail="Generator not initialized")
    
    deleted = []
    errors = []
    
    for ep_id in episode_ids:
        try:
            scene_episodes = [x for x in generator_instance.episode_list if x.episode_id == ep_id]
            if not scene_episodes:
                errors.append(f"Episode {ep_id} not found")
                continue
            episode = scene_episodes[0]
            save_path = generator_instance.get_data_path(episode, "generated_instructions.json")
            if os.path.exists(save_path):
                os.remove(save_path)
                deleted.append(ep_id)
            else:
                errors.append(f"Episode {ep_id} has no results to delete")
        except Exception as e:
            errors.append(f"Error deleting episode {ep_id}: {str(e)}")
            
    return {"status": "success", "deleted": deleted, "errors": errors}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

