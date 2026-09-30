import sys
import os
import time
import json
import signal
import select
from .logging import setup_tee
from navverse.utils.socket_client import request_data

class TaskQueue:
    def __init__(self, args, manager):
        self.args = args
        self.tasks_by_scene = manager.dict() # key: scene_id, value: list of tasks
        self.scene_ids = manager.list() # to keep track of available scenes
        self.total_tasks = manager.Value('i', 0)
        self.reserved_tasks = manager.dict() # worker_id -> task
        self.reserved_batches = manager.dict() # worker_id -> same-scene task list
        self.load_results()
    
    def load_results(self):
        result_path = f'{self.args.dump_location}/{self.args.exp_name}/result.json'
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = []
        self.result_labels = list(set([result['episode_label'] for result in self.results]))
    
    def add_tasks(self, tasks):
        # group by scene
        count = 0
        for task in tasks:
            scene_id, episode_label = task['scene_id'], task['episode_label']
            if episode_label in self.result_labels:
                print(f"[TaskQueue] Skipping task {episode_label} because it is already in the results.")
                continue
            existing = self.tasks_by_scene.get(scene_id, [])
            if len(existing) == 0:
                self.scene_ids.append(scene_id)
            existing.append(task)
            count += 1
            self.tasks_by_scene[scene_id] = existing
        self.total_tasks.value += count
        print(f"[TaskQueue] Added {count} tasks across {len(self.tasks_by_scene)} scenes.")

    def assign_task(self, worker_id, task):
        if str(worker_id) in self.reserved_batches:
            raise RuntimeError("Worker has a batch reservation; use release_tasks first")
        if task is None:
            self.reserved_tasks.pop(str(worker_id), None)
            return None
        self.reserved_tasks[str(worker_id)] = task
        return task

    def requeue_task(self, task, front=True):
        if task is None:
            return False
        episode_label = task["episode_label"]
        scene_id = task["scene_id"]
        if episode_label in self.result_labels:
            return False
        if any(
            reserved_task.get("episode_label") == episode_label
            for reserved_task in self.reserved_tasks.values()
        ):
            return False
        if any(
            reserved_task.get("episode_label") == episode_label
            for batch in self.reserved_batches.values()
            for reserved_task in batch
        ):
            return False
        existing = self.tasks_by_scene.get(scene_id, [])
        if any(queued_task.get("episode_label") == episode_label for queued_task in existing):
            return False
        if front:
            existing = [task] + list(existing)
        else:
            existing = list(existing) + [task]
        self.tasks_by_scene[scene_id] = existing
        if scene_id in self.scene_ids:
            self.scene_ids.remove(scene_id)
        if front:
            self.scene_ids.insert(0, scene_id)
        else:
            self.scene_ids.append(scene_id)
        print(f"[TaskQueue] Re-queued task {episode_label} for scene {scene_id}")
        print(f"[TaskQueue] Remaining tasks: {len(self)}, remaining scenes: {self.scene_ids}")
        return True

    def release_task(self, worker_id, episode_label=None):
        worker_key = str(worker_id)
        batch = self.reserved_batches.get(worker_key)
        if batch is not None:
            if episode_label is None:
                self.reserved_batches.pop(worker_key, None)
                return True
            remaining = [task for task in batch if task["episode_label"] != episode_label]
            if len(remaining) == len(batch):
                return False
            if remaining:
                self.reserved_batches[worker_key] = remaining
            else:
                self.reserved_batches.pop(worker_key, None)
            return True
        task = self.reserved_tasks.get(worker_key)
        if task is None:
            return False
        if episode_label is not None and task.get("episode_label") != episode_label:
            return False
        self.reserved_tasks.pop(worker_key, None)
        return True

    def release_tasks(self, worker_id):
        worker_key = str(worker_id)
        batch = self.reserved_batches.pop(worker_key, None)
        if batch is not None:
            return list(batch)
        task = self.reserved_tasks.pop(worker_key, None)
        return [task] if task is not None else []

    def _pop_from_scene(self, scene_id):
        tasks = self.tasks_by_scene.get(scene_id, [])
        if not tasks:
            return None
        task = tasks.pop(0)
        self.tasks_by_scene[scene_id] = tasks
        if not tasks and scene_id in self.tasks_by_scene:
            del self.tasks_by_scene[scene_id]
        if scene_id in self.scene_ids:
            self.scene_ids.remove(scene_id)
        print(f"[TaskQueue] Popped task from scene {scene_id}: {task['episode_label']}")
        print(f"[TaskQueue] Remaining tasks: {len(self)}, remaining scenes: {self.scene_ids}")
        return task

    def _pop_batch_from_scene(self, scene_id, max_tasks):
        tasks = self.tasks_by_scene.get(scene_id, [])
        batch = list(tasks[:max_tasks])
        if not batch:
            return []
        remaining = list(tasks[max_tasks:])
        if remaining:
            self.tasks_by_scene[scene_id] = remaining
        else:
            self.tasks_by_scene.pop(scene_id, None)
        if scene_id in self.scene_ids:
            self.scene_ids.remove(scene_id)
        print(f"[TaskQueue] Claimed {len(batch)} tasks from scene {scene_id}")
        return batch

    def _pop_first_scene(self):
        if not self.scene_ids:
            return None
        scene_id = self.scene_ids[0]
        self.scene_ids.remove(scene_id)
        return self._pop_from_scene(scene_id)

    def _pop_any_task(self):
        if not self.tasks_by_scene:
            return None
        scene_id = next(iter(self.tasks_by_scene.keys()))
        return self._pop_from_scene(scene_id)

    def get_task(self, worker_id, last_scene_id=None):
        if str(worker_id) in self.reserved_batches:
            raise RuntimeError("Worker has a batch reservation; use get_tasks")
        task = None
        if str(worker_id) in self.reserved_tasks:
            task = self.reserved_tasks.pop(str(worker_id))
            return task
        # 1. Sticky logic: Try to get task from last_scene_id
        if last_scene_id and last_scene_id in self.tasks_by_scene:
            task = self._pop_from_scene(last_scene_id)
        # 2. Fallback: Pick from first available scene
        if task is None:
            task = self._pop_first_scene()
        # 3. Final fallback: any remaining task
        if task is None:
            task = self._pop_any_task()
        return task

    def reserve_task(self, worker_id):
        if str(worker_id) in self.reserved_batches:
            raise RuntimeError("Worker has a batch reservation; use reserve_tasks")
        if str(worker_id) in self.reserved_tasks:
            return self.reserved_tasks[str(worker_id)]
        task = self._pop_first_scene()
        if task is None:
            task = self._pop_any_task()
        if task is not None:
            self.reserved_tasks[str(worker_id)] = task
        return task

    def reserve_tasks(self, worker_id, max_tasks, last_scene_id=None):
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        worker_key = str(worker_id)
        if worker_key in self.reserved_batches:
            return list(self.reserved_batches[worker_key])

        first = self.reserved_tasks.pop(worker_key, None)
        if first is None:
            if last_scene_id in self.tasks_by_scene:
                scene_id = last_scene_id
            elif self.scene_ids:
                scene_id = self.scene_ids[0]
            elif self.tasks_by_scene:
                scene_id = next(iter(self.tasks_by_scene))
            else:
                return []
            batch = self._pop_batch_from_scene(scene_id, max_tasks)
        else:
            scene_id = first["scene_id"]
            batch = [first] + self._pop_batch_from_scene(scene_id, max_tasks - 1)
        if not batch:
            return []
        self.reserved_batches[worker_key] = batch
        return list(batch)

    def get_tasks(self, worker_id, max_tasks, last_scene_id=None):
        return self.reserve_tasks(worker_id, max_tasks, last_scene_id)

    def complete_task(self, worker_id, episode_label=None):
        worker_key = str(worker_id)
        batch = self.reserved_batches.get(worker_key)
        if batch is not None:
            if episode_label is None:
                return False
            remaining = [task for task in batch if task["episode_label"] != episode_label]
            if len(remaining) == len(batch):
                return False
            if episode_label not in self.result_labels:
                self.result_labels.append(episode_label)
            if remaining:
                self.reserved_batches[worker_key] = remaining
            else:
                self.reserved_batches.pop(worker_key, None)
            return True
        task = self.reserved_tasks.get(worker_key)
        if episode_label is not None and episode_label not in self.result_labels:
            self.result_labels.append(episode_label)
        if task is None:
            return False
        if episode_label is not None and task.get("episode_label") != episode_label:
            return False
        self.reserved_tasks.pop(worker_key, None)
        return True
    
    def __len__(self):
        return (
            sum(len(tasks) for tasks in self.tasks_by_scene.values())
            + len(self.reserved_tasks)
            + sum(len(batch) for batch in self.reserved_batches.values())
        )
    
    def empty(self):
        return len(self) == 0

def input_thread_func(command_queue, reset_flag, quit_event):
    """Thread function to listen for keyboard input."""
    # ignore SIGINT, let the main process handle it
    while not quit_event.is_set():
        try:
            # This will block the input_thread, but not the main_thread
            rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
            if not rlist:
                continue
            command = rlist[0].readline().strip()
            if command == 'reset':
                print("Received reset command")
                reset_flag.set()
            elif command == 'quit':
                print("Received quit command")
                quit_event.set()
            else:
                # print(f"Received command: {command}")
                command_queue.put(command)
        except Exception as e:
            print(f"Input thread error: {e}")
            break
    print("Input thread finished.")


def foxglove_vis_thread_func(args, foxglove_log_queue, quit_event):
    setup_tee(args.log_file)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from .logging import FoxgloveVisualizer
    print(f"Starting foxglove visualizer on {args.foxglove_host}:{args.foxglove_port}")
    foxglove_vis = FoxgloveVisualizer(args, foxglove_log_queue)
    while not quit_event.is_set():
        foxglove_vis.run()

def save_result_thread_func(args, result_queue, quit_event):
    setup_tee(args.log_file)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while not quit_event.is_set() or not result_queue.empty():
        new_results = []
        while not result_queue.empty():
            try:
                new_results.append(result_queue.get_nowait())
            except Exception as e:
                print(f"Error: {e}")
                break
        if len(new_results) > 0:
            result_path = f'{args.dump_location}/{args.exp_name}/result.json'
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            all_results = []
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    all_results = json.load(f)
            all_results.extend(new_results)
            with open(result_path, 'w+') as f:
                json.dump(all_results, f, indent=4)
            print(f"Saved result to {result_path}")
        time.sleep(0.1)
    print("Save result thread finished.")

def load_tasks(args, control_port, task_queue, quit_event):
    if args.backend != "real_world":
        # load episode set list from server (real_world requires manual input)
        episode_set_list = None
        while episode_set_list is None and not quit_event.is_set():
            try:
                payload = request_data(args.host, args.port, request_type="GET_EPISODE_LIST", verbose=False)
                episode_set_list = payload
            except Exception:
                print(f"Waiting for episode list from {args.host}:{args.port}")
            time.sleep(1.0)
        if quit_event.is_set():
            return
        # check episode_set_list
        if len(episode_set_list["all"])==0:
            print(f"[ERROR] Received episode list from the server is invalid")
        
        # Determine valid labels to run
        target_labels = []
        all_episode_list = sorted(list(set([label for labels in episode_set_list.values() for label in labels])))
        if "," in args.episode_labels:
            # run specific episodes
            requested = args.episode_labels.split(",")
            target_labels = requested
        elif args.episode_labels in episode_set_list:
            # run specific split
            target_labels = episode_set_list[args.episode_labels]
        elif args.episode_labels == "all":
            # run all episodes
            target_labels = all_episode_list.copy()
        elif args.episode_labels in all_episode_list:
            # single label
            target_labels = [args.episode_labels]
        else:
            print(f"[ERROR] Episode label {args.episode_labels} not found.")
            return

        print(f"Loaded {len(target_labels)} tasks")
        
        # Populate TaskQueue
        tasks = []
        for episode_label in target_labels:
            scene_id = "_".join(episode_label.split('_')[:-1])
            episode_id = episode_label.split('_')[-1]
            tasks.append({
                "host": args.host,
                "control_port": control_port,
                "episode_label": episode_label,
                "scene_id": scene_id,
                "episode_id": episode_id
            })
            
        task_queue.add_tasks(tasks)
