import sys
import time
from arguments import get_args
import signal
import threading
import multiprocessing as mp
from evaluation_worker import EvaluationWorker
from utils.logging import setup_tee
from utils.mp_utils import TaskQueue, load_tasks, save_result_thread_func, foxglove_vis_thread_func, input_thread_func
from utils.wandb_logging import save_step_data_thread_func

sys.stdout.reconfigure(line_buffering=True)
signal.signal(signal.SIGPIPE, signal.SIG_IGN)

# --- Configuration ---
def main():
    args, remaining_args = get_args()
    # print message to terminal while logging to a file
    setup_tee(args.log_file)
    control_port = args.port
    # pass the remaining arguments to the baselines
    sys.argv = [sys.argv[0]] + remaining_args
    args.process_label = 0
    # shared queues
    manager = mp.Manager()
    step_data_queue = mp.Queue()
    quit_event = mp.Event()
    result_queue = mp.Queue()
    
    # Use TaskQueue
    task_queue = TaskQueue(args, manager)
    
    foxglove_log_queue = mp.Queue()

    # create evaluation workers
    agent_list = []
    for gpu_id, num_processes in enumerate(args.gpu_assign):
        num_processes = int(num_processes)
        for process_id in range(num_processes):
            args.process_label = len(agent_list)
            args.gpu_id = gpu_id
            args.port = control_port + args.process_label
            # start the agent
            worker = EvaluationWorker(args, task_queue, result_queue, step_data_queue, foxglove_log_queue)
            agent_list.append(worker)

    # load tasks
    load_tasks(args, control_port, task_queue, quit_event)
    
    # handle SIGINT
    def handle_sigint(signum, frame):
        print("\nCaught Ctrl-C via signal, cleaning up...")
        quit_event.set()

    signal.signal(signal.SIGINT, handle_sigint)
    # use command line to control the first agent
    listener_thread = threading.Thread(name="CliInputThread", target=input_thread_func, args=[agent_list[0].command_queue, agent_list[0].reset_flag, quit_event])
    listener_thread.start()
    # start the save result process
    save_result_process = mp.Process(name="SaveResultProcess", target=save_result_thread_func, args=[args, result_queue, quit_event])
    save_result_process.start()
    save_step_data_process = None
    if args.save_step_data:
        save_step_data_process = mp.Process(
            name="SaveStepDataProcess",
            target=save_step_data_thread_func,
            args=[args, step_data_queue, quit_event],
        )
        save_step_data_process.start()
    # start the foxglove visualizer process
    if args.foxglove:
        foxglove_vis_process = mp.Process(name="FoxgloveVisProcess", target=foxglove_vis_thread_func, args=[args, foxglove_log_queue, quit_event])
        foxglove_vis_process.start()
    for worker in agent_list:
        worker.start()
    print("All evaluation workers started.")

    # block until all evaluation workers finish or SIGINT is received
    while not quit_event.is_set():
        if all(worker.evaluation_over.is_set() for worker in agent_list):
            print("All evaluation workers finished.")
            quit_event.set()
            break
        time.sleep(0.01)

    # quit all processes and threads
    print("Quitting...")
    quit_event.set()
    for worker in agent_list:
        if not worker.evaluation_over.is_set():
            worker.evaluation_over.set()
            start_time = time.time()
            while worker.is_alive():
                time.sleep(0.1)
                if time.time() - start_time > 5:
                    worker.terminate()
                    worker.join()
                    break
    print("Evaluation workers finished.")
    listener_thread.join()
    print("Listener thread finished.")
    save_result_process.join()
    print("Save result process finished.")
    if save_step_data_process is not None:
        save_step_data_process.join()
        print("Save step data process finished.")
    if args.foxglove:
        foxglove_vis_process.join()
        print("Foxglove visualizer process finished.")
    for q in (result_queue, step_data_queue, foxglove_log_queue):
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass
    print("Queues closed.")

    print("All processes finished.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
    print("Main finished.")
