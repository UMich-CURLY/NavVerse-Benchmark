class BaseAgent:
    def __init__(self, args):
        self.args = args
    def reset(self):
        raise NotImplementedError
    def act(self, obs, step_data_queue=None):
        raise NotImplementedError
    def shutdown(self):
        """Close all child processes and threads and release all resources (e.g. shared memory, queues, etc.)"""
        pass