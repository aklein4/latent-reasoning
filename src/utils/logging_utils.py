
class LogSection:

    def __init__(self, name: str):
        """ A context manager for logging sections of code.
         - prints a starting message when entering
         - prints a finishing message when exiting

        Args:
            name (str): name of the section
        """
        self.name = name
    
    
    def __enter__(self):
        print(f"Starting {self.name}...", flush=True)
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            print(f"Finished {self.name}.", flush=True)
        else:
            raise exc_type(exc_value).with_traceback(traceback)