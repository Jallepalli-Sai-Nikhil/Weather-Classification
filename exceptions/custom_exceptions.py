class ModelLoadError(Exception):
    """Custom exception for model loading error."""
    def __init__(self,message="Failed to load model or preprocessor."):
        self.message = message
        super().__init__(self.message)