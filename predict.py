import os
from cog import BasePredictor, Input, Path, File, BaseModel
from ultralytics import YOLO

class Output(BaseModel):
    file: File
    text: str

class Predictor(BasePredictor):
    def setup(self):
        """Load the model during setup to avoid reloading for each prediction."""
        self.model = YOLO("yolo11n.pt")
    
    def predict(
        self,
        image: Path = Input(description="Path to an image")
    ) -> Output:

        # Perform object detection on an input image
        results = self.model(str(image))
        results[0].save(show=False)  # Save the result without displaying
        
        return Output(file=File(path=results[0].save_dir), text="Model exported to {export_path}, object detection completed and saved.")

