import os
from cog import BasePredictor, Input, Path
from ultralytics import YOLO

class Predictor(BasePredictor):
    def setup(self):
        """Load the model during setup to avoid reloading for each prediction."""
        self.model = YOLO("yolo11n.pt")
    
    def predict(
        self,
        image: Path = Input(description="Path to an image"),
        epochs: int = Input(description="Number of epochs for training", default=100),
        img_size: int = Input(description="Training image size", default=640),
        device: str = Input(description="Device to use for training/inference", default="cpu"),
        export_format: str = Input(description="Format to export the model", default="onnx")
    ) -> str:
        """Run training, validation, detection, and model export."""
        
        # Train the model
        self.model.train(
            data="coco8.yaml",  # path to dataset YAML
            epochs=epochs,  # number of training epochs
            imgsz=img_size,  # training image size
            device=device  # device to run on
        )

        # Validate model performance
        self.model.val()

        # Perform object detection on an input image
        results = self.model(str(image))
        results[0].save(show=False)  # Save the result without displaying
        
        # Export the model
        export_path = self.model.export(format=export_format)
        
        return f"Model exported to {export_path}, object detection completed and saved."

