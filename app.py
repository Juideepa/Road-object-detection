from collections import defaultdict
import cv2
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")  # put correct path if needed

# Custom class names
class_names = [
    'Trafic Light Signal',
    'Stop Signal',
    'Speedlimit Signal',
    'Crosswalk Signal',
    'Crosswalk',
    'Pedestrian',
    'Bus',
    'Car',
    'Truck'
]

st.title("🚦 Road Object Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

def count_objects(results, class_names, image):
    object_count = defaultdict(int)

    for result in results:
        # Override YOLO class names
        result.names = {i: name for i, name in enumerate(class_names)}

        for det in result.boxes:
            class_id = int(det.cls[0])

            if class_id < len(class_names):
                class_name = class_names[class_id]
                object_count[class_name] += 1

                label = f"{object_count[class_name]} {class_name}"

                x1, y1, x2, y2 = map(int, det.xyxy[0])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    return image, object_count

if uploaded_file is not None:
    # Read image as numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # YOLO prediction
    results = model.predict(image)

    # Count & label objects
    labeled_image, counts = count_objects(results, class_names, image)

    # Show image
    st.image(labeled_image, caption="Detected Objects", use_container_width=True)

    # Show counts
    st.subheader("📊 Object Count")
    for k, v in counts.items():
        st.write(f"**{k}** : {v}")
