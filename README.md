🎈 YOLO-Based Balloon Detection & Attribute Analysis

This project implements a real-time balloon detection system using YOLOv8.
The system performs:

🎯 Balloon detection with bounding boxes

📊 Confidence score visualization

🎨 Color extraction using HSV-based image processing

🔵 Shape analysis using contour-based circularity metrics

🎥 Real-time webcam inference

Unlike standard object detection demos, this project separates detection and attribute extraction into different layers:

Deep Learning Layer (YOLO) → Candidate object detection

Classical Computer Vision Layer → Color & shape analysis

Visualization Layer → Real-time annotated output

This hybrid architecture improves interpretability and reduces false positives while keeping the model lightweight.
