# Explainable-AI using Gradient Class Activation Map (Grad-CAM)

Ever wondered how large CNN models are able to best perform on the detecting objects?

This technique (Grad-CAM) reveals the exact spots in an image that drive the model's decision-making process. It's like having a window into the model's brain, showing you why it thinks what it thinks. Not only does it make AI more transparent and dependable, but it also lets you see how skilled the AI is at recognizing objects and patterns

# Visual Examples
![image](https://github.com/DurgaSandeep25/Grad-CAM-Explanaible-AI/assets/38128597/a9736a15-50c1-447e-8217-0c9c80744e4f)

# Objective
There are many advanced approaches in Grad-CAM (Ablation CAM, Grad-CAM++,..). In this project, I will be focussing on implementing the basic Grad-CAM, with minimal changes in your torch pipeline. Refer to grad_cam_compute_vision.py

# Model Architecture Overview
![image](https://github.com/DurgaSandeep25/Grad-CAM-Explanaible-AI/assets/38128597/8ad7ad1d-a8eb-4d92-8a38-dac440fbe7a1)

# Training and Evaluation
Curve is not smooth because of very small dataset during training and evaluation (due to GPU requirements)
![image](https://github.com/DurgaSandeep25/Grad-CAM-Explanaible-AI/assets/38128597/e2a2239c-77d6-455b-a410-46063c5570bb)


