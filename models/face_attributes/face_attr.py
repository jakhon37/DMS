from deepface.DeepFace import analyze

# Example usage of the analyze function
# Assuming you have an image path or image data
img_path = "data/indoor_020.png"
result = analyze(img_path, actions=['age', 'gender', 'emotion', 'race'])

print("Analysis result:", result)
