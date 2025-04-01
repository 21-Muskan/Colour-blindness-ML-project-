
from predict import correct_image

# Test on different types of images
test_images = ["test1.jpg", "test2.png", "test3.jpg"]
for img in test_images:
    correct_image(
        input_path=f"static/{img}",
        model_path="models/color_correction.pth",
        output_path=f"static/corrected_{img}"
    )
    print(f"Processed {img} -> corrected_{img}")