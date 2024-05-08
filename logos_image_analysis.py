from PIL import Image

def image_to_patches(image_path, output_folder):
    # Open the image
    img = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Calculate the dimensions of each sub-image
    sub_width = width // 5
    sub_height = height // 6
    
    # Loop through the grid and save each sub-image
    for i in range(6):
        for j in range(5):
            # Define the coordinates to crop the sub-image
            left = j * sub_width
            upper = i * sub_height
            right = left + sub_width
            lower = upper + sub_height
            
            # Crop the sub-image
            sub_img = img.crop((left, upper, right, lower))
            
            # Save the sub-image
            sub_img.save(f"{output_folder}/subimage_{i}_{j}.png")

# Example usage:
input_image_path = "./data/logos/nba_logos.jpg"  
output_folder_path = "./data/logos"

image_to_patches(input_image_path, output_folder_path)