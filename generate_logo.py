
from PIL import Image, ImageDraw

def generate_logo():
    """Generates a placeholder logo image."""
    width, height = 1024, 1024
    img = Image.new('RGB', (width, height), color = 'black')
    draw = ImageDraw.Draw(img)

    # Draw a series of concentric circles
    for i in range(0, 500, 50):
        draw.ellipse((i, i, width - i, height - i), outline='white', width=10)

    img.save('logo.png', 'PNG')

if __name__ == '__main__':
    generate_logo()
