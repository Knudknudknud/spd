from PIL import Image

img = Image.open(r"C:\Users\bruger\uni\spd\thesis\written_product\Images\softmax_CC.png")

# Option 1: scale by a factor (e.g. half size)
small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)

# Option 2: target a specific width, keep aspect ratio
new_w = 800
new_h = round(img.height * new_w / img.width)
small = img.resize((new_w, new_h), Image.LANCZOS)

small.save("figure_small.png")