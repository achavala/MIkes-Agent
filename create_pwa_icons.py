#!/usr/bin/env python3
"""
Create PWA icons for Mike Agent
Generates simple icon files (you can replace with custom icons later)
"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create icons directory
os.makedirs('static', exist_ok=True)

# Create 192x192 icon
icon_192 = Image.new('RGB', (192, 192), color='#000000')
draw = Image.Draw(icon_192)
# Draw a simple "M" for Mike Agent
draw.rectangle([60, 40, 132, 152], fill='#00ff88')
draw.text((70, 70), "M", fill='#000000')
icon_192.save('static/icon-192.png')

# Create 512x512 icon
icon_512 = Image.new('RGB', (512, 512), color='#000000')
draw = Image.Draw(icon_512)
draw.rectangle([160, 100, 352, 392], fill='#00ff88')
draw.text((200, 200), "M", fill='#000000')
icon_512.save('static/icon-512.png')

print("✅ Created PWA icons:")
print("  - static/icon-192.png")
print("  - static/icon-512.png")
print("")
print("💡 You can replace these with custom icons later!")

