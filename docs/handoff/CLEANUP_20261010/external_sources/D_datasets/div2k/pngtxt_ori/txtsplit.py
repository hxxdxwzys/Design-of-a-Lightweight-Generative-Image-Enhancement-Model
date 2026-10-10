import re

# The content of the original text file
content = """
0001.png: ['The image features a large, pink starfish lying on a rocky seabed. The starfish is surrounded by various sea creatures, such as sponges and anemones, adding to the vibrant and colorful underwater scene. The image captures the beauty and diversity of the marine ecosystem, emphasizing the importance of preserving and protecting these habitats for future generations.']
0005.png: ['The image features a close-up view of a collection of white and green pebbles, arranged in a grid-like pattern. The pebbles are placed on top of each other, creating a textured surface. The arrangement of the pebbles resembles a patchwork quilt or a mosaic, emphasizing the use of light and color to create a visually appealing composition.']
0003.png: ['The image depicts an outdoor swimming pool surrounded by palm trees, located in a luxurious resort or hotel. The pool area is decorated with pink lounge chairs, adding a touch of elegance to the scene. The sun is shining brightly, casting a warm and inviting glow over the pool and surrounding area. In addition to the pool, there are several potted plants scattered throughout the scene, further enhancing the lush and tropical atmosphere.']
0002.png: ['In this image, a person is riding a horse near a body of water, surrounded by snow-capped mountains in the background. The scenery is breathtakingly beautiful, with the mountainous landscape reflecting in the water, creating a serene and peaceful atmosphere. The person on the horse seems to be enjoying the tranquil surroundings, perhaps taking a moment to appreciate the natural beauty of the area.']
0004.png: ["The image depicts a scenic view of a body of water surrounded by mountains and houses. The water is calm and reflects the surrounding buildings, creating a serene and peaceful atmosphere. There are multiple houses visible in the scene, with some located near the water's edge and others situated further away from the water. A boat can also be seen in the water, adding to the tranquility of the setting. Overall, the image conveys a sense of serenity and relaxation, inviting viewers to take a moment to appreciate the beauty of the natural surroundings."]
"""
with open('captions_crop_short.txt', 'r', encoding='utf-8') as file:
    content = file.read()
# Split the content into lines
lines = content.strip().split('\n')

# Regex pattern to match the filename and content
pattern = re.compile(r'(\d{4}\.png): \[(["\'])(.*)\2\]')

for line in lines:
    match = pattern.match(line)
    if match:
        filename = match.group(1).replace('.png', '.txt')
        print(filename)
        text = match.group(3)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)


print("Files have been successfully created.")
