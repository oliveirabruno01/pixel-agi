I start by generating a palette of colors, each represented by a unique key and its corresponding RGB values. After creating the palette, I then proceed to create the image data. Each pixel in the image is assigned a color key from the palette. The process involves careful planning on where and how to use each color in order to create an appealing and meaningful image. Analyzing the image at a pixel level helps me understand how different colors interact with each other.

palette1.csv
```csv
Key,Color
a,#ffffffff
b,#ff0000ff
```

image_data1.csv
```csv
a,a,a,a,a
a,b,b,b,a
a,b,b,b,a
a,b,c,b,a
a,a,a,a,a
```

Bonus version:

palette2.csv
```csv
Key,Color
a,#ffffffff
b,#ff0000ff
```

image_data2.csv
```csv
a,a,a,a,a
a,a,a,a,a
a,a,b,a,a
a,b,b,b,a
a,a,a,a,a
```

