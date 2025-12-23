from PIL import Image
from PIL import Image, ImageDraw

def q_1_5(input_path):
    file = open(input_path + "label.txt", 'r')
    lines = file.readlines()
    img_count = 0

    images = []
    xs = []
    ys = []
    hs = []
    ws = []

    mark_x=[]
    mark_y=[]

    image = None
    for l in lines:
        # get to next image
        if (l[0] == "#"):
            img_count+=1
            if(img_count>5):
                break

            if(len(xs)<=0):
                img_name = l.split(' ')
                image = Image.open(input_path + 'images/' + img_name[1][:-1])
                continue

            point_radius = 3
            for i in range(len(xs)):
                draw = ImageDraw.Draw(image)
                draw.rectangle([xs[i], ys[i],
                                xs[i]+ws[i], ys[i]+hs[i]], outline=(0,255,0))

            for i in range(len(mark_x)):
                x = mark_x[i]
                y = mark_y[i]
                box = (
                    x - point_radius,
                    y - point_radius,
                    x + point_radius,
                    y + point_radius
                )
                draw.ellipse(box, fill=(255,0,0), outline=(255,0,0))

            # show image
            images.append(image)

            # for next
            img_name = l.split(' ')
            image = Image.open(input_path + 'images/' + img_name[1][:-1])
            xs = []
            ys = []
            hs = []
            ws = []
            mark_x = []
            mark_y = []

        else:
            lang_split = l.split(" ")
            x, y, w, h = lang_split[:4]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            land_xs = lang_split[4:-1:3]
            land_ys = lang_split[5:-1:3]
            for i in range(len(land_xs)):
                if(float(land_xs[i])>0):
                    mark_x.append(float(land_xs[i]))
                if (float(land_ys[i]) > 0):
                    mark_y.append(float(land_ys[i]))


            if (x >= 0 and y >= 0 and w >= 0 and h >= 0):
                xs.append(x)
                ys.append(y)
                hs.append(h)
                ws.append(w)

                # print(mark_x)
                # print(mark_y)

    widths,heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    all_image = Image.new("RGB", (max_width*2, max_height*2))
    x_offset = 0
    y_offset = 0
    i = 0
    for image in images:
        i+=1
        all_image.paste(image, (x_offset, y_offset))
        image.show()
        if(i==2):
            x_offset = 0
            i=0
            y_offset+=max_height
        else:
            x_offset += image.width

    all_image.show()

q_1_5('../widerface_homework/train/')