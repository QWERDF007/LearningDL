import io


def png_infomation(path):
    """
    获取png图像的信息
    http://www.libpng.org/pub/png/spec/1.2/PNG-Structure.html
    png file signature: 137 80 78 71 13 10 26 10
    :param path:
    :return: H，W，C
    """
    f = open(path, 'rb')
    header = [int(v) for v in f.read(8)]
    if header[0] == 137 and header[1] == 80 and header[2] == 78 and header[3] == 71 and header[4] == 13 and \
            header[5] == 10 and header[6] == 26 and header[7] == 10:
        chunk_data_length_bytes = [int(v) for v in f.read(4)]
        chunk_type_bytes = [int(v) for v in f.read(4)]
        width_bytes = [int(v) for v in f.read(4)]
        height_bytes = [int(v) for v in f.read(4)]
        bit_depth = int(f.read(1)[0])
        color_type = int(f.read(1)[0])
        depth = 0
        # grayscale
        if color_type == 0:
            depth = 1
        # RGB
        elif color_type == 2:
            depth = 3
        # palette index
        elif color_type == 3:
            depth = 1
        # grayscale + alpha
        elif color_type == 4:
            depth = 2
        # RGBA
        elif color_type == 6:
            depth = 4
        width = 0
        width |= width_bytes[0] << 24
        width |= width_bytes[1] << 16
        width |= width_bytes[2] << 8
        width |= width_bytes[3]
        height = 0
        height |= height_bytes[0] << 24
        height |= height_bytes[1] << 16
        height |= height_bytes[2] << 8
        height |= height_bytes[3]
        f.close()
        return height, width, depth
    else:
        f.close()
        return None


def jpg_infomation(path):
    """
    获取jpg图像的信息
    https://www.ccoderun.ca/programming/2017-01-31_jpeg/
    0xffd8
    :param path:
    :return: H，W，C
    """
    f = open(path, 'rb')
    header = [int(v) for v in f.read(2)]
    if header[0] == 0xff and header[1] == 0xd8:
        while True:
            bit_0 = int(f.read(1)[0])
            # find S0F0 -> 0xffc0
            if bit_0 == 0xff:
                bit_1 = int(f.read(1)[0])
                if bit_1 == 0xc0:
                    break
        f.seek(3, io.SEEK_CUR)
        height_bytes = [int(v) for v in f.read(2)]
        width_bytes = [int(v) for v in f.read(2)]
        depth = int(f.read(1)[0])
        width = 0
        width |= width_bytes[0] << 8
        width |= width_bytes[1]
        height = 0
        height |= height_bytes[0] << 8
        height |= height_bytes[1]
        f.close()
        return height, width, depth
    else:
        f.close()
        return None
