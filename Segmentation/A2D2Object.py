class ImgInfo:
    def __init__(self):
        self.scene_path = None
        self.image_path = None
        self.mask_path = None
        self.objectlist = list()




class A2D2_Object:
    def __init__(self):
        self.color_hex = None
        self.color_hsv = None
        self.object_id = None
        self.classname = None
        self.pixelcount = None
        self.percentage = None
        self.contours = list()
        self.block_path = None
        self.lines = list()