import scipy.misc, numpy as np, os, sys
import cv2

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

def exists(p, msg):
    assert os.path.exists(p), msg

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files

def preserve_colors(content_rgb, styled_rgb):
    """Extract luminance from styled image and apply colors from content"""
    if content_rgb.shape != styled_rgb.shape:
      new_shape = (content_rgb.shape[1], content_rgb.shape[0])
      styled_rgb = cv2.resize(styled_rgb, new_shape)
    styled_yuv = cv2.cvtColor(styled_rgb, cv2.COLOR_RGB2YUV)
    Y_s, U_s, V_s = cv2.split(styled_yuv)
    image_YUV = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2YUV)
    Y_i, U_i, V_i = cv2.split(image_YUV)
    styled_rgb = cv2.cvtColor(np.stack([Y_s, U_i, V_i], axis=-1), cv2.COLOR_YUV2RGB)
    return styled_rgb

def luminance_only(x, y):
    w = np.asarray([0.114, 0.587, 0.299], dtype=np.float32)
    x_shape = x.shape
    y_shape = y.shape

    x = x.reshape(x_shape[:2] + (-1,))
    xl = np.zeros((x.shape[0], 1, x.shape[2]), dtype=np.float32)
    for i in range(len(x)):
        xl[i,:] = w.dot(x[i])
    xl_mean = np.mean(xl, axis=2, keepdims=True)
    xl_std = np.std(xl, axis=2, keepdims=True)

    y = y.reshape(y_shape[:2] + (-1,))
    yl = np.zeros((y.shape[0], 1, y.shape[2]), dtype=np.float32)
    for i in range(len(y)):
        yl[i,:] = w.dot(y[i])
    yl_mean = np.mean(yl, axis=2, keepdims=True)
    yl_std = np.std(yl, axis=2, keepdims=True)

    xl = (xl - xl_mean) / xl_std * yl_std + yl_mean
    # return np.repeat(xl, 3, axis=1).reshape(x_shape)
    return xl