from __future__ import division, print_function

import argparse
import cv2
# import dlib
import numpy as np
from src import transform, vgg
from src.utils import preserve_colors
import tensorflow as tf
from imutils.video import FPS
from threading import Thread
import os
import time
from mvnc import mvncapi as mvnc


parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('--graph', type=str, help='Checkpoint directory', default='ncs_style.graph')
parser.add_argument('-d', '--downsample', type=float, default=1, help='Downsample factor')
parser.add_argument('--video', type=str, help="Stream from input video file", default=None)
parser.add_argument('--video-out', type=str, help="Save to output video file", default=None)
parser.add_argument('--width', type=int, help='Webcam video width', default=None)
parser.add_argument('--height', type=int, help='Webcam video height', default=None)
parser.add_argument('--fps', type=int, help="Frames Per Second for output video file", default=10)
# parser.add_argument('--skip', type=int, help="Speed up processing by skipping frames", default=0)
parser.add_argument('--no-gui', action='store_true', help="Don't render the gui", default=False)
# parser.add_argument('--skip-fails', action='store_true', help="Don't render frames where no face is detected", default=False)
parser.add_argument('--scale', type=float, help="Scale the output image", default=1)
parser.add_argument('--zoom', type=float, help="Zoom factor", default=1)
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--concat', action='store_true', help="Concatenate image and stylized output", default=False)
parser.add_argument('--device', type=str,
                        dest='device', help='Device to perform compute on',
                        default='/cpu:0')
args = parser.parse_args()


class WebcamVideoStream:
    '''From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/'''
    def __init__(self, src=0, width=None, height=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)

        if width is not None and height is not None: # Both are needed to change default dims
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        (self.ret, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
            # otherwise, read the next frame from the stream
            (self.ret, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return (self.ret, self.frame)
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class FastStyle(object):
    def __init__(self, graph_file, img_shape, device_t):
        self.img_shape = img_shape

        # mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

        print("[INFO] finding NCS devices...")
        devices = mvnc.enumerate_devices()
        # if no devices found, exit the script
        if len(devices) == 0:
            print("[INFO] No devices found. Please plug in a NCS")
            quit()

        print("[INFO] found {} devices. device0 will be used. "
            "opening device0...".format(len(devices)))
        device = mvnc.Device(devices[0])
        device.open()

        # open the CNN graph file
        print("[INFO] loading the graph file into memory...")
        with open(graph_file, mode="rb") as f:
            graph_in_memory = f.read()
         
        # load the graph into the NCS
        print("[INFO] allocating the graph on the NCS...")
        # self.graph = device.AllocateGraph(graph_in_memory)
        self.graph = mvnc.Graph('graph1')
        self.input_fifo, self.output_fifo = self.graph.allocate_with_fifos(device, graph_in_memory)

    def predict(self, X):
        X = np.expand_dims(X, 0) / 255.
        X = X.astype(np.float32)

        start = time.time()

        # self.graph.LoadTensor(X.astype(np.float32), "user object")
        self.graph.queue_inference_with_fifo_elem(self.input_fifo, self.output_fifo, X, None)

        # (preds, userobj) = self.graph.GetResult()
        preds, userobj = self.output_fifo.read_elem()
        end = time.time()
        print("[INFO] inference took {:.5} seconds".format(end - start))
        print(preds.shape)

        # Get execution time
        inference_time = self.graph.get_option(mvnc.GraphOption.RO_TIME_TAKEN)
        print(sum(inference_time))
        # import IPython; IPython.embed()
        
        img = preds.reshape((224,224,3))
        # img = img[:,:,::-1]
        # img = preds[::3].reshape((224,224))
        # img = np.array([img,img,img]).transpose((1,2,0))
        # img = img.transpose((0,2,1))
        # img = preds.reshape(self.img_shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        print(img)

        return img


def main():
    while True:  # Cam sometimes doesn't open immediately, keep trying until it's ready
        if args.video is not None:
            cap = WebcamVideoStream(args.video).start()
        else:
            cap = WebcamVideoStream(args.video_source).start()
        _, frame = cap.read()
        print(frame)
        if frame is not None:
            break
        time.sleep(1)

    frame_resize = cv2.resize(frame, None, fx=1 / args.downsample, fy=1 / args.downsample)
    img_shape = frame_resize.shape
    img_shape = (224,224,3)
    fast_style = FastStyle(args.graph, img_shape, args.device)

    if args.video_out is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if args.concat:
            shp = (int(2*img_shape[1]*args.scale),int(img_shape[0]*args.scale))
        else:
            shp = (int(img_shape[1]*args.scale),int(img_shape[0]*args.scale))
        out = cv2.VideoWriter(args.video_out, fourcc, args.fps, shp)

    fps = FPS().start()

    count = 0

    while(True):
        ret, frame = cap.read()

        if ret is True:       
            if args.zoom > 1:
                o_h, o_w, _ = frame.shape
                frame = cv2.resize(frame, None, fx=args.zoom, fy=args.zoom)
                h, w, _ = frame.shape
                off_h, off_w = int((h - o_h) / 2), int((w - o_w) / 2)
                frame = frame[off_h:h-off_h, off_w:w-off_w, :]

            # resize image and detect face
            frame_resize = cv2.resize(frame, None, fx=1 / args.downsample, fy=1 / args.downsample)
            count += 1
            print("Frame:",count,"Shape:",frame_resize.shape)

            # generate prediction

            image_rgb = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR 
            image_rgb = cv2.resize(image_rgb[:480,:480,:], (224,224))

            # cv2.imshow('test', image_rgb)

            # Run the frame through the style network
            styled_rgb = fast_style.predict(image_rgb)

            # if args.keep_colors:
            #     # Preserve the color of the content image
            #     styled_rgb = preserve_colors(image_rgb, styled_rgb)
            
            if args.concat:
                combined_image = np.concatenate([image_rgb, styled_rgb], axis=1)
            else:
                combined_image = styled_rgb
            
            combined_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                
            if args.scale != 1:
                combined_bgr = cv2.resize(combined_bgr, None, fx=args.scale, fy=args.scale)

            if args.video_out is not None:
                out.write(combined_bgr)

            if args.no_gui is False:
                cv2.imshow('fast style', combined_bgr)

            fps.update()

            key = cv2.waitKey(10) 
            if key & 0xFF == ord('q'):
                break
        else:
            # We're done here
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.stop()
    
    if args.video_out is not None:
        out.release()
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
