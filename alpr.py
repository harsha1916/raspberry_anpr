import os 
import cv2
import time
import numpy as np
import tools.predict_system as predict_sys
import tools.predict_lpd as predict_lpd
from tools.fast_detector import ThreadedPlateDetector
from PIL import Image, ImageFont, ImageDraw
import re
import threading
import queue


class ALPR():
    def __init__(self, out_dir: str) -> None:
        self.text_system = predict_sys.TextSystem()
        self.lp_detector = predict_lpd.LisencePlateDetector()
        self.output_directory = out_dir

    def detect_lp(self,path: str, Bbox : bool, save : bool, show : bool):
        
        img = cv2.imread(path)
        Name = os.path.basename(path).split('.')[0]
        bbox, _, _, _ = self.lp_detector(img)
        if len(bbox):
            pts = bbox[0]
            if Bbox:
                xmin = int(min(pts[0]))
                ymin = int(min(pts[1]))
                xmax = int(max(pts[0]))
                ymax = int(max(pts[1]))
                cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (0, 255, 0), 2)
            else:
                pt1 = [int(pts[0][0]),int(pts[1][0])]
                pt2 = [int(pts[0][1]),int(pts[1][1])]
                pt3 = [int(pts[0][2]),int(pts[1][2])]
                pt4 = [int(pts[0][3]),int(pts[1][3])]
                ptslist = np.array([[pt1,pt2,pt3,pt4]],dtype=np.int32)
                cv2.drawContours(img, ptslist, -1, (0, 255, 0), 2)
            
            if save:
                cv2.imwrite('%s/%s_lpd.png' % (self.output_directory, Name), img)

            if show:
                cv2.imshow(Name,img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        else:
            print('No License Plates Detected')
    
    def blur_lp(self,path: str, save : bool, show : bool):
        
        img = cv2.imread(path)
        Name = os.path.basename(path).split('.')[0]
        max_len = max(img.shape[0],img.shape[1])
        w_k = int(0.2*max_len)
        if w_k%2 == 0:
            w_k = w_k+1
        blurred_img = cv2.GaussianBlur(img, (w_k,w_k), 0)
        mask = np.zeros(img.shape, dtype=np.uint8)
        bbox, _, _, _ = self.lp_detector(img)
        if len(bbox):
            pts = bbox[0]
            
            pt1 = [int(pts[0][0]),int(pts[1][0])]
            pt2 = [int(pts[0][1]),int(pts[1][1])]
            pt3 = [int(pts[0][2]),int(pts[1][2])]
            pt4 = [int(pts[0][3]),int(pts[1][3])]
            ptslist = np.array([[pt1,pt2,pt3,pt4]],dtype=np.int32)
            mask = cv2.fillPoly(mask,ptslist,(255,255,255))
            img = np.where(mask==0, img, blurred_img)
            
            if save:
                cv2.imwrite('%s/%s_blurred.png' % (self.output_directory, Name), img)

            if show:
                cv2.imshow(Name,img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        else:
            print('No License Plates Detected')
        
    def recognize_lp(self, path: str, save : bool, show : bool, f_scale : float):

        img = cv2.imread(path)
        Name = os.path.basename(path).split('.')[0]
        bbox, _,  LlpImgs, _ = self.lp_detector(img)
        if len(bbox):
            pts = bbox[0]
            xmin = int(min(pts[0]))
            ymin = int(min(pts[1]))
            xmax = int(max(pts[0]))
            ymax = int(max(pts[1]))
            Width = int(xmax-xmin)
            cv2.rectangle(img, (xmin,ymin),(xmax,ymax), (0, 255, 0), int(2/1.5*f_scale))
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            _, rec_res = self.text_system(Ilp*255.)
            print(rec_res)
            if rec_res is not None:
                text_sum = 0
                for text, score in rec_res[::-1]:
                    text = strip_chinese(text)
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 2)
                    text_w, text_h = text_size
                    text_sum+=text_h
                    img = draw_text(img, text,
                                    pos=(xmin, ymin-int(text_sum)),
                                    font=cv2.FONT_HERSHEY_PLAIN,
                                    font_scale=f_scale,
                                    text_color=(0, 0, 0),
                                    font_thickness=2,
                                    text_color_bg=(0, 255, 0)
                                    )
            
                if save:
                    cv2.imwrite('%s/%s_alpr.png' % (self.output_directory, Name), img)

                if show:
                    cv2.imshow(Name,img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print('No License Plates Detected')



        
        else:
            print('No License Plates Detected')
     
        

    def recognize_stream(self, rtsp_url: str, display: bool = True, save: bool = False, out_video_path: str = None, max_width: int = 640, frame_skip: int = 2, f_scale: float = 1.2, headless: bool = False):
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print('Failed to open stream')
            return

        writer = None
        if save and out_video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 15
            ret, frame = cap.read()
            if not ret:
                print('Failed to read first frame for writer init')
                cap.release()
                return
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                w = int(w * scale)
                h = int(h * scale)
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

            # rewind by reopening to include first frame in loop uniformly
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Downscale for CPU efficiency
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Skip frames to reduce load
            if frame_skip > 0 and (frame_index % (frame_skip + 1)) != 0:
                if display and not headless:
                    cv2.imshow('ALPR', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if writer is not None:
                    writer.write(frame)
                frame_index += 1
                continue

            # Run plate detection and recognition
            bbox, _, LlpImgs, _ = self.lp_detector(frame)
            if len(bbox):
                pts = bbox[0]
                xmin = int(min(pts[0]))
                ymin = int(min(pts[1]))
                xmax = int(max(pts[0]))
                ymax = int(max(pts[1]))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), int(2/1.5*f_scale))
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                _, rec_res = self.text_system(Ilp*255.)
                if rec_res is not None:
                    text_sum = 0
                    for text, score in rec_res[::-1]:
                        text = strip_chinese(text)
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 2)
                        text_w, text_h = text_size
                        text_sum += text_h
                        frame = draw_text(frame, text,
                                          pos=(xmin, max(0, ymin-int(text_sum))),
                                          font=cv2.FONT_HERSHEY_PLAIN,
                                          font_scale=f_scale,
                                          text_color=(0, 0, 0),
                                          font_thickness=2,
                                          text_color_bg=(0, 255, 0))
            if display and not headless:
                cv2.imshow('ALPR', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if writer is not None:
                writer.write(frame)

            frame_index += 1

        cap.release()
        if writer is not None:
            writer.release()
        if display and not headless:
            cv2.destroyAllWindows()

    def recognize_stream_fast(self, rtsp_url: str, plate_model_path: str, display: bool = True, 
                             save: bool = False, out_video_path: str = None, max_width: int = 640, 
                             frame_skip: int = 2, f_scale: float = 1.2, headless: bool = False,
                             conf_threshold: float = 0.35, nms_threshold: float = 0.45,
                             downscale: float = 0.5, roi_fraction: tuple = (0.25, 0.35, 0.5, 0.22)):
        """
        Fast RTSP streaming with optimized plate detection and existing OCR system.
        
        Args:
            rtsp_url: RTSP stream URL
            plate_model_path: Path to YOLO-based plate detection ONNX model
            display: Show video window
            save: Save annotated video
            out_video_path: Output video path
            max_width: Maximum frame width for processing
            frame_skip: Process every N+1 frames
            f_scale: Font scale for text overlay
            headless: Disable display to avoid Qt/Wayland issues
            conf_threshold: Plate detection confidence threshold
            nms_threshold: NMS threshold for plate detection
            downscale: Downscale factor for detection (0.5 = half size)
            roi_fraction: ROI as (fx, fy, fw, fh) fractions of frame
        """
        # Initialize fast detector with 384x384 input size
        fast_detector = ThreadedPlateDetector(
            model_path=plate_model_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            downscale=downscale,
            frame_skip=frame_skip,
            ort_threads=1
        )
        fast_detector.start()
        
        # Open video stream
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print('Failed to open RTSP stream')
            fast_detector.stop()
            return
        
        # Setup video writer if saving
        writer = None
        if save and out_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 15
            ret, frame = cap.read()
            if not ret:
                print('Failed to read first frame for writer init')
                cap.release()
                fast_detector.stop()
                return
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                w = int(w * scale)
                h = int(h * scale)
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
        
        # ROI setup
        roi_fx, roi_fy, roi_fw, roi_fh = roi_fraction
        
        # Tracking for consecutive detections
        consec_map = {}
        ocr_results = {}
        frame_count = 0
        fps_start = time.time()
        fps_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                fps_count += 1
                
                # Downscale for processing
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / float(w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    h, w = frame.shape[:2]
                
                # Get ROI
                roi_x = int(roi_fx * w)
                roi_y = int(roi_fy * h)
                roi_w = int(roi_fw * w)
                roi_h = int(roi_fh * h)
                roi_rect = [roi_x, roi_y, roi_w, roi_h]
                
                # Process frame with fast detector
                detections = fast_detector.process_frame(frame)
                
                # Process detections
                for det in detections:
                    bbox = det['bbox']
                    conf = det['confidence']
                    x, y, w_det, h_det = bbox
                    
                    # Check if detection is in ROI
                    cx = x + w_det / 2
                    cy = y + h_det / 2
                    in_roi = (roi_x <= cx <= roi_x + roi_w and roi_y <= cy <= roi_y + roi_h)
                    
                    # Create tracking key
                    key = (int(round(cx / 10)), int(round(cy / 10)))
                    
                    if in_roi:
                        consec_map[key] = consec_map.get(key, 0) + 1
                    else:
                        if key in consec_map:
                            consec_map.pop(key, None)
                    
                    # Draw detection
                    cv2.rectangle(frame, (x, y), (x + w_det, y + h_det), (0, 180, 255), 2)
                    cv2.putText(frame, f"P:{conf:.2f}", (x, max(12, y-6)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
                    
                    # Run OCR on stable detections
                    if consec_map.get(key, 0) >= 2:  # Require 2 consecutive detections
                        # Extract plate crop with padding
                        pad_x = int(0.03 * w_det)
                        pad_y = int(0.05 * h_det)
                        sx = max(0, x - pad_x)
                        sy = max(0, y - pad_y)
                        ex = min(w, x + w_det + pad_x)
                        ey = min(h, y + h_det + pad_y)
                        
                        plate_crop = frame[sy:ey, sx:ex].copy()
                        
                        if plate_crop.size > 0:
                            # Run OCR on crop
                            try:
                                # Convert to format expected by existing OCR
                                gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                gray_3ch = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
                                
                                # Run existing OCR system
                                _, rec_res = self.text_system(gray_3ch * 255.)
                                
                                if rec_res is not None:
                                    # Process OCR results
                                    text_parts = []
                                    for text, score in rec_res:
                                        clean_text = strip_chinese(text)
                                        if clean_text:
                                            text_parts.append(clean_text)
                                    
                                    if text_parts:
                                        final_text = ''.join(text_parts)
                                        ocr_results[key] = {
                                            'text': final_text,
                                            'time': time.time(),
                                            'bbox': bbox,
                                            'conf': conf
                                        }
                                        
                                        # Draw OCR result
                                        cv2.putText(frame, final_text, (x, y + h_det + 20), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 255, 0), 2)
                                        
                                        # Reset consecutive counter
                                        consec_map[key] = 0
                                        
                            except Exception as e:
                                if frame_count % 30 == 0:  # Print error occasionally
                                    print(f"OCR error: {e}")
                
                # Draw ROI
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
                cv2.putText(frame, "ROI", (roi_x, max(roi_y-6, 0)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Clean old OCR results
                current_time = time.time()
                for k in list(ocr_results.keys()):
                    if current_time - ocr_results[k].get('time', 0) > 6.0:
                        del ocr_results[k]
                
                # Display FPS
                if time.time() - fps_start >= 2.0:
                    fps = fps_count / (time.time() - fps_start)
                    if frame_count % 30 == 0:
                        print(f"FPS: {fps:.2f}")
                    fps_start = time.time()
                    fps_count = 0
                
                # Display frame
                if display and not headless:
                    cv2.imshow('Fast ALPR', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break
                
                # Save frame
                if writer is not None:
                    writer.write(frame)
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            print("Shutting down...")
            fast_detector.stop()
            cap.release()
            if writer is not None:
                writer.release()
            if display and not headless:
                cv2.destroyAllWindows()

def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 0, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (int(x + 1.1*text_w), y + 2*text_h), text_color_bg, -1)
    im_p = Image.fromarray(img)
    draw = ImageDraw.Draw(im_p)
    font = ImageFont.truetype("fonts/simfang.ttf",int(32*font_scale/1.5))
    draw.text((x, y ),text,text_color,font=font)
    result_o = np.array(im_p)
    # cv2.putText(img, text, (x, int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)
    return result_o

def strip_chinese(string):
    en_list = re.findall(u'[^\u4E00-\u9FA5]', string)
    for c in string:
        if c not in en_list:
            string = string.replace(c, '')
    return string