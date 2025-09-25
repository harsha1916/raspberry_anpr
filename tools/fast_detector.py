"""
Fast license plate detector using YOLO-based ONNX model
Optimized for Raspberry Pi with threading and downscaling
"""
import time
import threading
import queue
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path


class FastPlateDetector:
    def __init__(self, model_path, conf_threshold=0.35, nms_threshold=0.45, 
                 downscale=0.5, frame_skip=2, ort_threads=1):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.downscale = downscale
        self.frame_skip = frame_skip
        self.ort_threads = ort_threads
        
        # Initialize ONNX session
        self.session = self._make_ort_session()
        self.frame_count = 0
        
        # Print model input info for debugging
        input_info = self.session.get_inputs()[0]
        print(f"[FastDetector] Model input: {input_info.name}, shape: {input_info.shape}")
        target_h, target_w = self._get_model_input_size()
        print(f"[FastDetector] Expected input size: {target_h}x{target_w}")
        
    def _make_ort_session(self):
        """Create optimized ONNX session for CPU inference"""
        opts = ort.SessionOptions()
        try:
            opts.intra_op_num_threads = int(self.ort_threads)
            opts.inter_op_num_threads = 1
        except Exception:
            pass
        opts.log_severity_level = 3
        sess = ort.InferenceSession(str(self.model_path), sess_options=opts, 
                                   providers=["CPUExecutionProvider"])
        return sess
    
    def _get_model_input_size(self):
        """Get the expected input size from the model"""
        input_shape = self.session.get_inputs()[0].shape
        # Assume NCHW format, get H and W
        if len(input_shape) >= 4:
            return input_shape[2], input_shape[3]  # H, W
        elif len(input_shape) >= 3:
            return input_shape[1], input_shape[2]  # H, W
        else:
            return 384, 384  # Default fallback
    
    def _preprocess_image(self, img, target_size=None):
        """Preprocess image for YOLO inference"""
        if target_size is None:
            target_h, target_w = self._get_model_input_size()
            target_size = target_h  # Assume square input
        else:
            target_h = target_w = target_size
            
        h, w = img.shape[:2]
        
        # Resize with aspect ratio preservation
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterbox
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, ...]
        
        return input_tensor, scale, x_offset, y_offset
    
    def _parse_detections(self, outputs, original_shape, scale, x_offset, y_offset):
        """Parse YOLO outputs and convert to image coordinates"""
        boxes = []
        confidences = []
        
        for output in outputs:
            if output.ndim == 3:
                output = output[0]  # Remove batch dimension
            
            # YOLO format: [x_center, y_center, width, height, confidence, class_scores...]
            for detection in output:
                if len(detection) < 5:
                    continue
                    
                x_center, y_center, width, height, conf = detection[:5]
                
                if conf < self.conf_threshold:
                    continue
                
                # Convert from letterbox coordinates to original image coordinates
                x_center = (x_center - x_offset) / scale
                y_center = (y_center - y_offset) / scale
                width = width / scale
                height = height / scale
                
                # Convert to top-left corner format
                x = int(x_center - width / 2)
                y = int(y_center - height / 2)
                w = int(width)
                h = int(height)
                
                # Clamp to image bounds
                x = max(0, min(x, original_shape[1] - 1))
                y = max(0, min(y, original_shape[0] - 1))
                w = max(1, min(w, original_shape[1] - x))
                h = max(1, min(h, original_shape[0] - y))
                
                boxes.append([x, y, w, h])
                confidences.append(float(conf))
        
        return boxes, confidences
    
    def _nms_boxes(self, boxes, confidences):
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not boxes:
            return []
        
        # Convert to OpenCV format
        rects = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
        
        try:
            indices = cv2.dnn.NMSBoxes(rects, confidences, self.conf_threshold, self.nms_threshold)
            if len(indices) == 0:
                return []
            
            # Handle different return types from different OpenCV versions
            if hasattr(indices, 'flatten'):
                return [int(i) for i in indices.flatten()]
            else:
                return [int(i[0]) for i in indices]
        except Exception as e:
            print(f"NMS error: {e}")
            return list(range(len(boxes)))
    
    def detect(self, frame):
        """Detect license plates in a single frame"""
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1)) != 0:
            return []
        
        # Downscale for faster processing
        if self.downscale < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.downscale)
            new_h = int(h * self.downscale)
            small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            small_frame = frame
        
        try:
            # Preprocess (auto-detect model input size)
            input_tensor, scale, x_offset, y_offset = self._preprocess_image(small_frame)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # Parse detections
            boxes, confidences = self._parse_detections(outputs, small_frame.shape, scale, x_offset, y_offset)
            
            # Apply NMS
            keep_indices = self._nms_boxes(boxes, confidences)
            
            # Scale back to original frame coordinates
            final_boxes = []
            for idx in keep_indices:
                box = boxes[idx]
                conf = confidences[idx]
                
                # Scale back to original frame size
                if self.downscale < 1.0:
                    scale_back = 1.0 / self.downscale
                    x = int(box[0] * scale_back)
                    y = int(box[1] * scale_back)
                    w = int(box[2] * scale_back)
                    h = int(box[3] * scale_back)
                else:
                    x, y, w, h = box
                
                # Clamp to original frame bounds
                x = max(0, min(x, frame.shape[1] - 1))
                y = max(0, min(y, frame.shape[0] - 1))
                w = max(1, min(w, frame.shape[1] - x))
                h = max(1, min(h, frame.shape[0] - y))
                
                final_boxes.append({
                    'bbox': [x, y, w, h],
                    'confidence': conf
                })
            
            return final_boxes
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []


class ThreadedPlateDetector:
    """Threaded plate detector for real-time processing"""
    
    def __init__(self, model_path, conf_threshold=0.35, nms_threshold=0.45,
                 downscale=0.5, frame_skip=2, ort_threads=1):
        self.detector = FastPlateDetector(model_path, conf_threshold, nms_threshold,
                                        downscale, frame_skip, ort_threads)
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the detection thread"""
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the detection thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is not None:
                    detections = self.detector.detect(frame)
                    
                    # Put result, drop old if queue is full
                    try:
                        self.result_queue.put_nowait(detections)
                    except queue.Full:
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(detections)
                        except queue.Empty:
                            pass
                            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection thread error: {e}")
                time.sleep(0.01)
    
    def process_frame(self, frame):
        """Submit frame for detection and get latest results"""
        # Submit frame (drop old if queue is full)
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass
        
        # Get latest results
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return []
    
    def get_latest_detections(self):
        """Get the most recent detection results"""
        detections = []
        while not self.result_queue.empty():
            try:
                detections = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return detections
