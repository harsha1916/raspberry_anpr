from alpr import ALPR

x = ALPR(out_dir='lpd_results')
# x.detect_lp(path ='Test_Images/Cars438.jpeg',Bbox=False,show=True,save=False)
# x.blur_lp(path ='Test_Images/Cars422.png',show=True,save=False)
x.recognize_lp(path ='Test_Images/Cars450.jpeg',show=True,save=False,f_scale=1.5)

# Example for RTSP stream (uncomment and set your RTSP URL)
# rtsp_url = 'rtsp://username:password@IP:554/stream'
# For headless operation on Raspberry Pi (no display), use:
# x.recognize_stream(rtsp_url=rtsp_url, display=False, headless=True, save=True, out_video_path='output.mp4', max_width=480, frame_skip=3, f_scale=1.0)
# For display operation:
# x.recognize_stream(rtsp_url=rtsp_url, display=True, headless=False, save=False, out_video_path=None, max_width=640, frame_skip=2, f_scale=1.0)