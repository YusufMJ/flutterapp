import cv2
from django.http import HttpResponse, JsonResponse
from django.middleware.csrf import get_token
from django.conf import settings
import os

def upload_video(request):
   if request.method == 'POST':
        if request.FILES.get('video'):
            video_file = request.FILES['video']
            # Define the path where you want to save the video
            # Make sure the 'media' directory exists in your Django project
            media_root = os.path.join(settings.MEDIA_ROOT, 'videos')
            if not os.path.exists(media_root):
                os.makedirs(media_root)
            file_path = os.path.join(media_root, video_file.name)
            # Save the uploaded video file
            with open(file_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            # Apply filter to the uploaded video
            filtered_video_path = apply_greyscale_filter_to_video(file_path)
            if filtered_video_path:
                return JsonResponse({'message': 'Video uploaded and filtered successfully', 'filtered_video_path': filtered_video_path})
            else:
                return JsonResponse({'error': 'Failed to apply filter to video'}, status=500)
        else:
            return JsonResponse({'error': 'No video file provided'}, status=400)
   else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_csrf_token(request):
    token = get_token(request)
    return JsonResponse({'csrf_token': token})


def apply_greyscale_filter_to_video(video_path):
    # Check if video file exists
    if not os.path.exists(video_path):
        return JsonResponse({'error': 'Video file not found'}, status=404)
    
    # Define output file path
    output_file_path = os.path.splitext(video_path)[0] + '_greyscaled.mp4'
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JsonResponse({'error': 'Failed to open video file'}, status=500)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (codec may vary based on the system)
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height), isColor=False)  # Set isColor=False for greyscale
    
    # Apply greyscale filter to each frame and write to output video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to greyscale
        out.write(grey_frame)  # Write greyscaled frame to output video
    
    # Release video capture and writer objects
    cap.release()
    out.release()
    
    return output_file_path  # Return the path to the greyscaled video file


def get_processed_video(request, video_file_name):
    video_path = os.path.join('video_upload_server/media/videos', video_file_name)
    print(video_path)
    if os.path.exists(video_path):
        with open(video_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
            response['Content-Disposition'] = f'attachment; filename="{video_file_name}"'
            return response
    else:
        return JsonResponse({'error': 'Processed video file not found'}, status=404)