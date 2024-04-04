# face_editor/views.py
import cv2
from django.http import JsonResponse
import numpy as np

from django.shortcuts import render, redirect
from django.core.mail import send_mail

from .forms import UserRegistrationForm, UserProfileForm
from .models import UserPhoto, UserProfile
from skimage.metrics import structural_similarity


def scan_and_send_email(request):
    if request.method == 'POST':
        # Get user email from the form
        recipient_email = request.POST.get('email')

        # Initialize the camera
        cap = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not cap.isOpened():
            return render(request, 'error.html', {'message': 'Failed to open camera.'})

        while True:
            # Capture frame from camera
            ret, frame = cap.read()

            # Display the frame in a window
            cv2.imshow('Camera Preview', frame)

            # Check for key press (wait for 'q' key to capture image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera
        cap.release()

        # Perform face detection (assuming you have haarcascade_frontalface_default.xml)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no face detected, return error message
        if len(faces) == 0:
            return render(request, 'error.html', {'message': 'No face detected. Please try again.'})

        # Retrieve user profile photos from the database
        user_profiles = UserProfile.objects.all()

        # Compare captured face with faces from user profiles
        for profile in user_profiles:
            img_data = profile.face_image.read()
            nparr = np.frombuffer(img_data, np.uint8)
            db_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Perform face comparison
                db_face = cv2.resize(db_image, (w, h))
                mse = np.mean((roi_gray - cv2.cvtColor(db_face, cv2.COLOR_BGR2GRAY)) ** 2)

                # If mean squared error is below a certain threshold, consider it a match
                if mse < 1000:
                    # Send the modified image via email
                    subject = 'Scanned Face'
                    message = 'Here is your scanned face image.'
                    sender_email = 'your@email.com'  # Change this to your sender email
                    send_mail(subject, message, sender_email, [recipient_email], fail_silently=False)
                    return render(request, 'success.html', {'email': recipient_email})

        # If no match found, return error message
        return render(request, 'error.html', {'message': 'Face not recognized. Please try again.'})

    return render(request, 'scan.html')



def register_user(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        profile_form = UserProfileForm(request.POST, request.FILES)
        if user_form.is_valid() and profile_form.is_valid():
            user = user_form.save()
            profile = profile_form.save(commit=False)
            profile.user = user
            profile.save()

            # Save uploaded photo to UserPhoto model
            if 'photo' in request.FILES:
                photo = request.FILES['photo']
                user_photo = UserPhoto.objects.create(user=user, photo=photo)
                user_photo.save()

            return redirect('registration_success')

    else:
        user_form = UserRegistrationForm()
        profile_form = UserProfileForm()
    return render(request, 'register.html', {'user_form': user_form, 'profile_form': profile_form})


def compare_photos(request):
    if request.method == 'POST':
        uploaded_photo = request.FILES['photo']
        # Read the uploaded photo
        uploaded_image = cv2.imdecode(np.fromstring(uploaded_photo.read(), np.uint8), cv2.IMREAD_COLOR)

        # Retrieve user profile photos from the database
        user_profiles = UserProfile.objects.all()

        # Loop through user profiles and compare photos
        for profile in user_profiles:
            profile_photo = cv2.imdecode(np.fromstring(profile.face_image.read(), np.uint8), cv2.IMREAD_COLOR)
            # Perform image comparison (e.g., using histogram comparison)
            similarity_score = compare_images(uploaded_image, profile_photo)
            # Check if similarity score exceeds threshold
            if similarity_score > cv2.threshold:
                # Match found, do something (e.g., return profile details)
                return render(request, 'success.html', {'profile': profile})
        
        # No match found
        return render(request, 'no_match_found.html')

    # Handle GET request (e.g., render upload form)
    return render(request, 'upload_form.html')

def compare_images(image1, image2):
# Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute Structural Similarity Index (SSIM)
    similarity = structural_similarity(gray1, gray2)
    return similarity

def capture_photo(request):
    if request.method == 'POST':
        email = request.POST.get('email')

        # Initialize the camera
        cap = cv2.VideoCapture(0)

        # Check if the camera is opened successfully
        if not cap.isOpened():
            return JsonResponse({'message': 'Failed to open camera.'}, status=500)

        # Capture image from camera
        ret, frame = cap.read()

        # Release the camera
        cap.release()

        # Perform face detection (assuming you have haarcascade_frontalface_default.xml)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no face detected, return error message
        if len(faces) == 0:
            return JsonResponse({'message': 'No face detected. Please try again.'}, status=400)

        # Retrieve user profile with the submitted email
        try:
            user_profile = UserProfile.objects.get(user__email=email)
        except UserProfile.DoesNotExist:
            return JsonResponse({'message': 'User profile not found for the submitted email.'}, status=400)

        # Retrieve the user profile photo from the database
        img_data = user_profile.face_image.read()
        nparr = np.frombuffer(img_data, np.uint8)
        db_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Compare captured face with the user profile photo
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Perform face comparison
            db_face = cv2.resize(db_image, (w, h))
            mse = np.mean((roi_gray - cv2.cvtColor(db_face, cv2.COLOR_BGR2GRAY)) ** 2)

            # If mean squared error is below a certain threshold, consider it a match
            if mse < 1000:
                # Return success response
                return JsonResponse({'message': 'Face matched successfully.'})

        # If no match found, return error message
        return JsonResponse({'message': 'Face not recognized. Please try again.'}, status=400)

    return JsonResponse({'message': 'Invalid request method.'}, status=405)




def registration_success(request):
    return render(request, 'rsuccess.html')
