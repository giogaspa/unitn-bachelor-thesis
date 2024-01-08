image_width = 640
image_height = 640
image_ar = image_height / image_width

movenet_image_height = 256
movenet_image_width = 256
movenet_image_landscape_height_offset = int((movenet_image_height - (image_ar * movenet_image_width)) / 2)

camera_list = [
    {
        "id" : "mac-webcam",
        "name" : "Mac webcam",
        "resolution" : (640,480),
        "notes" : ""
    },
    {
        "id" : "logitech",
        "name" : "Logitech webcam",
        "resolution" : (640,480),
        "notes" : "720p"
    }
]

# Posizione della camera rispetto al corpo
camera_position = [
    "frontal",
    "lateral-left",
    "lateral-right",
]

subject = [
    "gioacchino",
    "veronica"
]

subject_position = [
    "sitting",
    "upright",
    "lying"
]

acquisition_info = {
    "exercise_id": None,
    "exercise" : None,
    "camera_id": None,
    "camera_position": None,
    "subject": None,
    "subject_position" : None,
    "model": "movenet_thunder"
}