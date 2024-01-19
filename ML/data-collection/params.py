camera_list = [
    {
        "id" : "mac-webcam",
        "name" : "Mac webcam",
        "resolution" : (1280,720),
        "notes" : ""
    },
    {
        "id" : "logitech",
        "name" : "Logitech webcam",
        "resolution" : (1280,720),
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