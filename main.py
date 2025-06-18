from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import asyncio
import onnxruntime as ort # Importation d'ONNX Runtime

app = FastAPI()

latest_frame = None

# --- Définir les noms de classes pour YOLOv8 (COCO) ---
# Vous devrez les inclure car ONNX Runtime ne les fournit pas
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- Charger le modèle ONNX ---
# Si vous copiez yolov8n.onnx dans l'image Docker
# Ou le chemin où il sera téléchargé/accessible
# Assurez-vous que 'yolov8n.onnx' est bien dans le répertoire de l'application
# Ou téléchargez-le si vous ne l'incluez pas dans l'image (voir note ci-dessous)
ONNX_MODEL_PATH = "yolov8n.onnx"

try:
    # Créer une session d'inférence ONNX Runtime
    # Utilisez providers=["CPUExecutionProvider"] pour le déploiement CPU
    # Si vous avez un GPU, vous pouvez essayer "CUDAExecutionProvider" mais c'est lourd
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f"✅ Modèle ONNX chargé : {ONNX_MODEL_PATH}")

    # Obtenir les noms des entrées et sorties du modèle
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # La taille d'entrée que le modèle ONNX attend (par ex. 416x416)
    input_height, input_width = session.get_inputs()[0].shape[2:] # Normalement (1, 3, H, W)
    print(f"Modèle ONNX - Taille d'entrée attendue: {input_width}x{input_height}")

except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle ONNX : {e}")
    session = None # Marquer le modèle comme non chargé

# Fonctions de pré-traitement et post-traitement pour ONNX YOLOv8
def preprocess(image):
    # Redimensionner l'image à la taille d'entrée du modèle (par ex. 416x416)
    # Puis padding si nécessaire pour maintenir le ratio d'aspect
    # Pour YOLOv8, le redimensionnement simple est courant
    resized = cv2.resize(image, (input_width, input_height))
    # Convertir BGR en RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normaliser [0, 255] à [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    # Changer l'ordre des axes de (H, W, C) à (C, H, W) et ajouter une dimension de batch
    input_tensor = np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)
    return input_tensor

def postprocess(output, original_shape, conf_threshold=0.25, iou_threshold=0.45):
    # Les sorties de YOLOv8 ONNX sont un peu différentes d'ultralytics
    # La sortie est généralement de forme (1, num_classes + 4 + num_segments, num_boxes)
    # ou (1, num_boxes, num_classes + 4 + num_segments)
    # Nous devons la transposer pour avoir (num_boxes, num_classes + 4 + num_segments)
    output = output[0].T # Transposer pour avoir les boîtes en lignes

    # Filtrer les détections par seuil de confiance
    scores = output[:, 4:] # Les scores des classes commencent à l'indice 4
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Appliquer le seuil de confiance
    mask = confidences > conf_threshold
    boxes = output[mask, :4]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convertir les coordonnées des boîtes
    # YOLOv8 renvoie (center_x, center_y, width, height)
    # Il faut les convertir en (x1, y1, x2, y2)
    # Et les redimensionner à la forme de l'image originale
    h, w = original_shape[:2]
    scale_x = w / input_width
    scale_y = h / input_height

    x_center = boxes[:, 0]
    y_center = boxes[:, 1]
    box_width = boxes[:, 2]
    box_height = boxes[:, 3]

    x1 = (x_center - box_width / 2) * scale_x
    y1 = (y_center - box_height / 2) * scale_y
    x2 = (x_center + box_width / 2) * scale_x
    y2 = (y_center + box_height / 2) * scale_y

    detections = np.column_stack((x1, y1, x2, y2, confidences, class_ids)).astype(np.float32)

    # Appliquer la Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(detections[:, :4], detections[:, 4], conf_threshold, iou_threshold)

    # Vérifier si indices n'est pas vide avant de l'utiliser
    if len(indices) > 0:
        final_detections = detections[indices.flatten()]
    else:
        final_detections = np.array([]) # Retourner un tableau vide si aucune détection

    return final_detections


@app.get("/")
async def root():
    return {"message": "Host OK"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_frame
    await websocket.accept()
    print("✅ WebSocket accepté !")
    while True:
        try:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            latest_frame = frame
        except Exception as e:
            print(f"❌ Erreur WebSocket : {e}")
            break

@app.get("/video_feed")
async def video_feed():
    boundary = "frame"

    async def generate():
        global latest_frame
        if session is None: # Si le modèle n'a pas pu être chargé
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nError: Model not loaded\r\n"
            return

        while True:
            await asyncio.sleep(0.05) # Contrôle le framerate de sortie
            if latest_frame is None:
                continue

            try:
                frame_to_process = latest_frame.copy()
                original_h, original_w = frame_to_process.shape[:2]

                # Pré-traitement de l'image
                input_tensor = preprocess(frame_to_process)

                # Inférence ONNX Runtime
                outputs = session.run([output_name], {input_name: input_tensor})
                output_data = outputs[0]

                # Post-traitement pour obtenir les boîtes et les scores
                detections = postprocess(output_data, frame_to_process.shape)

                # Dessiner les détections sur l'image
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = map(int, det[:6])
                    class_name = CLASS_NAMES[int(cls_id)]
                    label = f"{class_name} {conf:.2f}"

                    cv2.rectangle(frame_to_process, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_to_process, label, (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Encoder la trame traitée en JPEG
                ret, jpeg = cv2.imencode(".jpg", frame_to_process, [int(cv2.IMWRITE_JPEG_QUALITY), 70]) # Qualité 70%
                if not ret:
                    continue

                frame_bytes = jpeg.tobytes()
                yield (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame_bytes)}\r\n\r\n"
                ).encode() + frame_bytes + b"\r\n"

            except Exception as e:
                print(f"❌ Erreur stream ou détection ONNX: {e}")
                # En cas d'erreur, peut-être envoyer une image vide ou un message d'erreur
                # Pour éviter de bloquer le générateur
                await asyncio.sleep(1) # Attendre un peu avant de réessayer
                continue

    headers = {"Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"}
    return StreamingResponse(generate(), headers=headers)
