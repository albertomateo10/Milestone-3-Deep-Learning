import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import detection
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para evitar problemas en sistemas sin entorno gráfico
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# -----------------------------
# Código del modelo (basado en lo que has provisto)
# -----------------------------

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_inner_model(model: detection) -> any:
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def torch_load_cpu(load_path: str) -> any:
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def load_model_path(path: str, model: detection, device: torch.device, optimizer: torch.optim = None):
    load_data = torch_load_cpu(path)
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    torch.set_rng_state(load_data['rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

    if 'optimizer' in load_data and optimizer is not None:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    initial_epoch = load_data['initial_epoch']
    return model, optimizer, initial_epoch

def torchvision_model(model_name: str, pretrained: bool = False, num_classes: int = 2):
    model_dict = {
        'faster_rcnn_v1': detection.fasterrcnn_resnet50_fpn,
        'faster_rcnn_v2': detection.fasterrcnn_resnet50_fpn_v2,
        'faster_rcnn_v3': detection.fasterrcnn_mobilenet_v3_large_fpn,
        'retinanet_v1': detection.retinanet_resnet50_fpn,
        'retinanet_v2': detection.retinanet_resnet50_fpn_v2,
        'ssd_v1': detection.ssd300_vgg16,
        'ssd_v2': detection.ssdlite320_mobilenet_v3_large,
    }

    if model_name in model_dict:
        model = model_dict[model_name](weights='COCO_V1' if pretrained else None)

        if 'faster_rcnn' in model_name:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        elif 'retinanet' in model_name:
            in_features = model.head.classification_head.cls_logits.in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = detection.retinanet.RetinaNetClassificationHead(
                in_features, num_anchors, num_classes
            )
        elif 'ssd_v1' in model_name:
            in_features = [module.in_channels for module in model.head.classification_head.module_list]
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = detection.ssd.SSDClassificationHead(
                in_features, num_anchors, num_classes
            )
        elif 'ssd_v2' in model_name:
            in_features = [module[0][0].in_channels for module in model.head.classification_head.module_list]
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = detection.ssd.SSDClassificationHead(
                in_features, num_anchors, num_classes
            )

    else:
        raise ValueError('Modelo no encontrado en la lista.')

    return model

def get_model(model_name: str, model_path: str = '', num_classes: int = 2,
              lr_data: list = None, pretrained: bool = False,
              use_gpu: bool = False):
    device_name = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    model = torchvision_model(model_name, pretrained, num_classes).to(device)
    if use_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    if lr_data:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr_data[0], momentum=lr_data[1], weight_decay=lr_data[2])
    else:
        optimizer = None

    if os.path.isfile(model_path):
        model, optimizer, initial_epoch = load_model_path(model_path, model, device, optimizer)
    else:
        initial_epoch = 0
        print('Pesos no encontrados')

    return model, optimizer, initial_epoch, device

# Parámetros del modelo
model_name = 'retinanet_v2'
model_path = 'model/model_Adadelta_1280_0005.pt' # Ajusta la ruta a donde tengas el modelo
num_classes = 2
pretrained = False
use_gpu = True
CLASSES = ["Person", "None"]
COLORS = {"Person": 'blue', "None": 'green'}

model, _, _, device = get_model(model_name, model_path, num_classes, None, pretrained, use_gpu)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict_objects(image_path, output_dir='static/detections'):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    predictions = outputs[0]
    labels = predictions['labels']
    scores = predictions['scores']
    boxes = predictions['boxes']

    # Filtrar por confianza > 0.21
    high_confidence_indices = scores > 0.21
    labels = labels[high_confidence_indices]
    scores = scores[high_confidence_indices]
    boxes = boxes[high_confidence_indices]

    image_np = np.array(image)
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    detected_classes = set()
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.cpu().numpy()
        class_name = CLASSES[label.item()]
        detected_classes.add(class_name)
        color = COLORS[class_name]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    handles = [patches.Patch(color=COLORS[class_name], label=class_name) for class_name in detected_classes]
    if handles:
        ax.legend(handles=handles, loc='upper right')

    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, os.path.splitext(base_filename)[0] + '_detection.jpg')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close(fig)
    return output_image_path

# -----------------------------
# Código Flask
# -----------------------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verificar si se ha enviado un fichero
        if 'file' not in request.files:
            return "No file found", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        # Realizar la predicción
        output_path = predict_objects(image_path)
        return render_template('result.html', original_image=url_for('static', filename='uploads/' + filename), 
                               detected_image=url_for('static', filename='detections/' + os.path.basename(output_path)))
    return render_template('index.html')

# Ruta para servir las imágenes estáticas
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
