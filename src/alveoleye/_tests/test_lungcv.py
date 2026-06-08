from alveoleye._workers import ProcessingWorker
# tmp_path is a pytest fixture


def test_config_load():
    worker = ProcessingWorker()
    assert 'TITLE' in worker.config_data['ProcessingActionBox'].keys()


def test_load_model():
    from alveoleye.lungcv.model_operations import init_trained_model
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    worker = ProcessingWorker()
    model = init_trained_model("src/alveoleye/data/default.pth")
    assert type(model) is MaskRCNN


def load_proxy_viewer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    assert type(viewer) is not None