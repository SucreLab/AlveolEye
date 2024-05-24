from alveoleye._workers import ProcessingWorker
# tmp_path is a pytest fixture

def process_image_completely():
    from alveoleye.lungcv.model_operations import init_trained_model
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    worker = ProcessingWorker()
    model = init_trained_model("src/alveoleye/data/default.pth")

def load_proxy_viewer(make_napari_viewer_proxy):
    import napari
    from alveoleye._widget import WidgetMain
    viewer: napari.Viewer = make_napari_viewer_proxy()
    viewer.window.add_plugin_dock_widget("AlveolEye")
    worker = ProcessingWorker()
    worker.set_napari_viewer(viewer)

    widget = WidgetMain(viewer)
    widget.init_ui()

    widget.processing_group_box.import_paths["image"] = "alveoleye/data/2.png"
    widget.processing_group_box.on_import_image_press()

    assert len(viewer.layers) > 0