from automated_lung_morphometry._workers import ProcessingWorker
# tmp_path is a pytest fixture


def test_config_load():
    worker = ProcessingWorker()
    assert 'TITLE' in worker.config_data['ProcessingActionBox'].keys()


def test_true():
    assert True is True
