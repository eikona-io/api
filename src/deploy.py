import modal
import modal.runner
from modal_apps.modal_downloader import modal_downloader_app

print(modal.__version__)

if __name__ == "__main__":
    result = modal.runner.deploy_app(modal_downloader_app)
    print(result)