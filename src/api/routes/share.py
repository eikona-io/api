import os
import dub
import logging
from typing import Optional

# Replace the module-level initialization with a singleton pattern
class DubClient:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            dub_api_key = os.getenv("DUB_API_KEY")
            if not dub_api_key:
                logging.warning("DUB_API_KEY environment variable is not set")
                return None
            cls._instance = dub.Dub(token=dub_api_key)
        return cls._instance

def _check_dub_client() -> bool:
    client = DubClient.get_instance()
    if not client:
        logging.error("Dub client not initialized - missing API key")
        return False
    return True


async def create_dub_link(url: str, slug: str) -> Optional[str]:
    if not _check_dub_client():
        return None

    res = None
    client = DubClient.get_instance()

    try:
        res = client.links.create(
            request={
                "url": url,
                "domain": "comfydeploy.link",
                "doIndex": True,
                "tagIds": "tag_Oxo856QUGcEhziqjHZ3PO0Hv",
                "external_id": slug,
                "proxy": True,
                "title": f"Comfy Deploy Share - {slug}",
                # "rewrite": True,
            }
        )
    except Exception as e:
        logging.error(f"Error creating dub link: {str(e)}")
        return None

    if res is not None:
        logging.info(f"link created: {res.short_link}")
        return res.short_link
    return None


async def get_dub_link(slug: str) -> Optional[str]:
    if not _check_dub_client():
        return None

    try:
        res = DubClient.get_instance().links.get(request={"external_id": f"ext_{slug}"})
        if res is not None:
            logging.info(f"link found: {res.short_link}")
            return res
    except Exception as e:
        logging.info(f"Link not found for slug: {slug}")
        return None

    return None


async def update_dub_link(link_id: str, url: str, slug: str) -> Optional[str]:
    if not _check_dub_client():
        return None

    res = None
    client = DubClient.get_instance()

    try:
        res = client.links.update(
            link_id=link_id,
            request_body={
                "url": url,
                "domain": "comfydeploy.link",
                "doIndex": True,
                "tagIds": "tag_Oxo856QUGcEhziqjHZ3PO0Hv",
                "external_id": slug,
                "proxy": True,
                "title": f"Comfy Deploy Share - {slug}",
                # "rewrite": True,
            },
        )
    except Exception as e:
        logging.error(f"Error updating dub link: {str(e)}")
        return None

    if res is not None:
        logging.info(f"link updated: {res.short_link}")
        return res
    else:
        return None
