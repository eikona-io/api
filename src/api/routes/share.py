import os
import dub
import logging
from typing import Optional

dub_api_key = os.getenv("DUB_API_KEY")
d = None

if not dub_api_key:
    logging.error("DUB_API_KEY environment variable is not set")
else:
    d = dub.Dub(token=dub_api_key)


def _check_dub_client() -> bool:
    if not d:
        logging.error("Dub client not initialized - missing API key")
        return False
    return True


async def create_dub_link(url: str, slug: str) -> Optional[str]:
    if not _check_dub_client():
        return None

    res = None

    try:
        res = d.links.create(
            request={
                "url": url,
                "domain": "comfydeploy.link",
                "doIndex": True,
                "tagIds": "tag_Oxo856QUGcEhziqjHZ3PO0Hv",
                "external_id": slug,
                "proxy": True,
                "title": f"Comfy Deploy Share - {slug}",
                "rewrite": True,
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
        res = d.links.get(request={"external_id": f"ext_{slug}"})
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

    try:
        res = d.links.update(
            link_id=link_id,
            request_body={
                "url": url,
                "domain": "comfydeploy.link",
                "doIndex": True,
                "tagIds": "tag_Oxo856QUGcEhziqjHZ3PO0Hv",
                "external_id": slug,
                "proxy": True,
                "title": f"Comfy Deploy Share - {slug}",
                "rewrite": True,
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
