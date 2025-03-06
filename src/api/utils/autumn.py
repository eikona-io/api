import os
import aiohttp
from datetime import datetime
import logfire

async def send_autumn_usage_event(
    customer_id: str,
    gpu_type: str,
    start_time: datetime,
    end_time: datetime,
    environment: str = None,
    idempotency_key: str = None
) -> bool:
    """
    Send GPU usage data to Autumn API.
    
    Args:
        customer_id: The org_id or user_id of the customer
        gpu_type: The type of GPU used (e.g., 'T4', 'A100')
        start_time: The start time of the GPU usage
        end_time: The end time of the GPU usage
        environment: Optional environment tag
        idempotency_key: Optional unique key to prevent duplicate events
        
    Returns:
        bool: True if the event was sent successfully, False otherwise
    """
    autumn_api_key = os.getenv("AUTUMN_SECRET_KEY")
    if not autumn_api_key:
        return False
    
    if (environment == "public-share"):
        # Ignoring public-share events
        return True
           
    try:
        # Calculate duration in seconds
        duration = (end_time - start_time).total_seconds()
        
        # Skip if duration is negative or zero
        if duration <= 0:
            logfire.warning(f"Invalid duration {duration} for GPU event")
            return False
        
        # Prepare Autumn API request
        autumn_headers = {
            "Authorization": f"Bearer {autumn_api_key}",
            "Content-Type": "application/json"
        }
        
        autumn_payload = {
            "customer_id": customer_id,
            "event_name": gpu_type.lower() if gpu_type else "cpu",
            "properties": {
                "value": str(duration)
            }
        }
        
        # Add environment if provided
        if environment:
            autumn_payload["properties"]["environment"] = environment
            
        # Add idempotency key if provided
        if idempotency_key:
            autumn_payload["idempotency_key"] = idempotency_key
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.useautumn.com/v1/events",
                headers=autumn_headers,
                json=autumn_payload
            ) as response:
                if response.status != 200:
                    logfire.error(f"Failed to send usage data to Autumn: {await response.text()}")
                    return False
                else:
                    logfire.info(f"Successfully sent GPU usage data to Autumn for {gpu_type}")
                    return True
                    
    except Exception as e:
        logfire.error(f"Error sending usage data to Autumn: {str(e)}")
        return False

async def get_autumn_customer(customer_id: str) -> dict:
    autumn_api_key = os.getenv("AUTUMN_SECRET_KEY")
    if not autumn_api_key:
        logfire.error("AUTUMN_SECRET_KEY not found in environment variables")
        return None

    try:
        # Prepare Autumn API request
        autumn_headers = {
            "Authorization": f"Bearer {autumn_api_key}",
            "Content-Type": "application/json",
        }

        url = f"https://api.useautumn.com/v1/customers/{customer_id}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=autumn_headers) as response:
                if response.status == 200:
                    customer_data = await response.json()
                    return customer_data
                else:
                    error_text = await response.text()
                    # Try to parse the error response
                    try:
                        import json

                        error_data = json.loads(error_text)
                        if error_data.get("code") == "customer_not_found":
                            logfire.warning(f"Customer not found in Autumn: {customer_id}")
                            return None
                    except:
                        pass
                    # Log other errors as usual
                    logfire.error(
                        f"Failed to get customer data from Autumn: {error_text}"
                    )
                    return None

    except Exception as e:
        logfire.error(f"Error retrieving customer data from Autumn: {str(e)}")
        return None
