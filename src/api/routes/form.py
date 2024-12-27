from api.routes.platform import get_clerk_user
from api.routes.utils import select
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, Optional
from datetime import datetime

# from discordwebhook import Discord  # You'll need to install this package
# from discord_webhook import AsyncDiscordWebhook, DiscordEmbed
from pydantic import BaseModel
import os
from api.database import get_db  # You'll need to create this
# from api.services.email import send_email_form_submission  # You'll need to create this

import resend
import os
from typing import Union, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update
from api.models import FormSubmission
import discord
from discord.ext import commands

# Add this new function to handle Discord operations
async def send_discord_message(channel_id: int, embed: discord.Embed):
    """Handle Discord message sending with proper client lifecycle management"""
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    if not discord_token:
        raise ValueError("Discord configuration missing")
    
    intents = discord.Intents.default()
    async with discord.Client(intents=intents) as client:
        await client.login(discord_token)
        channel = await client.fetch_channel(channel_id)
        return await channel.send(embed=embed)

async def send_email_form_submission(to: Union[str, List[str]]):
    """Send form submission email using Resend"""

    try:
        # Filter out None values if to is a list
        if isinstance(to, list):
            to = [email for email in to if email is not None]
        
        # Return early if no valid emails
        if not to or (isinstance(to, list) and len(to) == 0):
            return None

        params: resend.Emails.SendParams = {
            "from": "Nick & Benny <founders@comfydeploy.com>",
            "to": to,
            "subject": "Question...",
            "html": """<div>Hey quick question: what do you want to get out of Comfy Deploy?
<br>
<br>
Thanks!
<br>
Nick
</div>""",
        }
        # TODO can we get this to run async
        email: resend.Email = resend.Emails.send(params)
        return email

    except Exception as e:
        raise Exception(f"Failed to send email: {str(e)}")


router = APIRouter(tags=["Form"])


class OnboardingForm(BaseModel):
    inputs: Dict[str, Any]


class FormCheckResponse(BaseModel):
    success: bool
    has_submitted: bool
    submission: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/form/onboarding")
async def submit_onboarding_form(
    request: Request, form_data: OnboardingForm, db: AsyncSession = Depends(get_db)
):
    current_user = request.state.current_user
    org_id = request.state.current_user.get("org_id")
    user_id = request.state.current_user.get("user_id")

    try:
        # Check existing submission
        query = select(FormSubmission).where(
            FormSubmission.user_id == user_id, FormSubmission.org_id == org_id
        )
        result = await db.execute(query)
        existing_submission = result.scalar_one_or_none()

        thread_id = None

        if not existing_submission:
            # Create new submission
            new_submission = FormSubmission(
                user_id=user_id, org_id=org_id, inputs=form_data.inputs
            )
            db.add(new_submission)
            await db.commit()
            await db.refresh(new_submission)

            # Format Discord message
            formatted_message = "\n".join(
                f"**{key}:** {value}" for key, value in form_data.inputs.items()
            )

            # Get Discord configuration
            discord_token = os.getenv("DISCORD_BOT_TOKEN")  # This should be your bot token
            channel_id = int(os.getenv("DISCORD_CHANNEL_ID"))  # Channel ID needs to be an integer

            if not discord_token or not channel_id:
                raise HTTPException(
                    status_code=500, detail="Discord configuration missing"
                )
                
            clerk_user = await get_clerk_user(user_id)

            # Create embed
            embed = discord.Embed(
                title=f"Form Submission - {clerk_user.get('first_name')} {clerk_user.get('last_name')}",
                description=formatted_message,
                color=5814783
            )
            embed.set_footer(
                text=f"User ID: {user_id}{f' | Org ID: {org_id}' if org_id else ''}"
            )

            # Send message using the new function
            message = await send_discord_message(channel_id, embed)
            thread_id = message.id

            if thread_id:
                new_submission.discord_thread_id = str(thread_id)
                await db.commit()

        else:
            # Update existing submission
            stmt = (
                update(FormSubmission)
                .where(FormSubmission.id == existing_submission.id)
                .values(inputs=form_data.inputs, updated_at=datetime.utcnow())
            )
            await db.execute(stmt)
            await db.commit()

        # Get final submission state
        query = select(FormSubmission).where(
            FormSubmission.user_id == user_id, FormSubmission.org_id == org_id
        )
        result = await db.execute(query)
        submission = result.scalar_one_or_none()

        # Send email
        try:
            clerk_user = await get_clerk_user(user_id)
            # print(clerk_user)
            # print(clerk_user["email_addresses"][0]["email_address"])
            # print(form_data.inputs.get("workEmail"))
            await send_email_form_submission(
                [clerk_user["email_addresses"][0]["email_address"], form_data.inputs.get("workEmail")]
            )
        except Exception as e:
            print(f"Error sending email: {e}")

        return submission.to_dict()

    except Exception as e:
        print(f"Error submitting form: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit form")


@router.get("/form/onboarding")
async def check_form_submission(request: Request, db: AsyncSession = Depends(get_db)):
    # current_user = request.state.current_user
    # org_id = request.state.current_user.get("org_id")
    # user_id = request.state.current_user.get("user_id")

    try:
        query = select(FormSubmission).apply_org_check(request=request)
        result = await db.execute(query)
        submission = result.scalar_one_or_none()

        return submission.to_dict() if submission else None

    except Exception as e:
        print(f"Error checking form submission: {e}")
        return {
            "error": "Failed to check form submission",
        }


class FormUpdateRequest(BaseModel):
    call_booked: bool


@router.patch("/form/onboarding")
async def set_call_booked(
    request: Request, form_update: FormUpdateRequest, db: AsyncSession = Depends(get_db)
):
    current_user = request.state.current_user
    org_id = request.state.current_user.get("org_id")
    user_id = request.state.current_user.get("user_id")

    clerk_user = await get_clerk_user(user_id)

    try:
        stmt = (
            update(FormSubmission)
            .where(FormSubmission.user_id == user_id, FormSubmission.org_id == org_id)
            .values(call_booked=form_update.call_booked)
        )
        result = await db.execute(stmt)
        await db.commit()

        if result.rowcount == 0:
            return {"success": False, "error": "No matching submission found"}

        # Get updated submission for thread_id
        query = select(FormSubmission).where(
            FormSubmission.user_id == user_id, FormSubmission.org_id == org_id
        )
        result = await db.execute(query)
        submission = result.scalar_one_or_none()

        thread_id = submission.discord_thread_id if submission else None
        if thread_id:
            # Get Discord configuration
            discord_token = os.getenv("DISCORD_BOT_TOKEN")
            channel_id = int(os.getenv("DISCORD_CHANNEL_ID"))

            if not discord_token or not channel_id:
                print("Discord configuration missing")
                return {"success": True, "call_booked": form_update.call_booked}

            try:
                embed = discord.Embed(
                    title="Call Booked",
                    description="The user has booked a call.",
                    color=5763719
                )

                intents = discord.Intents.default()
                async with discord.Client(intents=intents) as client:
                    await client.login(discord_token)
                    channel = await client.fetch_channel(channel_id)
                    message = await channel.fetch_message(int(thread_id))
                    
                    thread = message.thread
                    if not thread:
                        thread = await message.create_thread(
                            name="Call Booked Discussion",
                            auto_archive_duration=10080
                        )
                    
                    await thread.send(embed=embed)

            except Exception as e:
                print(f"Error sending Discord message: {e}")

        return submission.to_dict()

    except Exception as e:
        print(f"Error updating call_booked status: {e}")
        return {"success": False, "error": "Failed to update call_booked status"}
