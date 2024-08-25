import os
import clickhouse_connect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
load_dotenv()

# Use environment variables for database connection
DATABASE_URL = os.getenv("DATABASE_URL")

# Ensure the URL uses the asyncpg dialect
if DATABASE_URL and not DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

def init_db():
    # You can add any initialization logic here if needed
    pass

clickhouse_client = None

async def get_clickhouse_client():
    global clickhouse_client
    if clickhouse_client is None:
        clickhouse_client = await clickhouse_connect.get_async_client(
            host=os.getenv("CLICKHOUSE_HOST"),
            user=os.getenv("CLICKHOUSE_USER"),
            password=os.getenv("CLICKHOUSE_PASSWORD"),
            secure=True,
        )
    return clickhouse_client
