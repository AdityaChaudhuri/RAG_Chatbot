from supabase import create_client, Client
from app.config import settings

# Service-role client: bypasses RLS — used server-side only.
# Never expose this key to the frontend.
_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _client


# Convenience alias used throughout the app
supabase: Client = get_supabase()
