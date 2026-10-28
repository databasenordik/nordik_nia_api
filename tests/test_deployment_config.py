import pytest

from app.config import Settings, get_settings, validate_production_settings
from app.tts.xai_tts import XAITTSProvider


def test_cloud_run_environment_overrides_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / '.env'
    dotenv.write_text('REASONING_MODEL=local\nXAI_TTS_URL=wss://local/tts\n')
    monkeypatch.setenv('REASONING_MODEL', 'cloud-model')
    monkeypatch.setenv('XAI_TTS_URL', 'wss://internal.example/tts')
    monkeypatch.setenv('DB_POOL_MAX', '4')
    settings = Settings(_env_file=dotenv)
    assert settings.reasoning_model == 'cloud-model'
    assert settings.xai_tts_url == 'wss://internal.example/tts'
    assert settings.db_pool_max == 4


def test_speech_provider_uses_environment_endpoint(monkeypatch):
    monkeypatch.setenv('XAI_TTS_URL', 'wss://internal.example/tts')
    get_settings.cache_clear()
    try:
        assert XAITTSProvider()._url == 'wss://internal.example/tts'
    finally:
        get_settings.cache_clear()


def test_production_rejects_missing_credentials():
    settings = Settings(_env_file=None, app_env='production', jwt_secret='',
                        livekit_api_secret='', standalone_demo_username='',
                        standalone_demo_password='', standalone_demo_principal_id='',
                        auth_adapter='standalone')
    with pytest.raises(RuntimeError, match='JWT_SECRET'):
        validate_production_settings(settings)


def test_production_accepts_explicit_website_configuration():
    settings = Settings(_env_file=None, app_env='production', jwt_secret='s' * 32,
                        auth_cookie_secure=True, livekit_api_secret='configured',
                        auth_adapter='website', website_jwt_secret='w' * 32,
                        website_jwt_issuer='website', website_jwt_audience='nia',
                        cors_origins='')
    validate_production_settings(settings)
