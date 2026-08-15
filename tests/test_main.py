import asyncio

import main


def test_parse_cors_origins(monkeypatch):
    monkeypatch.setenv(
        'CORS_ORIGINS',
        'http://localhost:3001, http://localhost:3005,',
    )

    assert main.parse_cors_origins() == [
        'http://localhost:3001',
        'http://localhost:3005',
    ]


def test_health_reports_degraded_without_model(monkeypatch):
    monkeypatch.setattr(main, 'model', None)
    monkeypatch.setattr(main, 'tokenizer', None)

    response = asyncio.run(main.health_check())

    assert response == {
        'status': 'degraded',
        'model_loaded': False,
    }
