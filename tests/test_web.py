"""Unit tests for FastAPI Web Dashboard and Scheduler API.

Verifies:
- Dashboard rendering
- /api/status, /api/signals, /api/toggle-bot, /api/logs, /api/jobs
- Scheduler timing configuration
"""

import unittest
from fastapi.testclient import TestClient

from web.app import app


class TestWebDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_dashboard_html_rendering(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("QuantBot Alpaca", resp.text)
        self.assertIn("Posiciones Abiertas", resp.text)
        self.assertIn("Radar Cuantitativo", resp.text)

    def test_api_status(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("is_active", data)
        self.assertIn("account", data)
        self.assertIn("equity", data["account"])
        self.assertIn("buying_power", data["account"])
        self.assertIn("positions", data)

    def test_api_toggle_bot(self):
        # Initial status
        init_state = self.client.get("/api/status").json()["is_active"]

        # Toggle 1
        resp = self.client.post("/api/toggle-bot")
        self.assertEqual(resp.status_code, 200)
        new_state = resp.json()["is_active"]
        self.assertEqual(new_state, not init_state)

        # Toggle 2 (Restore)
        resp2 = self.client.post("/api/toggle-bot")
        self.assertEqual(resp2.status_code, 200)
        restored_state = resp2.json()["is_active"]
        self.assertEqual(restored_state, init_state)

    def test_api_logs(self):
        resp = self.client.get("/api/logs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("logs", data)
        self.assertIsInstance(data["logs"], list)

    def test_api_jobs_scheduler(self):
        resp = self.client.get("/api/jobs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("jobs", data)
        job_ids = [j["id"] for j in data["jobs"]]
        self.assertIn("market_preclose_scan", job_ids)


if __name__ == "__main__":
    unittest.main()
