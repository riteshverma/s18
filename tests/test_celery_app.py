import importlib
import sys
import types
import unittest
from unittest.mock import patch

import core.celery_app as celery_module


def _reload_celery_app():
    return importlib.reload(celery_module)


class CeleryRoutingTests(unittest.TestCase):
    def test_defaults_use_celery_for_runs_and_ingest_for_ingest_tasks(self):
        with patch.dict("os.environ", {}, clear=False):
            mod = _reload_celery_app()
            self.assertEqual(mod.RUNS_QUEUE, "celery")
            self.assertEqual(mod.INGEST_QUEUE, "ingest")
            self.assertEqual(mod.celery_app.conf.task_default_queue, "celery")
            self.assertEqual(
                mod.celery_app.conf.task_routes["s18share.run_agent"]["queue"], "celery"
            )
            self.assertEqual(
                mod.celery_app.conf.task_routes["s18share.ingest.materialize"]["queue"],
                "ingest",
            )

    def test_env_can_override_runs_and_ingest_queue_names(self):
        with patch.dict(
            "os.environ",
            {
                "S18_CELERY_RUNS_QUEUE": "runs_high",
                "S18_CELERY_INGEST_QUEUE": "ingest_bulk",
            },
            clear=False,
        ):
            mod = _reload_celery_app()
            self.assertEqual(mod.RUNS_QUEUE, "runs_high")
            self.assertEqual(mod.INGEST_QUEUE, "ingest_bulk")
            self.assertEqual(mod.celery_app.conf.task_default_queue, "runs_high")
            self.assertEqual(
                mod.celery_app.conf.task_routes["s18share.resume_agent"]["queue"],
                "runs_high",
            )
            self.assertEqual(
                mod.celery_app.conf.task_routes["s18share.ingest.embed_and_index"][
                    "queue"
                ],
                "ingest_bulk",
            )


class CeleryExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_run_enqueue_marks_run_failed(self):
        from core.run_executor import execute_run
        from integrations.contracts import CanonicalRunRequest

        class FakeStore:
            def __init__(self):
                self.row = {
                    "id": "run_enqueue_fail",
                    "status": "accepted",
                    "metadata": {"source_system": "test"},
                }

            def get_run(self, _run_id):
                return dict(self.row)

            def update_status(self, run_id, status, **fields):
                self.row.update({"id": run_id, "status": status, **fields})
                return dict(self.row)

            def update_run(self, run_id, **fields):
                self.row.update({"id": run_id, **fields})
                return dict(self.row)

        class FailingTask:
            def delay(self, *_args, **_kwargs):
                raise RuntimeError("Celery is not installed")

        fake_store = FakeStore()
        fake_agent_tasks = types.ModuleType("workers.agent_tasks")
        fake_agent_tasks.run_agent_task = FailingTask()

        with patch.dict("os.environ", {"S18_RUN_EXECUTOR": "celery"}, clear=False), patch.dict(
            sys.modules,
            {"workers.agent_tasks": fake_agent_tasks},
        ), patch("core.run_store.get_run_store", return_value=fake_store):
            with self.assertRaisesRegex(RuntimeError, "Celery is not installed"):
                await execute_run(
                    "run_enqueue_fail",
                    CanonicalRunRequest(query="check system"),
                )

        self.assertEqual(fake_store.row["status"], "failed")
        self.assertEqual(fake_store.row["error"], "Celery is not installed")
        self.assertEqual(fake_store.row["metadata"]["source_system"], "test")
        self.assertEqual(fake_store.row["metadata"]["execution_backend"], "celery")
        self.assertEqual(
            fake_store.row["metadata"]["enqueue_error"],
            "Celery is not installed",
        )


if __name__ == "__main__":
    unittest.main()
