import importlib
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


if __name__ == "__main__":
    unittest.main()
