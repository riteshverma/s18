import asyncio
import json

from core import scheduler as scheduler_module


def test_scheduler_save_jobs_async(tmp_path):
    jobs_file = tmp_path / "jobs.json"
    original_jobs_file = scheduler_module.JOBS_FILE
    scheduler_module.JOBS_FILE = jobs_file
    try:
        service = scheduler_module.SchedulerService()
        service.jobs = {}
        service.jobs["job1"] = scheduler_module.JobDefinition(
            id="job1",
            name="Test Job",
            cron_expression="*/5 * * * *",
            agent_type="PlannerAgent",
            query="hello",
        )
        asyncio.run(service.save_jobs_async())
        payload = json.loads(jobs_file.read_text(encoding="utf-8"))
        assert payload[0]["id"] == "job1"
    finally:
        scheduler_module.JOBS_FILE = original_jobs_file
