import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel
import uuid
import pytz
from config.settings_loader import settings

# Setup logging
logger = logging.getLogger("scheduler")

# Path to persist jobs
JOBS_FILE = Path("data/system/jobs.json")
_scheduler_tz = settings.get("scheduler", {}).get("timezone", "UTC")
SCHEDULER_TIMEZONE = pytz.timezone(_scheduler_tz)

class JobDefinition(BaseModel):
    id: str
    name: str
    cron_expression: str
    agent_type: str
    query: str
    skill_id: Optional[str] = None  # Link to a specific skill
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_output: Optional[str] = None

class SchedulerService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SchedulerService, cls).__new__(cls)
            cls._instance.scheduler = AsyncIOScheduler(timezone=SCHEDULER_TIMEZONE)
            cls._instance.jobs: Dict[str, JobDefinition] = {}
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return
        
        # Ensure data directory exists
        if not JOBS_FILE.parent.exists():
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            
        self.load_jobs()
        self.scheduler.start()
        logger.info("✅ Scheduler Service Started")
        self.initialized = True

    def load_jobs(self):
        """Load jobs from JSON and schedule them."""
        if not JOBS_FILE.exists():
            return

        try:
            data = json.loads(JOBS_FILE.read_text())
            for job_data in data:
                job_def = JobDefinition(**job_data)
                self.jobs[job_def.id] = job_def
                if job_def.enabled:
                    self._schedule_job(job_def)
            logger.info(f"Loaded {len(self.jobs)} jobs from disk.")
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")

    def save_jobs(self):
        """Persist jobs to JSON."""
        data = [job.dict() for job in self.jobs.values()]
        JOBS_FILE.write_text(json.dumps(data, indent=2))

    async def save_jobs_async(self):
        await asyncio.to_thread(self.save_jobs)

    def _schedule_job(self, job: JobDefinition):
        """Internal method to add job to APScheduler."""
        
        # Define the function wrapper
        async def job_wrapper():
            # Lazy import to avoid circular dependency
            from routers.runs import process_run
            from .skills.lifecycle import resolve_skill, run_skill_failure, run_skill_success
            from core.event_bus import event_bus
            from routers.inbox import send_to_inbox
            from integrations.contracts import CanonicalRunRequest
            
            run_id = f"auto_{job.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Emit Start Event
            log_msg = f"⏰ Triggering Scheduled Job: {job.name} ({run_id})"
            logger.info(log_msg)
            await event_bus.publish(
                "log",
                "scheduler",
                {
                    "message": log_msg,
                    "metadata": {"job_id": job.id, "run_id": run_id}
                }
            )
            
            # Update last run
            job.last_run = datetime.now().isoformat()
            await self.save_jobs_async()
            
            skill = None
            try:
                # Skill Lifecycle Execution
                skill, effective_query, resolved_skill_id = await resolve_skill(
                    query=job.query,
                    run_id=run_id,
                    agent_id=job.agent_type,
                    explicit_skill_id=job.skill_id,
                    integration_id="default",
                    workflow_id="generic",
                )
                if skill and resolved_skill_id:
                    msg = f"🧠 Skill '{resolved_skill_id}' modified prompt: {effective_query[:50]}..."
                    logger.info(msg)
                    await event_bus.publish("log", "scheduler", {"message": msg})

                # 3. Execution (The standard run)
                canonical_request = CanonicalRunRequest(query=effective_query)
                result = await process_run(run_id, canonical_request)
                
                # 4. Hook: On Success (The "Doing")
                skill_result = await run_skill_success(
                    skill,
                    result if isinstance(result, dict) else {"output": str(result)},
                )

                # Notify Success
                success_msg = f"✅ Job '{job.name}' completed successfully."
                await event_bus.publish(
                    "success", 
                    "scheduler", 
                    {
                        "message": success_msg,
                        "metadata": {"job_id": job.id, "run_id": run_id}
                    }
                )
                
                # Update job with result
                job.last_output = skill_result.get("summary") if skill_result else (result.get("summary") if result else "Success")
                await self.save_jobs_async()

                # Build rich notification body
                notif_body = f"Job '{job.name}' finished.\n\n"
                if skill_result and skill_result.get("summary"):
                    notif_body += f"**Summary**: {skill_result['summary']}\n\n"
                elif result and result.get("summary"):
                    notif_body += f"**Summary**: {result['summary'][:200]}...\n\n"
                
                notif_body += f"*Run ID: {run_id}*"
                
                send_to_inbox(
                    source="Scheduler",
                    title=f"Completed: {job.name}",
                    body=notif_body,
                    priority=1,
                    metadata={
                        "job_id": job.id, 
                        "run_id": run_id,
                        "file_path": skill_result.get("file_path") if skill_result else None
                    }
                )

            except Exception as e:
                error_msg = f"❌ Job {job.name} failed: {e}"
                logger.error(error_msg)
                
                await event_bus.publish(
                    "error",
                    "scheduler",
                    {
                        "message": error_msg
                    }
                )

                send_to_inbox(
                    source="Scheduler",
                    title=f"Job Failed: {job.name}",
                    body=f"Error: {str(e)}",
                    priority=2 # High priority for failures
                )

                await run_skill_failure(skill, str(e))

        # Parse cron expression (simple space-separated 5 fields)
        try:
            self.scheduler.add_job(
                job_wrapper,
                CronTrigger.from_crontab(job.cron_expression),
                id=job.id,
                name=job.name,
                replace_existing=True
            )
            # Update next run time
            aps_job = self.scheduler.get_job(job.id)
            if aps_job and aps_job.next_run_time:
                job.next_run = aps_job.next_run_time.isoformat()
                asyncio.create_task(self.save_jobs_async())
                
        except Exception as e:
            logger.error(f"Invalid cron expression for {job.name}: {e}")

    def add_job(self, name: str, cron_expression: str, agent_type: str, query: str) -> JobDefinition:
        """Add a new scheduled job."""
        # Auto-detect skill
        from .skills.manager import skill_manager
        # Ensure registry is loaded
        if not skill_manager.skill_classes:
            skill_manager.initialize()
            
        skill_id = skill_manager.match_intent(query)
        if skill_id:
            logger.info(f"🧠 Smart Scheduler: Matched query '{query}' to Skill '{skill_id}'")

        job_id = str(uuid.uuid4())[:8]
        job = JobDefinition(
            id=job_id,
            name=name,
            cron_expression=cron_expression,
            agent_type=agent_type,
            query=query,
            skill_id=skill_id
        )
        self.jobs[job_id] = job
        self._schedule_job(job)
        self.save_jobs()
        return job

    def trigger_job(self, job_id: str):
        """Force a job to run immediately."""
        if job_id not in self.jobs:
            logger.warning(f"Trigger failed: Job {job_id} not found in registry")
            return
            
        if self.scheduler.get_job(job_id):
             self.scheduler.modify_job(job_id, next_run_time=datetime.now())
             logger.info(f"👉 Setup immediate execution for {job_id}")
        else:
             # If it was paused or missing
             self._schedule_job(self.jobs[job_id])
             self.scheduler.modify_job(job_id, next_run_time=datetime.now())

    def delete_job(self, job_id: str):
        """Remove a job."""
        if job_id in self.jobs:
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            self.save_jobs()

    def list_jobs(self) -> List[JobDefinition]:
        """List all jobs with updated next-run times."""
        # Update next_run times from scheduler
        for job_id, job in self.jobs.items():
            aps_job = self.scheduler.get_job(job_id)
            if aps_job and aps_job.next_run_time:
                job.next_run = aps_job.next_run_time.isoformat()
        
        return list(self.jobs.values())

    def register_morning_briefing(self):
        """Register the built-in Morning Briefing job at 7 AM daily (Scheduler + Inbox + EventBus)."""
        from core.morning_briefing import run_morning_briefing

        async def wrapper():
            await run_morning_briefing()

        job_id = "morning_briefing"
        try:
            self.scheduler.add_job(
                wrapper,
                CronTrigger(hour=7, minute=0),
                id=job_id,
                name="Morning Briefing",
                replace_existing=True,
            )
            logger.info("✅ Morning Briefing scheduled daily at 7:00 AM")
        except Exception as e:
            logger.error("Failed to schedule Morning Briefing: %s", e)

# Global Instance
scheduler_service = SchedulerService()
