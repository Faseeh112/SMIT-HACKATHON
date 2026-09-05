"""Tests for individual agent tools (no LLM involved)."""
from __future__ import annotations

import pytest

from tools import check_eligibility, create_complaint, schedule_callback


def test_check_eligibility_missing_fields_returns_insufficient_information():
    result = check_eligibility.run(program="saylani_it_courses")
    assert result["status"] == "insufficient_information"
    assert "age" in result["missing_fields"] or "city" in result["missing_fields"]


def test_check_eligibility_eligible_case():
    result = check_eligibility.run(program="saylani_it_courses", age=20, city="Karachi")
    assert result["status"] == "eligible"


def test_check_eligibility_not_eligible_age():
    result = check_eligibility.run(program="saylani_it_courses", age=10, city="Karachi")
    assert result["status"] == "not_eligible"
    assert any("age" in r for r in result["reasons"])


def test_check_eligibility_unknown_program():
    result = check_eligibility.run(program="not_a_real_program")
    assert result["status"] == "insufficient_information"


def test_create_complaint_requires_fields(temp_db):
    result = create_complaint.run(user_id="u1", category="", description="")
    assert result["status"] == "error"


def test_create_complaint_success(temp_db):
    temp_db.ensure_user("u1")
    result = create_complaint.run(user_id="u1", category="course", description="Class was cancelled.")
    assert result["status"] == "success"
    assert result["complaint_id"].startswith("CMP-")


def test_schedule_callback_validates_date_format(temp_db):
    temp_db.ensure_user("u1")
    result = schedule_callback.run(user_id="u1", date="not-a-date", time="10:00", contact="0300-1234567")
    assert result["status"] == "error"


def test_schedule_callback_validates_contact(temp_db):
    temp_db.ensure_user("u1")
    result = schedule_callback.run(user_id="u1", date="2026-09-01", time="10:00", contact="???")
    assert result["status"] == "error"


def test_schedule_callback_success(temp_db):
    temp_db.ensure_user("u1")
    result = schedule_callback.run(
        user_id="u1", date="2026-09-01", time="10:00", contact="0300-1234567", reason="Course question"
    )
    assert result["status"] == "success"
    assert result["callback_id"].startswith("CB-")
