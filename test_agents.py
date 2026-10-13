from agents import classify_issue, find_duplicate, triage_issue

def test_classify_login_issue():
    result = classify_issue("Login fails", "Users cannot login after password reset")
    assert result["category"] == "Authentication"
    assert result["severity"] in {"Critical", "High", "Medium", "Low"}

def test_duplicate_detection():
    issues = [{
        "issue_id": "BUG-1",
        "title": "Login fails after reset",
        "description": "Users cannot login after resetting password"
    }]
    duplicate_id, score = find_duplicate(
        "Login failure after reset",
        "Users cannot login after password reset",
        issues,
        threshold=0.30
    )
    assert duplicate_id == "BUG-1"
    assert score >= 0.30

def test_missing_fields_rejected():
    try:
        triage_issue({"title": "", "description": ""}, [])
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
