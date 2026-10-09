# Incident Escalation

This runbook describes how support agents classify and escalate production incidents.

## Severity levels

| Severity | Definition | Example | Initial response target |
|----------|------------|---------|-------------------------|
| SEV1 | Complete outage or security breach affecting many customers | Trading platform unavailable | 15 minutes |
| SEV2 | Major feature degraded with no workaround | Withdrawals delayed by more than 4 hours | 1 hour |
| SEV3 | Minor issue with a workaround | A single report renders incorrectly | 1 business day |

## How to escalate

Support agents open an incident in the incident tracker, assign a severity, and attach customer reports.
For SEV1 and SEV2 incidents the agent also pages the on-call engineer through the paging system. SEV3
incidents are handled during business hours by the owning team.

## Escalation path

If a SEV1 incident is not resolved within 30 minutes, the on-call engineer escalates it to the Engineering
Manager. If it is still not resolved after 1 hour, the Head of Engineering is notified and an incident
commander is appointed. SEV2 incidents are escalated to the Engineering Manager after 4 hours without
resolution.

## Customer communication

During a SEV1 incident the public status page is updated every 30 minutes. During a SEV2 incident it is
updated every 2 hours. Agents must use the approved status page wording and must not give customers an
estimated fix time unless the incident commander has confirmed it.

## Security incidents

Suspected account compromise or data breaches are always treated as SEV1. The agent notifies the security
team immediately and must not discuss details with the customer until the security team approves the
message.

## Post-incident review

A blameless post-incident review is written within 5 business days for every SEV1 and SEV2 incident. The
review records the timeline, root cause, customer impact, and follow-up actions with owners.
