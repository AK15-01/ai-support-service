# API Rate Limits

The Acme Markets REST and WebSocket APIs enforce rate limits per API key. Limits protect platform stability
and are applied independently to each key, even when several keys belong to the same account.

## Limits by plan

| Plan | Requests per minute | Requests per day | Concurrent WebSocket connections |
|------|---------------------|------------------|----------------------------------|
| Basic | 60 | 10,000 | 2 |
| Pro | 300 | 100,000 | 10 |
| Institutional | 1,200 | 1,000,000 | 50 |

## Rate limit headers

Every REST response includes three headers: `X-RateLimit-Limit` (the quota for the current window),
`X-RateLimit-Remaining` (requests left in the window) and `X-RateLimit-Reset` (the Unix time in seconds
when the window resets).

## Exceeding the limit

When a client exceeds the rate limit, the API returns HTTP 429 Too Many Requests with a `Retry-After` header
giving the number of seconds to wait. Clients should retry with exponential backoff, starting at 1 second and
capped at 60 seconds. If a key receives more than 50 HTTP 429 responses within 10 minutes, the key is
temporarily suspended for 15 minutes.

## Endpoint-specific limits

The order placement endpoint (`POST /v1/orders`) is limited to 20 requests per second per key on every plan.
The token endpoint (`POST /v1/auth/token`) is limited to 10 requests per minute per IP address. These limits
apply in addition to the plan limits above.

## Requesting a higher limit

Pro and Institutional customers can request a higher limit from the developer portal. Requests must describe
the use case and expected traffic, and are reviewed within 3 business days. Basic plan customers must upgrade
their plan before requesting an increase.
