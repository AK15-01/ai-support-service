# Authentication and Password Reset

Acme Markets accounts sign in with a registered email address and a password. Two-factor
authentication (2FA) is strongly recommended and is mandatory for accounts with API access.

## Password requirements

- Passwords must be at least 12 characters long.
- Passwords must contain at least one uppercase letter, one lowercase letter, one number and one symbol.
- The last 5 passwords used on the account cannot be reused.
- Passwords never expire automatically, but customers are prompted to change them after a suspected compromise.

## Resetting your password

1. Select **Forgot password** on the sign-in page.
2. Enter the email address registered to the account.
3. Open the reset link sent to that email address and choose a new password.

Reset links are valid for 30 minutes and can be used only once. Requesting a new link invalidates any
earlier link that has not been used yet.

## Password reset limits

Each account may make up to 5 password reset attempts per hour. Additional reset requests within the same
hour are rejected with the message "Too many reset attempts", and the counter clears 60 minutes after the
first attempt in the window. Support agents cannot override this limit.

## Failed sign-in lockout

After 10 consecutive failed sign-in attempts, the account is locked for 30 minutes. A successful password
reset removes the lockout immediately. Repeated lockouts within 24 hours trigger a security review.

## Two-factor authentication

Supported 2FA methods are authenticator apps (TOTP) and SMS one-time codes. SMS is offered only as a
fallback when an authenticator app is unavailable. When 2FA is enabled, customers receive 10 single-use
backup codes. Changing the password signs out all other active sessions but does not disable 2FA.

## Lost access to the registered email

If a customer can no longer access the registered email address, the password cannot be reset through the
self-service flow. The customer must contact support, complete identity verification, and then support
updates the email address before a new reset link is issued.
