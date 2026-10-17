"""
app.engines
===========
Deterministic financial calculation engines. No LLM involved here — every
function returns exact, auditable numbers. The LLM agent calls these as
*tools* and explains the results in natural language; it never does the
arithmetic itself. This separation is what makes the platform trustworthy
for financial advice: the math is always correct, the LLM only narrates it.
"""
