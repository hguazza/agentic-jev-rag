# A LiveKit voice front end that follows the interview graph

Live interviews get a voice through a LiveKit Agents worker with a cascaded pipeline: LiveKit Inference speech-to-text, LiveKit's turn detector, a Voice host LLM, LiveKit Inference text-to-speech. The worker is a thin HTTP client of the Recruiter API's `/interviews` endpoints.

- **The interview graph is the authority.** It decides what is asked, when to probe, when to stop, and the assessment. The Voice host only delivers: it may smooth a question's wording but not change, add, skip or reorder questions.
- **Answers are transcript-verbatim.** The host's only tool, `submit_answer()`, takes no text. The worker sends the candidate's final transcripts since the question as they are, so what is graded is exactly what was heard, and the host cannot paraphrase or inject.
- **Speech models are a scoped exception to ADR 0005.** OpenRouter has no real-time speech, so STT and TTS run on LiveKit Inference. Every LLM, the Voice host included, still goes through OpenRouter.
- **Cascaded, not speech-to-speech.** The slow step in a turn is the graph's System 2 call, not speech; a cascaded pipeline keeps the graded text identical to the transcript and gives the turn detector for thinking pauses. A Realtime model remains a one-line swap of `AgentSession(llm=...)`.
- **Retries never record an answer twice.** The worker resends a reply only after a 5xx or a connection failure; the API then continues from its checkpoint instead of recording the message again. A timeout ends the interview, since the API may still be processing the answer.
