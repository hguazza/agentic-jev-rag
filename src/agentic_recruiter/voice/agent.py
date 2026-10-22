"""The Voice host as a LiveKit agent: it delivers the interview graph's questions and hands the answers back."""

from livekit.agents import Agent, RunContext, ToolError, function_tool

from agentic_recruiter.voice.interview import EmptyAnswer, InterviewFailed, InterviewOver, Next, VoiceInterview
from agentic_recruiter.voice.script import FAILED, FILLER, GREETING, INSTRUCTIONS, next_instruction


class InterviewerVoice(Agent):
    def __init__(self, interview: VoiceInterview, first: Next):
        super().__init__(instructions=INSTRUCTIONS)
        self._interview = interview
        self._first = first

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions=GREETING + next_instruction(self._first))

    @function_tool()
    async def submit_answer(self, context: RunContext) -> str:
        """Send the candidate's spoken answer to the current question to the interview system.

        Call it only once the candidate has finished answering. It returns what to say next.
        """
        try:
            # Speaks only if the graph takes longer than 3 s after the acknowledgement.
            async with context.with_filler(FILLER, delay=3.0):
                next = await self._interview.submit()
        except EmptyAnswer as error:
            raise ToolError("The candidate has not answered yet. Wait for their answer, or repeat the question.") from error
        except InterviewOver as error:
            raise ToolError("The interview is already over. Say goodbye.") from error
        except InterviewFailed:
            return FAILED
        return next_instruction(next)
