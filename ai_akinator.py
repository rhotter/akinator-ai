#!/usr/bin/env python3
"""
AI vs AI Akinator Game
Two AIs play against each other - one knows a concept, the other tries to guess it.
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI


class AIAkinator:
    def __init__(self, api_key: Optional[str] = None, stream: bool = False):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.stream = stream
        self.max_questions = 100
        self.question_count = 0
        self.conversation_history: List[Dict[str, str]] = []

    def _make_api_call(
        self, messages: List[Dict[str, str]], temperature: float = 0.7
    ) -> str:
        """Make an API call to OpenAI with optional streaming."""
        if self.stream:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            print()  # New line after streaming
            return full_response
        else:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=temperature
            )
            return response.choices[0].message.content

    def get_answerer_response(self, concept: str, question: str) -> str:
        """AI that knows the concept and answers yes/no questions."""

        messages = [
            {
                "role": "system",
                "content": f"""You are playing Akinator. You know the secret concept: "{concept}".
                
Your job is to answer yes/no questions about this concept. Answer only with:
- "Yes" if the question is true about the concept
- "No" if the question is false about the concept
- "Sometimes" or "Partially" if it's not clearly yes or no
- "I don't know" only if you genuinely cannot determine the answer
- "DONE" if the guesser has guessed the concept

Be accurate and helpful in your responses. The other AI is trying to guess your concept.""",
            },
            {"role": "user", "content": f"Question: {question}"},
        ]

        return self._make_api_call(messages, temperature=0.3)

    def get_guesser_question(self, conversation_history: List[Dict[str, str]]) -> str:
        """AI that asks questions to guess the concept."""
        history_text = ""
        if conversation_history:
            history_text = "\n".join(
                [
                    f"Q{i+1}: {item['question']}\nA{i+1}: {item['answer']}"
                    for i, item in enumerate(conversation_history)
                ]
            )

        messages = [
            {
                "role": "system",
                "content": """You are playing Akinator as the guesser. Your goal is to figure out what concept the other person is thinking of by asking strategic yes/no questions.

Guidelines:
- Ask clear, specific yes/no questions
- Start broad and narrow down based on answers
- Use the conversation history to inform your next question
- When you're confident about the answer, make a guess by saying "Is it [your guess]?"
- Be strategic and efficient with your questions

Don't repeat questions you've already asked.""",
            },
            {
                "role": "user",
                "content": f"""Based on this conversation history, what is your next question?

{history_text if history_text else "This is the first question."}

Ask your next question:""",
            },
        ]

        return self._make_api_call(messages, temperature=0.8)

    def play_game(self, concept: str) -> Dict[str, Any]:
        """Play a complete game and return the results."""
        print(f"🎮 Starting AI vs AI Akinator!")
        print(f"🤖 The answerer AI knows the concept: '{concept}'")
        print(f"🔍 The guesser AI will try to figure it out!")
        print(f"📊 Maximum questions allowed: {self.max_questions}")
        print("-" * 50)

        self.conversation_history = []
        self.question_count = 0

        while self.question_count < self.max_questions:
            self.question_count += 1

            print(f"\n📝 Question {self.question_count}")

            # Guesser asks a question
            if self.stream:
                print("🔍 Guesser: ", end="")
            question = self.get_guesser_question(self.conversation_history)
            if not self.stream:
                print(f"🔍 Guesser: {question}")

            # Answerer responds
            if self.stream:
                print("🤖 Answerer: ", end="")
            answer = self.get_answerer_response(concept, question)
            if not self.stream:
                print(f"🤖 Answerer: {answer}")

            # Record the exchange
            self.conversation_history.append({"question": question, "answer": answer})

            # Check if the answerer said DONE (correct guess)
            if answer == "DONE":
                print(f"\n🎉 SUCCESS! The guesser got it right!")
                print(f"✅ Correct answer: '{concept}'")
                print(f"🎯 Questions asked: {self.question_count}")
                return {
                    "success": True,
                    "concept": concept,
                    "questions_asked": self.question_count,
                    "conversation": self.conversation_history,
                }

        print(f"\n❌ GAME OVER! Maximum questions ({self.max_questions}) reached.")
        print(f"🤔 The guesser couldn't figure out: '{concept}'")
        return {
            "success": False,
            "concept": concept,
            "questions_asked": self.question_count,
            "conversation": self.conversation_history,
        }


def main():
    """Main function to run the game."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Please set your OPENAI_API_KEY environment variable")
        return

    # Use a default concept for demonstration
    concept = "Einstein"
    print(f"Using concept: {concept}")

    # Enable streaming for demonstration
    use_streaming = False
    print(f"Streaming enabled: {use_streaming}")

    # Create and run the game
    game = AIAkinator(stream=use_streaming)
    result = game.play_game(concept)

    # Print final summary
    print("\n" + "=" * 50)
    print("📊 GAME SUMMARY")
    print("=" * 50)
    print(f"Concept: {result['concept']}")
    print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"Questions asked: {result['questions_asked']}")


if __name__ == "__main__":
    main()
