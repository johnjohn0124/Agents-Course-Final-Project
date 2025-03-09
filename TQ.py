import random
from typing import List, Optional, Tuple, Dict

SEED_WORDS = ["apple", "elephant", "guitar", "mountain", "ocean", "telescope", "butterfly", "volcano", "pyramid", "kangaroo", 
              "computer", "airplane", "river", "moon", "castle", "tiger", "camera", "bridge", "planet", "statue"]


CITIES = ["New York", "London", "Paris", "Tokyo", "Sydney", "Berlin", "Dubai", "Mumbai", "Rio de Janeiro", "Cape Town", 
          "Toronto", "Singapore", "Hong Kong", "Los Angeles", "Chicago", "Moscow", "Rome", "Bangkok", "Istanbul", "Barcelona"]

class TwentyQuestionsEnvironment:

    def __init__(self, word_list: List[str] = SEED_WORDS, max_questions: int = 20):
        self.word_list = word_list
        self.max_questions = max_questions
        self.current_word: Optional[str] = None
        self.count = 0
        self.random = random.Random()
        self.log: List[Dict] = []  # Stores episode logs
    
    def reset(self, seed: Optional[int] = None) -> str:
        """Resets the environment and selects a new word."""
        self.count = 0
        if seed is not None:
            self.random.seed(seed)
        self.current_word = self.random.choice(self.word_list)
        self.log.append({"word": self.current_word, "questions": [], "success": False})
        return "Game started. Ask a yes/no question."
    
    def step(self, question: str) -> Tuple[str, bool]:
        """Processes a question and returns the answer along with whether the game is over."""
        if self.current_word is None:
            raise ValueError("Environment must be reset before playing.")
        
        self.count += 1
        answer = self._generate_answer(question)
        done = self.count >= self.max_questions
        
        # Log question and answer
        self.log[-1]["questions"].append({"question": question, "answer": answer})
        
        return answer, done
    
    def _generate_answer(self, question: str) -> str:
        """Generates a yes/no/maybe answer for the given question."""
        # A simple mock oracle that randomly responds for now
        return random.choice(["Yes", "No", "Maybe"])
    
    def get_answer(self) -> str:
        """Returns the correct answer when the game is over."""
        if self.current_word is None:
            raise ValueError("Environment must be reset before retrieving the answer.")
        
        # Mark success if guessed correctly (mocked here)
        self.log[-1]["success"] = True
        return f"The correct word was: {self.current_word}"
    
    def get_log(self) -> List[Dict]:
        """Returns the log of all episodes."""
        return self.log



class GuessMyCityEnvironment:

    def __init__(self, city_list: List[str] = CITIES, max_questions: int = 20):
        self.city_list = city_list
        self.max_questions = max_questions
        self.current_city: Optional[str] = None
        self.count = 0
        self.random = random.Random()
        self.log: List[Dict] = []  # Stores episode logs
    
    def reset(self, seed: Optional[int] = None) -> str:
        """Resets the environment and selects a new city."""
        self.count = 0
        if seed is not None:
            self.random.seed(seed)
        self.current_city = self.random.choice(self.city_list)
        self.log.append({"city": self.current_city, "questions": [], "success": False})
        return "Game started. Ask a question to guess the city."
    
    def step(self, question: str) -> Tuple[str, bool]:
        """Processes a question and returns the answer along with whether the game is over."""
        if self.current_city is None:
            raise ValueError("Environment must be reset before playing.")
        
        self.count += 1
        answer = self._generate_answer(question)
        done = self.count >= self.max_questions
        
        # Log question and answer
        self.log[-1]["questions"].append({"question": question, "answer": answer})
        
        return answer, done
    
    def _generate_answer(self, question: str) -> str:
        """Generates an open-ended response based on the question."""
        if "capital" in question.lower():
            return "Yes" if self.current_city in ["London", "Paris", "Tokyo", "Berlin", "Moscow", "Rome", "Bangkok", "Istanbul"] else "No"
        elif "continent" in question.lower():
            return f"The city is in {random.choice(['Europe', 'Asia', 'America', 'Africa', 'Australia'])}."
        elif "famous for" in question.lower():
            return f"{self.current_city} is famous for its {random.choice(['landmarks', 'culture', 'food', 'history', 'architecture'])}."
        else:
            return random.choice(["Yes", "No", "Maybe", "It depends.", "Interesting question!"])
    
    def get_answer(self) -> str:
        """Returns the correct answer when the game is over."""
        if self.current_city is None:
            raise ValueError("Environment must be reset before retrieving the answer.")
        
        # Mark success if guessed correctly (mocked here)
        self.log[-1]["success"] = True
        return f"The correct city was: {self.current_city}"
    
    def get_log(self) -> List[Dict]:
        """Returns the log of all episodes."""
        return self.log


#Basic Test Script

def test_environment():
    env = TwentyQuestionsEnvironment()
    
    print(env.reset(seed=42))  
    print(f"Selected word: {env.current_word}")  
    
    questions = [
        "Is it a living thing?", 
        "Is it an animal?", 
        "Is it found in the ocean?"
    ]

    for q in questions:
        answer, done = env.step(q)
        print(f"Q: {q} | A: {answer} | Done: {done}")
        if done:
            break
    
    print(env.get_answer())  
    print("Episode Log:", env.get_log())

test_environment()



#slIntegration Check for RL Algorithms

def dummy_agent_play():
    env = TwentyQuestionsEnvironment()
    obs = env.reset()
    done = False
    turns = 0
    
    while not done:
        question = random.choice(["Is it an animal?", "Is it big?", "Can you eat it?", "Is it found in nature?", "Does it fly?"])
        answer, done = env.step(question)
        print(f"Turn {turns+1}: {question} -> {answer}")
        turns += 1
    
    print("Final Answer:", env.get_answer())
    print("Full Log:", env.get_log())

dummy_agent_play()




def test_gmc_environment():
    env = GuessMyCityEnvironment()

    print(env.reset(seed=42))  
    print(f"Selected city: {env.current_city}")  # Should be consistent with the seed

    questions = [
        "Is this city a capital?",
        "Is it located in Europe?",
        "What is this city famous for?",
        "Is this city near the coast?",
        "Does it have a large population?"
    ]

    for q in questions:
        answer, done = env.step(q)
        print(f"Q: {q} | A: {answer} | Done: {done}")
        if done:
            break

    print(env.get_answer())  # Should return the correct city
    print("Episode Log:", env.get_log())

test_gmc_environment()

