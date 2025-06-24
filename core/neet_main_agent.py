#!/usr/bin/env python3
"""
AI NEET Coach - Main Agent
A sophisticated AI coaching system for NEET preparation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    CONCEPT_EXPLANATION = "concept"
    MCQ_SOLVING = "mcq"
    DOUBT_CLARIFICATION = "doubt"
    STRATEGY_ADVICE = "strategy"
    GENERAL_CHAT = "general"

@dataclass
class StudentQuery:
    text: str
    query_type: QueryType
    subject: Optional[str] = None
    difficulty_level: Optional[str] = None
    context: Optional[str] = None

@dataclass
class AgentResponse:
    content: str
    confidence: float
    sources: List[str]
    reasoning_steps: List[str]
    tricks_detected: List[str]
    follow_up_questions: List[str]

class NEETCoachAgent:
    """
    Main orchestrator for the AI NEET Coach system
    Coordinates multiple specialized agents for comprehensive tutoring
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.conversation_history = []
        self.student_profile = {
            "weak_areas": [],
            "strong_areas": [],
            "learning_patterns": {},
            "mistake_patterns": {}
        }
        
        # Initialize components
        self._initialize_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "vector_store_type": "faiss",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "ollama/llama3.1:8b",
            "max_context_length": 4000,
            "confidence_threshold": 0.7,
            "data_path": "./data",
            "subjects": ["physics", "chemistry", "biology"]
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            return default_config
    
    def _initialize_components(self):
        """Initialize all agent components"""
        from memory.vector_store import VectorStore
        from agents.reasoning_agent import ReasoningAgent
        from agents.trick_classifier import TrickClassifier
        from agents.math_solver import MathSolver
        from ingest.ncert_loader import NCERTLoader
        from ingest.mcq_parser import MCQParser
        
        # Initialize vector store and knowledge base
        self.vector_store = VectorStore(
            store_type=self.config["vector_store_type"],
            embedding_model=self.config["embedding_model"]
        )
        
        # Initialize specialized agents
        self.reasoning_agent = ReasoningAgent(
            llm_model=self.config["llm_model"],
            vector_store=self.vector_store
        )
        
        self.trick_classifier = TrickClassifier()
        self.math_solver = MathSolver()
        
        # Initialize data loaders
        self.ncert_loader = NCERTLoader(self.config["data_path"])
        self.mcq_parser = MCQParser()
        
        logger.info("All components initialized successfully")
    
    async def process_query(self, query: str) -> AgentResponse:
        """
        Main query processing pipeline
        """
        # Parse and classify the query
        student_query = await self._parse_query(query)
        
        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "timestamp": asyncio.get_event_loop().time(),
            "type": student_query.query_type.value
        })
        
        # Route to appropriate processing pipeline
        if student_query.query_type == QueryType.MCQ_SOLVING:
            return await self._handle_mcq(student_query)
        elif student_query.query_type == QueryType.CONCEPT_EXPLANATION:
            return await self._handle_concept_explanation(student_query)
        elif student_query.query_type == QueryType.DOUBT_CLARIFICATION:
            return await self._handle_doubt_clarification(student_query)
        elif student_query.query_type == QueryType.STRATEGY_ADVICE:
            return await self._handle_strategy_advice(student_query)
        else:
            return await self._handle_general_chat(student_query)
    
    async def _parse_query(self, query: str) -> StudentQuery:
        """Parse and classify incoming query"""
        # Simple classification logic (can be enhanced with ML)
        query_lower = query.lower()
        
        # Detect MCQ patterns
        if any(pattern in query_lower for pattern in ["option", "choose", "correct answer", "a)", "b)", "c)", "d)"]):
            query_type = QueryType.MCQ_SOLVING
        elif any(pattern in query_lower for pattern in ["explain", "what is", "how does", "why"]):
            query_type = QueryType.CONCEPT_EXPLANATION
        elif any(pattern in query_lower for pattern in ["doubt", "confused", "don't understand"]):
            query_type = QueryType.DOUBT_CLARIFICATION
        elif any(pattern in query_lower for pattern in ["strategy", "preparation", "study plan", "time management"]):
            query_type = QueryType.STRATEGY_ADVICE
        else:
            query_type = QueryType.GENERAL_CHAT
        
        # Detect subject
        subject = None
        if "physics" in query_lower or any(term in query_lower for term in ["force", "energy", "momentum", "electric"]):
            subject = "physics"
        elif "chemistry" in query_lower or any(term in query_lower for term in ["reaction", "molecule", "bond", "acid"]):
            subject = "chemistry"
        elif "biology" in query_lower or any(term in query_lower for term in ["cell", "organ", "dna", "protein"]):
            subject = "biology"
        
        return StudentQuery(
            text=query,
            query_type=query_type,
            subject=subject
        )
    
    async def _handle_mcq(self, query: StudentQuery) -> AgentResponse:
        """Handle MCQ solving with trick detection"""
        logger.info(f"Processing MCQ: {query.text[:50]}...")
        
        # Extract relevant knowledge
        relevant_docs = await self.vector_store.similarity_search(
            query.text, 
            k=5,
            filter_subject=query.subject
        )
        
        # Detect potential tricks
        tricks_detected = await self.trick_classifier.analyze_question(query.text)
        
        # Generate reasoning
        reasoning_response = await self.reasoning_agent.solve_mcq(
            question=query.text,
            context_docs=relevant_docs,
            detected_tricks=tricks_detected
        )
        
        # Check if mathematical solving is needed
        math_solution = None
        if self._requires_math_solving(query.text):
            math_solution = await self.math_solver.solve(query.text)
        
        # Compile response
        response_content = self._compile_mcq_response(
            reasoning_response, 
            tricks_detected, 
            math_solution
        )
        
        return AgentResponse(
            content=response_content,
            confidence=reasoning_response.get("confidence", 0.8),
            sources=[doc["source"] for doc in relevant_docs],
            reasoning_steps=reasoning_response.get("steps", []),
            tricks_detected=tricks_detected,
            follow_up_questions=self._generate_follow_up_questions(query)
        )
    
    async def _handle_concept_explanation(self, query: StudentQuery) -> AgentResponse:
        """Handle concept explanation requests"""
        logger.info(f"Explaining concept: {query.text[:50]}...")
        
        # Retrieve relevant knowledge
        relevant_docs = await self.vector_store.similarity_search(
            query.text, 
            k=8,
            filter_subject=query.subject
        )
        
        # Generate comprehensive explanation
        explanation = await self.reasoning_agent.explain_concept(
            concept=query.text,
            context_docs=relevant_docs,
            student_level=self._estimate_student_level()
        )
        
        return AgentResponse(
            content=explanation["content"],
            confidence=explanation.get("confidence", 0.8),
            sources=[doc["source"] for doc in relevant_docs],
            reasoning_steps=explanation.get("explanation_flow", []),
            tricks_detected=[],
            follow_up_questions=explanation.get("follow_up_questions", [])
        )
    
    async def _handle_doubt_clarification(self, query: StudentQuery) -> AgentResponse:
        """Handle doubt clarification with personalized approach"""
        logger.info(f"Clarifying doubt: {query.text[:50]}...")
        
        # Get context from conversation history
        recent_context = self._get_recent_context()
        
        # Retrieve relevant knowledge
        relevant_docs = await self.vector_store.similarity_search(
            query.text, 
            k=6,
            filter_subject=query.subject
        )
        
        # Generate clarification
        clarification = await self.reasoning_agent.clarify_doubt(
            doubt=query.text,
            context_docs=relevant_docs,
            conversation_context=recent_context,
            student_weaknesses=self.student_profile["weak_areas"]
        )
        
        return AgentResponse(
            content=clarification["content"],
            confidence=clarification.get("confidence", 0.8),
            sources=[doc["source"] for doc in relevant_docs],
            reasoning_steps=clarification.get("clarification_steps", []),
            tricks_detected=[],
            follow_up_questions=clarification.get("follow_up_questions", [])
        )
    
    async def _handle_strategy_advice(self, query: StudentQuery) -> AgentResponse:
        """Provide strategic study advice"""
        logger.info(f"Providing strategy advice: {query.text[:50]}...")
        
        # Analyze student's current performance
        performance_analysis = self._analyze_student_performance()
        
        # Generate personalized strategy
        strategy = await self.reasoning_agent.generate_strategy(
            query=query.text,
            performance_data=performance_analysis,
            student_profile=self.student_profile
        )
        
        return AgentResponse(
            content=strategy["content"],
            confidence=strategy.get("confidence", 0.9),
            sources=["Student Performance Analysis", "NEET Strategy Guidelines"],
            reasoning_steps=strategy.get("reasoning", []),
            tricks_detected=[],
            follow_up_questions=strategy.get("follow_up_questions", [])
        )
    
    async def _handle_general_chat(self, query: StudentQuery) -> AgentResponse:
        """Handle general conversation"""
        logger.info(f"General chat: {query.text[:50]}...")
        
        response = await self.reasoning_agent.general_response(
            query=query.text,
            conversation_history=self.conversation_history[-5:],  # Last 5 exchanges
            student_profile=self.student_profile
        )
        
        return AgentResponse(
            content=response["content"],
            confidence=response.get("confidence", 0.7),
            sources=[],
            reasoning_steps=[],
            tricks_detected=[],
            follow_up_questions=response.get("follow_up_questions", [])
        )
    
    def _requires_math_solving(self, text: str) -> bool:
        """Check if question requires mathematical computation"""
        math_indicators = [
            "calculate", "find the value", "solve for", "what is the",
            "numerical", "equation", "formula", "=", "+", "-", "*", "/"
        ]
        return any(indicator in text.lower() for indicator in math_indicators)
    
    def _compile_mcq_response(self, reasoning_response: Dict, tricks: List[str], math_solution: Optional[Dict]) -> str:
        """Compile a comprehensive MCQ response"""
        response_parts = []
        
        # Main answer
        response_parts.append(f"**Answer: {reasoning_response.get('answer', 'Need more analysis')}**\n")
        
        # Reasoning explanation
        if reasoning_response.get("explanation"):
            response_parts.append(f"**Explanation:**\n{reasoning_response['explanation']}\n")
        
        # Mathematical solution if available
        if math_solution:
            response_parts.append(f"**Mathematical Solution:**\n{math_solution.get('solution', '')}\n")
        
        # Trick alerts
        if tricks:
            response_parts.append("**⚠️ Potential Tricks Detected:**")
            for trick in tricks:
                response_parts.append(f"• {trick}")
            response_parts.append("")
        
        # Step-by-step reasoning
        if reasoning_response.get("steps"):
            response_parts.append("**Step-by-Step Reasoning:**")
            for i, step in enumerate(reasoning_response["steps"], 1):
                response_parts.append(f"{i}. {step}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _generate_follow_up_questions(self, query: StudentQuery) -> List[str]:
        """Generate relevant follow-up questions"""
        # Simple rule-based follow-up generation
        follow_ups = []
        
        if query.query_type == QueryType.MCQ_SOLVING:
            follow_ups = [
                "Would you like me to explain any of the incorrect options?",
                "Do you want to see similar questions on this topic?",
                "Would you like to review the underlying concept?"
            ]
        elif query.query_type == QueryType.CONCEPT_EXPLANATION:
            follow_ups = [
                "Would you like to see some practice questions on this topic?",
                "Do you want me to explain related concepts?",
                "Would you like to know common mistakes students make here?"
            ]
        
        return follow_ups[:2]  # Limit to 2 follow-ups
    
    def _get_recent_context(self) -> List[Dict]:
        """Get recent conversation context"""
        return self.conversation_history[-3:] if self.conversation_history else []
    
    def _estimate_student_level(self) -> str:
        """Estimate student's current level based on interaction history"""
        # Simple heuristic - can be enhanced with ML
        if len(self.conversation_history) < 5:
            return "beginner"
        
        # Analyze recent performance
        recent_interactions = self.conversation_history[-10:]
        concept_queries = sum(1 for interaction in recent_interactions 
                            if interaction["type"] == "concept")
        
        if concept_queries > 6:
            return "beginner"
        elif concept_queries > 3:
            return "intermediate"
        else:
            return "advanced"
    
    def _analyze_student_performance(self) -> Dict:
        """Analyze student's performance patterns"""
        return {
            "total_interactions": len(self.conversation_history),
            "subject_distribution": self._get_subject_distribution(),
            "query_type_distribution": self._get_query_type_distribution(),
            "weak_areas": self.student_profile["weak_areas"],
            "strong_areas": self.student_profile["strong_areas"]
        }
    
    def _get_subject_distribution(self) -> Dict[str, int]:
        """Get distribution of queries by subject"""
        distribution = {"physics": 0, "chemistry": 0, "biology": 0, "general": 0}
        for interaction in self.conversation_history:
            subject = interaction.get("subject", "general")
            if subject in distribution:
                distribution[subject] += 1
            else:
                distribution["general"] += 1
        return distribution
    
    def _get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types"""
        distribution = {}
        for interaction in self.conversation_history:
            query_type = interaction.get("type", "general")
            distribution[query_type] = distribution.get(query_type, 0) + 1
        return distribution
    
    def update_student_profile(self, feedback: Dict):
        """Update student profile based on feedback and performance"""
        if "weak_areas" in feedback:
            self.student_profile["weak_areas"].extend(feedback["weak_areas"])
            # Remove duplicates
            self.student_profile["weak_areas"] = list(set(self.student_profile["weak_areas"]))
        
        if "strong_areas" in feedback:
            self.student_profile["strong_areas"].extend(feedback["strong_areas"])
            self.student_profile["strong_areas"] = list(set(self.student_profile["strong_areas"]))
        
        logger.info("Student profile updated")
    
    async def initialize_knowledge_base(self):
        """Initialize the knowledge base with NCERT books and MCQs"""
        logger.info("Initializing knowledge base...")
        
        # Load NCERT books
        await self.ncert_loader.load_all_books()
        
        # Load MCQ datasets
        await self.mcq_parser.load_mcq_datasets()
        
        logger.info("Knowledge base initialization complete")
    
    def get_student_dashboard(self) -> Dict:
        """Get student performance dashboard data"""
        return {
            "performance_analysis": self._analyze_student_performance(),
            "recent_topics": self._get_recent_topics(),
            "recommended_topics": self._get_recommended_topics(),
            "study_streak": self._calculate_study_streak(),
            "weak_areas": self.student_profile["weak_areas"],
            "strong_areas": self.student_profile["strong_areas"]
        }
    
    def _get_recent_topics(self) -> List[str]:
        """Get recently studied topics"""
        recent_topics = []
        for interaction in self.conversation_history[-10:]:
            if interaction.get("subject"):
                recent_topics.append(interaction["subject"])
        return list(set(recent_topics))
    
    def _get_recommended_topics(self) -> List[str]:
        """Get recommended topics based on weak areas"""
        # Simple recommendation based on weak areas
        recommendations = []
        for weak_area in self.student_profile["weak_areas"]:
            # Add related topics (this would be enhanced with proper topic modeling)
            recommendations.append(f"Practice more {weak_area}")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _calculate_study_streak(self) -> int:
        """Calculate current study streak in days"""
        # Simple implementation - would need proper date tracking
        return len(self.conversation_history) // 5  # Rough approximation

# Example usage and testing
async def main():
    """Example usage of the NEET Coach Agent"""
    coach = NEETCoachAgent()
    
    # Initialize knowledge base (this would be done once)
    # await coach.initialize_knowledge_base()
    
    # Example interactions
    test_queries = [
        "What is Newton's first law of motion?",
        "Solve this MCQ: Which of the following is correct about photosynthesis? a) It occurs only in sunlight b) It produces glucose c) It requires CO2 d) All of the above",
        "I'm confused about organic chemistry reactions. Can you help?",
        "What's the best strategy to prepare for NEET in 6 months?"
    ]
    
    for query in test_queries:
        print(f"\n🧑‍🎓 Student: {query}")
        response = await coach.process_query(query)
        print(f"🤖 AI Coach:\n{response.content}")
        
        if response.tricks_detected:
            print(f"⚠️ Tricks detected: {', '.join(response.tricks_detected)}")
        
        if response.follow_up_questions:
            print(f"💡 Follow-up questions: {', '.join(response.follow_up_questions)}")
        
        print("-" * 80)
    
    # Show student dashboard
    dashboard = coach.get_student_dashboard()
    print(f"\n📊 Student Dashboard: {json.dumps(dashboard, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())