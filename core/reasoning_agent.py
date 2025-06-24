"""
Reasoning Agent - The core intelligence of the NEET Coach
Handles complex reasoning, MCQ solving, and concept explanation
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    step_number: int
    description: str
    confidence: float
    evidence: List[str]

class ReasoningAgent:
    """
    Advanced reasoning agent that can:
    1. Solve MCQs with step-by-step reasoning
    2. Explain concepts at appropriate levels
    3. Provide Socratic questioning
    4. Detect reasoning patterns and mistakes
    """
    
    def __init__(self, llm_model: str, vector_store):
        self.llm_model = llm_model
        self.vector_store = vector_store
        self.reasoning_templates = self._load_reasoning_templates()
        self.concept_hierarchy = self._build_concept_hierarchy()
        
    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load reasoning templates for different question types"""
        return {
            "mcq_template": """
            You are an expert NEET tutor. Analyze this MCQ step by step:
            
            Question: {question}
            Context: {context}
            Detected Tricks: {tricks}
            
            Provide:
            1. Step-by-step analysis
            2. Correct answer with reasoning
            3. Why other options are wrong
            4. Key concepts tested
            5. Common mistakes students make
            
            Be thorough but clear. Use examples when helpful.
            """,
            
            "concept_template": """
            You are an expert NEET tutor explaining concepts to a {level} student.
            
            Topic: {concept}
            Context from NCERT: {context}
            
            Provide:
            1. Clear, simple explanation
            2. Real-world examples
            3. Connection to other concepts
            4. Common misconceptions
            5. Practice questions suggestions
            
            Adjust complexity for {level} level. Use analogies and examples.
            """,
            
            "doubt_template": """
            You are helping a student clarify their doubt. Be patient and thorough.
            
            Student's Doubt: {doubt}
            Previous Context: {context}
            Student's Weak Areas: {weaknesses}
            
            Provide:
            1. Direct answer to the doubt
            2. Simplified explanation
            3. Examples to illustrate
            4. Related concepts to review
            5. Encouraging words
            
            Be empathetic and build confidence.
            """,
            
            "strategy_template": """
            You are a NEET preparation strategist. Provide actionable advice.
            
            Student Query: {query}
            Performance Data: {performance}
            Current Profile: {profile}
            
            Provide:
            1. Specific strategy recommendations
            2. Time allocation suggestions
            3. Priority topics to focus on
            4. Study techniques for weak areas
            5. Motivational guidance
            
            Be practical and realistic.
            """
        }
    
    def _build_concept_hierarchy(self) -> Dict[str, Dict]:
        """Build a hierarchy of concepts for better reasoning"""
        return {
            "physics": {
                "mechanics": ["motion", "force", "energy", "momentum"],
                "thermodynamics": ["heat", "temperature", "entropy", "cycles"],
                "optics": ["reflection", "refraction", "interference", "diffraction"],
                "electricity": ["current", "voltage", "resistance", "circuits"],
                "modern_physics": ["quantum", "relativity", "atomic", "nuclear"]
            },
            "chemistry": {
                "physical": ["thermodynamics", "kinetics", "equilibrium", "solutions"],
                "organic": ["hydrocarbons", "functional_groups", "reactions", "biomolecules"],
                "inorganic": ["periodic_table", "coordination", "metallurgy", "qualitative"]
            },
            "biology": {
                "diversity": ["classification", "morphology", "anatomy", "evolution"],
                "genetics": ["heredity", "molecular_biology", "biotechnology"],
                "ecology": ["ecosystem", "environment", "biodiversity"],
                "physiology": ["plant", "animal", "human"]
            }
        }
    
    async def solve_mcq(self, question: str, context_docs: List[Dict], 
                       detected_tricks: List[str]) -> Dict[str, Any]:
        """
        Solve MCQ with comprehensive reasoning
        """
        logger.info("Starting MCQ analysis...")
        
        # Extract options from question
        options = self._extract_options(question)
        
        # Analyze each option
        option_analysis = await self._analyze_options(question, options, context_docs)
        
        # Apply reasoning chain
        reasoning_chain = await self._build_reasoning_chain(
            question, options, option_analysis, context_docs, detected_tricks
        )
        
        # Determine final answer
        final_answer = self._determine_final_answer(reasoning_chain, option_analysis)
        
        # Generate explanation
        explanation = self._generate_explanation(
            question, final_answer, reasoning_chain, detected_tricks
        )
        
        return {
            "answer": final_answer,
            "explanation": explanation,
            "reasoning_chain": reasoning_chain,
            "option_analysis": option_analysis,
            "confidence": self._calculate_confidence(reasoning_chain),
            "key_concepts": self._extract_key_concepts(question, context_docs),
            "steps": [step.description for step in reasoning_chain]
        }
    
    async def explain_concept(self, concept: str, context_docs: List[Dict], 
                            student_level: str) -> Dict[str, Any]:
        """
        Provide comprehensive concept explanation
        """
        logger.info(f"Explaining concept: {concept}")
        
        # Identify concept category and related topics
        concept_info = self._analyze_concept(concept)
        
        # Build explanation structure
        explanation_structure = await self._build_explanation_structure(
            concept, concept_info, context_docs, student_level
        )
        
        # Generate main explanation
        main_explanation = self._generate_concept_explanation(
            concept, explanation_structure, student_level
        )
        
        # Add examples and analogies
        examples = self._generate_examples(concept, concept_info, student_level)
        
        # Generate follow-up questions
        follow_ups = self._generate_concept_follow_ups(concept, concept_info)
        
        return {
            "content": main_explanation,
            "examples": examples,
            "explanation_flow": explanation_structure,
            "follow_up_questions": follow_ups,
            "confidence": 0.85,
            "related_concepts": concept_info.get("related", [])
        }
    
    async def clarify_doubt(self, doubt: str, context_docs: List[Dict],
                          conversation_context: List[Dict], 
                          student_weaknesses: List[str]) -> Dict[str, Any]:
        """
        Clarify student doubts with personalized approach
        """
        logger.info(f"Clarifying doubt: {doubt[:50]}...")
        
        # Analyze the doubt
        doubt_analysis = self._analyze_doubt(doubt, conversation_context)
        
        # Generate clarification strategy
        clarification_strategy = self._generate_clarification_strategy(
            doubt_analysis, student_weaknesses
        )
        
        # Build clarification response
        clarification_content = self._build_clarification_response(
            doubt, doubt_analysis, clarification_strategy, context_docs
        )
        
        return {
            "content": clarification_content,
            "clarification_steps": clarification_strategy,
            "confidence": 0.8,
            "follow_up_questions": self._generate_doubt_follow_ups(doubt_analysis)
        }
    
    async def generate_strategy(self, query: str, performance_data: Dict,
                              student_profile: Dict) -> Dict[str, Any]:
        """
        Generate personalized study strategy
        """
        logger.info("Generating study strategy...")
        
        # Analyze current performance
        performance_analysis = self._analyze_performance_gaps(performance_data)
        
        # Identify priority areas
        priority_areas = self._identify_priority_areas(
            performance_analysis, student_profile
        )
        
        # Generate strategy recommendations
        strategy_content = self._generate_strategy_recommendations(
            query, priority_areas, performance_analysis
        )
        
        return {
            "content": strategy_content,
            "reasoning": [
                f"Performance analysis: {performance_analysis}",
                f"Priority areas identified: {priority_areas}",
                "Strategy tailored to student's current level and goals"
            ],
            "confidence": 0.9,
            "follow_up_questions": [
                "Would you like a detailed study schedule?",
                "Do you need specific resource recommendations?",
                "Would you like tips for time management?"
            ]
        }
    
    async def general_response(self, query: str, conversation_history: List[Dict],
                             student_profile: Dict) -> Dict[str, Any]:
        """
        Handle general conversation with contextual awareness
        """
        # Analyze conversation context
        context_analysis = self._analyze_conversation_context(conversation_history)
        
        # Generate contextual response
        response_content = self._generate_contextual_response(
            query, context_analysis, student_profile
        )
        
        return {
            "content": response_content,
            "confidence": 0.7,
            "follow_up_questions": self._generate_general_follow_ups(query, context_analysis)
        }
    
    def _extract_options(self, question: str) -> Dict[str, str]:
        """Extract options from MCQ question"""
        options = {}
        option_pattern = r'([a-d])\)\s*([^a-d\)]+?)(?=[a-d]\)|$)'
        
        matches = re.findall(option_pattern, question, re.IGNORECASE | re.DOTALL)
        for match in matches:
            option_key = match[0].lower()
            option_text = match[1].strip()
            options[option_key] = option_text
        
        return options
    
    async def _analyze_options(self, question: str, options: Dict[str, str],
                             context_docs: List[Dict]) -> Dict[str, Dict]:
        """Analyze each MCQ option for correctness"""
        analysis = {}
        
        for option_key, option_text in options.items():
            # Analyze option against context
            relevance_score = self._calculate_option_relevance(option_text, context_docs)
            correctness_indicators = self._find_correctness_indicators(
                question, option_text, context_docs
            )
            
            analysis[option_key] = {
                "text": option_text,
                "relevance_score": relevance_score,
                "correctness_indicators": correctness_indicators,
                "likely_correct": relevance_score > 0.7,
                "reasoning": self._generate_option_reasoning(option_text, correctness_indicators)
            }
        
        return analysis
    
    async def _build_reasoning_chain(self, question: str, options: Dict[str, str],
                                   option_analysis: Dict, context_docs: List[Dict],
                                   detected_tricks: List[str]) -> List[ReasoningStep]:
        """Build a chain of reasoning steps"""
        reasoning_chain = []
        
        # Step 1: Understand the question
        reasoning_chain.append(ReasoningStep(
            step_number=1,
            description=f"Understanding the question: {self._extract_question_core(question)}",
            confidence=0.9,
            evidence=[f"Question asks about: {self._identify_question_focus(question)}"]
        ))
        
        # Step 2: Identify key concepts
        key_concepts = self._extract_key_concepts(question, context_docs)
        reasoning_chain.append(ReasoningStep(
            step_number=2,
            description=f"Key concepts involved: {', '.join(key_concepts)}",
            confidence=0.85,
            evidence=[f"Found in context: {concept}" for concept in key_concepts]
        ))
        
        # Step 3: Apply knowledge from context
        relevant_facts = self._extract_relevant_facts(question, context_docs)
        reasoning_chain.append(ReasoningStep(
            step_number=3,
            description="Applying relevant knowledge from NCERT",
            confidence=0.8,
            evidence=relevant_facts[:3]  # Top 3 most relevant facts
        ))
        
        # Step 4: Evaluate each option
        for option_key, analysis in option_analysis.items():
            reasoning_chain.append(ReasoningStep(
                step_number=len(reasoning_chain) + 1,
                description=f"Option {option_key.upper()}: {analysis['reasoning']}",
                confidence=analysis['relevance_score'],
                evidence=analysis['correctness_indicators']
            ))
        
        # Step 5: Consider tricks and common mistakes
        if detected_tricks:
            reasoning_chain.append(ReasoningStep(
                step_number=len(reasoning_chain) + 1,
                description=f"Checking for tricks: {', '.join(detected_tricks)}",
                confidence=0.7,
                evidence=[f"Trick detected: {trick}" for trick in detected_tricks]
            ))
        
        return reasoning_chain
    
    def _determine_final_answer(self, reasoning_chain: List[ReasoningStep],
                              option_analysis: Dict) -> str:
        """Determine the final answer based on reasoning"""
        # Score each option based on analysis
        option_scores = {}
        for option_key, analysis in option_analysis.items():
            score = analysis['relevance_score']
            # Boost score if multiple reasoning steps support it
            supporting_steps = [step for step in reasoning_chain 
                              if option_key in step.description.lower()]
            score += len(supporting_steps) * 0.1
            option_scores[option_key] = score
        
        # Return the highest scoring option
        best_option = max(option_scores.items(), key=lambda x: x[1])
        return best_option[0].upper()
    
    def _generate_explanation(self, question: str, final_answer: str,
                            reasoning_chain: List[ReasoningStep],
                            detected_tricks: List[str]) -> str:
        """Generate comprehensive explanation"""
        explanation_parts = []
        
        # Start with the answer
        explanation_parts.append(f"The correct answer is option {final_answer}.")
        
        # Add main reasoning
        key_reasoning = [step for step in reasoning_chain 
                        if step.confidence > 0.7][:3]
        
        explanation_parts.append("\n**Key Reasoning:**")
        for step in key_reasoning:
            explanation_parts.append(f"• {step.description}")
        
        # Add concept explanation
        explanation_parts.append(f"\n**Concept Review:**")
        explanation_parts.append(self._generate_concept_review(question))
        
        # Add trick warnings if any
        if detected_tricks:
            explanation_parts.append(f"\n**Important Notes:**")
            for trick in detected_tricks:
                explanation_parts.append(f"• Watch out for: {trick}")
        
        return "\n".join(explanation_parts)
    
    def _calculate_confidence(self, reasoning_chain: List[ReasoningStep]) -> float:
        """Calculate overall confidence in the answer"""
        if not reasoning_chain:
            return 0.5
        
        # Average confidence of high-confidence steps
        high_conf_steps = [step for step in reasoning_chain if step.confidence > 0.6]
        if high_conf_steps:
            return sum(step.confidence for step in high_conf_steps) / len(high_conf_steps)
        else:
            return 0.5
    
    def _extract_key_concepts(self, question: str, context_docs: List[Dict]) -> List[str]:
        """Extract key concepts from question and context"""
        concepts = []
        
        # Extract from question
        question_lower = question.lower()
        for subject, categories in self.concept_hierarchy.items():
            for category, concept_list in categories.items():
                for concept in concept_list:
                    if concept in question_lower:
                        concepts.append(concept)
        
        # Extract from context documents
        for doc in context_docs[:3]:  # Top 3 most relevant docs
            content = doc.get('content', '').lower()
            for subject, categories in self.concept_hierarchy.items():
                for category, concept_list in categories.items():
                    for concept in concept_list:
                        if concept in content and concept not in concepts:
                            concepts.append(concept)
        
        return concepts[:5]  # Return top 5 concepts
    
    def _analyze_concept(self, concept: str) -> Dict[str, Any]:
        """Analyze concept to understand its context and relationships"""
        concept_lower = concept.lower()
        concept_info = {
            "subject": None,
            "category": None,
            "related": [],
            "difficulty": "intermediate"
        }
        
        # Find subject and category
        for subject, categories in self.concept_hierarchy.items():
            for category, concept_list in categories.items():
                if any(c in concept_lower for c in concept_list):
                    concept_info["subject"] = subject
                    concept_info["category"] = category
                    concept_info["related"] = [c for c in concept_list if c != concept_lower]
                    break
        
        return concept_info
    
    def _build_explanation_structure(self, concept: str, concept_info: Dict,
                                   context_docs: List[Dict], student_level: str) -> List[str]:
        """Build structure for concept explanation"""
        structure = []
        
        if student_level == "beginner":
            structure = [
                "Basic definition and introduction",
                "Simple examples and analogies",
                "Key points to remember",
                "Common misconceptions",
                "Practice suggestions"
            ]
        elif student_level == "intermediate":
            structure = [
                "Detailed explanation",
                "Mathematical relationships (if applicable)",
                "Real-world applications",
                "Connection to other concepts",
                "Typical exam questions"
            ]
        else:  # advanced
            structure = [
                "Comprehensive analysis",
                "Advanced applications",
                "Recent developments",
                "Complex problem-solving approaches",
                "Research connections"
            ]
        
        return structure
    
    def _generate_concept_explanation(self, concept: str, structure: List[str],
                                    student_level: str) -> str:
        """Generate the main concept explanation"""
        explanation_parts = []
        
        explanation_parts.append(f"# {concept.title()}\n")
        
        # Generate content for each structure element
        for element in structure:
            explanation_parts.append(f"## {element}")
            explanation_parts.append(self._generate_section_content(concept, element, student_level))
            explanation_parts.append("")
        
        return "\n".join(explanation_parts)
    
    def _generate_section_content(self, concept: str, section: str, level: str) -> str:
        """Generate content for a specific section"""
        # This would integrate with your LLM API
        # For now, providing template responses
        
        section_templates = {
            "Basic definition and introduction": f"Let me explain {concept} in simple terms...",
            "Simple examples and analogies": f"Think of {concept} like...",
            "Key points to remember": f"The most important things about {concept} are...",
            "Mathematical relationships (if applicable)": f"The mathematical expression for {concept} is...",
            "Real-world applications": f"You can see {concept} in everyday life when...",
            "Connection to other concepts": f"{concept} is related to other topics such as..."
        }
        
        return section_templates.get(section, f"Content about {concept} - {section}")
    
    def _generate_examples(self, concept: str, concept_info: Dict, student_level: str) -> List[str]:
        """Generate examples for the concept"""
        examples = []
        
        if concept_info["subject"] == "physics":
            examples = [
                f"Example 1: Simple demonstration of {concept}",
                f"Example 2: Real-world application of {concept}",
                f"Example 3: Problem-solving with {concept}"
            ]
        elif concept_info["subject"] == "chemistry":
            examples = [
                f"Example 1: Chemical reaction involving {concept}",
                f"Example 2: Laboratory observation of {concept}",
                f"Example 3: Industrial application of {concept}"
            ]
        elif concept_info["subject"] == "biology":
            examples = [
                f"Example 1: {concept} in living organisms",
                f"Example 2: {concept} in human body",
                f"Example 3: {concept} in ecosystem"
            ]
        
        return examples[:2 if student_level == "beginner" else 3]
    
    def _generate_concept_follow_ups(self, concept: str, concept_info: Dict) -> List[str]:
        """Generate follow-up questions for concept"""
        follow_ups = [
            f"Would you like to see practice problems on {concept}?",
            f"Should I explain how {concept} relates to {concept_info.get('category', 'other topics')}?",
            f"Do you want to know common exam questions about {concept}?"
        ]
        
        return follow_ups[:2]
    
    # Helper methods for various calculations and analysis
    def _calculate_option_relevance(self, option_text: str, context_docs: List[Dict]) -> float:
        """Calculate how relevant an option is based on context"""
        relevance_score = 0.0
        option_words = set(option_text.lower().split())
        
        for doc in context_docs:
            doc_words = set(doc.get('content', '').lower().split())
            overlap = len(option_words.intersection(doc_words))
            relevance_score += overlap / max(len(option_words), 1)
        
        return min(relevance_score / max(len(context_docs), 1), 1.0)
    
    def _find_correctness_indicators(self, question: str, option_text: str,
                                   context_docs: List[Dict]) -> List[str]:
        """Find indicators that suggest an option is correct"""
        indicators = []
        
        # Check for exact matches in context
        for doc in context_docs:
            if option_text.lower() in doc.get('content', '').lower():
                indicators.append(f"Exact match found in {doc.get('source', 'context')}")
        
        # Check for keyword alignment
        question_keywords = self._extract_keywords(question)
        option_keywords = self._extract_keywords(option_text)
        
        common_keywords = set(question_keywords).intersection(set(option_keywords))
        if common_keywords:
            indicators.append(f"Keywords align: {', '.join(common_keywords)}")
        
        return indicators
    
    def _generate_option_reasoning(self, option_text: str, indicators: List[str]) -> str:
        """Generate reasoning for why an option might be correct/incorrect"""
        if indicators:
            return f"This option is supported by: {'; '.join(indicators)}"
        else:
            return "This option lacks strong supporting evidence from the context"
    
    def _extract_question_core(self, question: str) -> str:
        """Extract the core of what the question is asking"""
        # Simple extraction - look for question words
        question_words = ["what", "which", "how", "why", "when", "where"]
        sentences = question.split('.')
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_words):
                return sentence.strip()
        
        return question.split('.')[0]  # Return first sentence if no question words found
    
    def _identify_question_focus(self, question: str) -> str:
        """Identify what the question is focusing on"""
        focus_indicators = {
            "definition": ["what is", "define", "meaning"],
            "process": ["how does", "process", "mechanism"],
            "comparison": ["difference", "compare", "unlike"],
            "application": ["application", "used for", "example"],
            "calculation": ["calculate", "find", "determine"]
        }
        
        question_lower = question.lower()
        for focus, indicators in focus_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return focus
        
        return "general knowledge"
    
    def _extract_relevant_facts(self, question: str, context_docs: List[Dict]) -> List[str]:
        """Extract relevant facts from context documents"""
        facts = []
        question_keywords = self._extract_keywords(question)
        
        for doc in context_docs:
            content = doc.get('content', '')
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Ignore very short sentences
                    sentence_keywords = self._extract_keywords(sentence)
                    overlap = set(question_keywords).intersection(set(sentence_keywords))
                    if len(overlap) >= 2:  # At least 2 keywords in common
                        facts.append(sentence)
        
        return facts[:5]  # Return top 5 most relevant facts
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        import re
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                     "of", "with", "by", "is", "are", "was", "were", "be", "been", "being"}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        return keywords[:10]  # Return top 10 keywords
    
    def _generate_concept_review(self, question: str) -> str:
        """Generate a concept review based on the question"""
        # This would be enhanced with actual concept knowledge
        return f"This question tests your understanding of key concepts. Review the fundamental principles and their applications."
    
    def _analyze_doubt(self, doubt: str, conversation_context: List[Dict]) -> Dict[str, Any]:
        """Analyze student's doubt to understand the root issue"""
        return {
            "doubt_type": self._classify_doubt_type(doubt),
            "complexity_level": self._assess_doubt_complexity(doubt),
            "related_topics": self._find_related_topics(doubt),
            "context_clues": self._extract_context_clues(conversation_context)
        }
    
    def _classify_doubt_type(self, doubt: str) -> str:
        """Classify the type of doubt"""
        doubt_lower = doubt.lower()
        
        if any(word in doubt_lower for word in ["confused", "don't understand", "unclear"]):
            return "conceptual_confusion"
        elif any(word in doubt_lower for word in ["why", "how", "what"]):
            return "seeking_explanation"
        elif any(word in doubt_lower for word in ["difference", "compare"]):
            return "comparison_doubt"
        elif any(word in doubt_lower for word in ["solve", "calculate", "find"]):
            return "problem_solving"
        else:
            return "general_inquiry"
    
    def _assess_doubt_complexity(self, doubt: str) -> str:
        """Assess the complexity level of the doubt"""
        # Simple heuristic based on question complexity
        if len(doubt.split()) < 10:
            return "simple"
        elif len(doubt.split()) < 20:
            return "moderate"
        else:
            return "complex"
    
    def _find_related_topics(self, doubt: str) -> List[str]:
        """Find topics related to the doubt"""
        related = []
        doubt_lower = doubt.lower()
        
        for subject, categories in self.concept_hierarchy.items():
            for category, concepts in categories.items():
                for concept in concepts:
                    if concept in doubt_lower:
                        related.extend([c for c in concepts if c != concept])
        
        return list(set(related))[:3]
    
    def _extract_context_clues(self, conversation_context: List[Dict]) -> List[str]:
        """Extract clues from conversation context"""
        clues = []
        for interaction in conversation_context:
            if interaction.get("type") == "concept":
                clues.append(f"Recently discussed: {interaction.get('subject', 'unknown topic')}")
        
        return clues
    
    def _generate_clarification_strategy(self, doubt_analysis: Dict,
                                       student_weaknesses: List[str]) -> List[str]:
        """Generate strategy for clarifying the doubt"""
        strategy = []
        
        doubt_type = doubt_analysis.get("doubt_type", "general_inquiry")
        
        if doubt_type == "conceptual_confusion":
            strategy = [
                "Start with basics and build up",
                "Use simple analogies",
                "Provide visual examples",
                "Check understanding at each step"
            ]
        elif doubt_type == "seeking_explanation":
            strategy = [
                "Provide clear, step-by-step explanation",
                "Use examples and applications",
                "Connect to previously learned concepts",
                "Summarize key points"
            ]
        elif doubt_type == "problem_solving":
            strategy = [
                "Break down the problem",
                "Identify relevant concepts",
                "Show solution methodology",
                "Practice with similar problems"
            ]
        
        return strategy
    
    def _build_clarification_response(self, doubt: str, doubt_analysis: Dict,
                                    strategy: List[str], context_docs: List[Dict]) -> str:
        """Build the clarification response"""
        response_parts = []
        
        response_parts.append(f"I understand your doubt about: {doubt}\n")
        response_parts.append("Let me help clarify this step by step:\n")
        
        # Generate response based on strategy
        for i, strategy_step in enumerate(strategy, 1):
            response_parts.append(f"{i}. {strategy_step}")
            response_parts.append(self._generate_strategy_content(strategy_step, doubt, context_docs))
            response_parts.append("")
        
        response_parts.append("I hope this helps! Feel free to ask if you need more clarification.")
        
        return "\n".join(response_parts)
    
    def _generate_strategy_content(self, strategy_step: str, doubt: str,
                                 context_docs: List[Dict]) -> str:
        """Generate content for each strategy step"""
        # Template responses - would be enhanced with actual LLM integration
        content_templates = {
            "Start with basics and build up": "Let's begin with the fundamental concept...",
            "Use simple analogies": "Think of this like...",
            "Provide visual examples": "Imagine this scenario...",
            "Provide clear, step-by-step explanation": "Here's how we can understand this:",
            "Break down the problem": "Let's break this problem into parts:"
        }
        
        return content_templates.get(strategy_step, f"Content for: {strategy_step}")
    
    def _generate_doubt_follow_ups(self, doubt_analysis: Dict) -> List[str]:
        """Generate follow-up questions for doubt clarification"""
        follow_ups = [
            "Does this explanation make sense to you?",
            "Would you like me to explain any part in more detail?",
            "Do you have any related questions?"
        ]
        
        doubt_type = doubt_analysis.get("doubt_type")
        if doubt_type == "problem_solving":
            follow_ups.append("Would you like to try a similar problem?")
        elif doubt_type == "conceptual_confusion":
            follow_ups.append("Should I explain the concept from a different angle?")
        
        return follow_ups[:2]
    
    def _analyze_performance_gaps(self, performance_data: Dict) -> Dict[str, Any]:
        """Analyze performance to identify gaps"""
        gaps = {
            "subject_gaps": {},
            "concept_gaps": [],
            "overall_weakness": "intermediate"
        }
        
        subject_dist = performance_data.get("subject_distribution", {})
        total_interactions = sum(subject_dist.values())
        
        if total_interactions > 0:
            for subject, count in subject_dist.items():
                if count / total_interactions < 0.2:  # Less than 20% focus
                    gaps["subject_gaps"][subject] = "needs_attention"
        
        return gaps
    
    def _identify_priority_areas(self, performance_analysis: Dict,
                               student_profile: Dict) -> List[str]:
        """Identify priority areas for improvement"""
        priorities = []
        
        # Add weak areas from profile
        priorities.extend(student_profile.get("weak_areas", []))
        
        # Add gaps from performance analysis
        subject_gaps = performance_analysis.get("subject_gaps", {})
        priorities.extend(subject_gaps.keys())
        
        return list(set(priorities))[:5]  # Top 5 priorities
    
    def _generate_strategy_recommendations(self, query: str, priority_areas: List[str],
                                         performance_analysis: Dict) -> str:
        """Generate strategy recommendations"""
        recommendations = []
        
        recommendations.append("# Personalized Study Strategy\n")
        
        if priority_areas:
            recommendations.append("## Priority Areas to Focus On:")
            for area in priority_areas:
                recommendations.append(f"• {area.title()}: Needs immediate attention")
            recommendations.append("")
        
        recommendations.append("## Recommended Study Plan:")
        recommendations.append("• Daily practice: 2-3 hours focused study")
        recommendations.append("• Weekly review: Revisit completed topics")
        recommendations.append("• Monthly assessment: Take mock tests")
        recommendations.append("")
        
        recommendations.append("## Study Techniques:")
        recommendations.append("• Active recall: Test yourself regularly")
        recommendations.append("• Spaced repetition: Review at increasing intervals")
        recommendations.append("• Practice problems: Solve MCQs daily")
        
        return "\n".join(recommendations)
    
    def _analyze_conversation_context(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze conversation context for general responses"""
        context = {
            "recent_topics": [],
            "interaction_pattern": "normal",
            "engagement_level": "moderate"
        }
        
        for interaction in conversation_history:
            if interaction.get("type"):
                context["recent_topics"].append(interaction["type"])
        
        return context
    
    def _generate_contextual_response(self, query: str, context_analysis: Dict,
                                    student_profile: Dict) -> str:
        """Generate contextual response for general chat"""
        # Simple contextual response generation
        recent_topics = context_analysis.get("recent_topics", [])
        
        if "mcq" in recent_topics:
            return "I see you've been working on MCQs. That's great practice! How can I help you further?"
        elif "concept" in recent_topics:
            return "You've been exploring some interesting concepts. What would you like to learn about next?"
        else:
            return "I'm here to help with your NEET preparation. What can I assist you with today?"
    
    def _generate_general_follow_ups(self, query: str, context_analysis: Dict) -> List[str]:
        """Generate follow-up questions for general conversation"""
        return [
            "Is there a specific topic you'd like to focus on?",
            "How can I best support your NEET preparation today?"
        ]