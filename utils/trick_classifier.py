"""
Trick Classifier for AI NEET Coach
Detects common traps, tricks, and misleading patterns in competitive exam questions
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TrickType(Enum):
    """Types of tricks commonly found in competitive exams"""
    UNIT_CONFUSION = "unit_confusion"
    SIGN_REVERSAL = "sign_reversal"
    MAGNITUDE_ORDER = "magnitude_order"
    WORD_PLAY = "word_play"
    EXCEPTION_CASE = "exception_case"
    INCOMPLETE_INFO = "incomplete_info"
    DISTRACTOR_OPTIONS = "distractor_options"
    SIMILAR_CONCEPTS = "similar_concepts"
    CALCULATION_TRAP = "calculation_trap"
    CONCEPTUAL_REVERSAL = "conceptual_reversal"
    TIME_CONTEXT = "time_context"
    CONDITION_REVERSAL = "condition_reversal"

@dataclass
class TrickPattern:
    """Structure for trick patterns"""
    trick_type: TrickType
    pattern: str
    description: str
    subject_specific: List[str]
    confidence_keywords: List[str]
    warning_message: str

@dataclass
class TrickDetection:
    """Result of trick detection"""
    trick_type: TrickType
    confidence: float
    matched_patterns: List[str]
    warning_message: str
    suggested_approach: str

class TrickClassifier:
    """Main trick detection and classification system"""
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _has_complex_calculation(self, question: str) -> bool:
        """Check if question involves complex calculations"""
        calculation_indicators = [
            r'\d+\.\d+',  # Decimal numbers
            r'\d+/\d+',   # Fractions
            r'\^',        # Powers
            r'√',         # Square roots
            r'log',       # Logarithms
            r'sin|cos|tan', # Trigonometry
            r'\d+\s*×\s*10\^', # Scientific notation
        ]
        
        for pattern in calculation_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        
        return False
    
    def _deduplicate_detections(self, detections: List[TrickDetection]) -> List[TrickDetection]:
        """Remove duplicate detections of same trick type"""
        seen_types = set()
        unique_detections = []
        
        for detection in detections:
            if detection.trick_type not in seen_types:
                unique_detections.append(detection)
                seen_types.add(detection.trick_type)
            else:
                # If duplicate, keep the one with higher confidence
                for i, existing in enumerate(unique_detections):
                    if existing.trick_type == detection.trick_type:
                        if detection.confidence > existing.confidence:
                            unique_detections[i] = detection
                        break
        
        return unique_detections
    
    def _get_suggested_approach(self, trick_type: TrickType) -> str:
        """Get suggested approach for handling specific trick types"""
        approaches = {
            TrickType.UNIT_CONFUSION: "Convert all quantities to the same unit system before calculating.",
            TrickType.SIGN_REVERSAL: "Draw a diagram showing directions. Establish a consistent sign convention.",
            TrickType.MAGNITUDE_ORDER: "Write numbers in scientific notation and compare exponents.",
            TrickType.WORD_PLAY: "Rephrase the question in your own words to clarify what's being asked.",
            TrickType.EXCEPTION_CASE: "Think of counterexamples or special cases that might invalidate absolute statements.",
            TrickType.INCOMPLETE_INFO: "List all given information and check if anything is missing for the solution.",
            TrickType.DISTRACTOR_OPTIONS: "Compare options systematically, highlighting key differences.",
            TrickType.SIMILAR_CONCEPTS: "Write down the precise definitions of similar concepts to distinguish them.",
            TrickType.CALCULATION_TRAP: "Break the calculation into smaller steps and verify each step.",
            TrickType.CONCEPTUAL_REVERSAL: "Understand the exact meaning of each term in the context.",
            TrickType.TIME_CONTEXT: "Identify the specific time point or interval the question refers to.",
            TrickType.CONDITION_REVERSAL: "Note the exact conditions (temperature, pressure, etc.) specified."
        }
        
        return approaches.get(trick_type, "Read the question carefully and think step by step.")
    
    def analyze_question_difficulty(self, question: str, options: List[str], subject: str) -> Dict[str, Any]:
        """Analyze overall question difficulty and trick potential"""
        detections = self.detect_tricks(question, options, subject)
        
        # Calculate overall trick score
        trick_score = sum(detection.confidence for detection in detections)
        
        # Determine difficulty level
        if trick_score >= 2.0:
            difficulty = "Very High"
        elif trick_score >= 1.5:
            difficulty = "High"
        elif trick_score >= 1.0:
            difficulty = "Medium"
        elif trick_score >= 0.5:
            difficulty = "Low"
        else:
            difficulty = "Very Low"
        
        return {
            "trick_score": trick_score,
            "difficulty": difficulty,
            "detected_tricks": len(detections),
            "primary_tricks": [d.trick_type.value for d in detections[:3]],
            "warnings": [d.warning_message for d in detections],
            "approaches": [d.suggested_approach for d in detections]
        }
    
    def generate_student_guidance(self, question: str, options: List[str], subject: str) -> str:
        """Generate comprehensive guidance for students"""
        analysis = self.analyze_question_difficulty(question, options, subject)
        detections = self.detect_tricks(question, options, subject)
        
        guidance = []
        
        if analysis["trick_score"] > 0.5:
            guidance.append(f"🎯 **Trick Alert!** This question has a trick difficulty of {analysis['difficulty']}")
            guidance.append("")
        
        if detections:
            guidance.append("⚠️ **Potential Traps:**")
            for detection in detections:
                guidance.append(f"• {detection.warning_message}")
            guidance.append("")
            
            guidance.append("💡 **Suggested Approach:**")
            for i, detection in enumerate(detections, 1):
                guidance.append(f"{i}. {detection.suggested_approach}")
            guidance.append("")
        
        # General advice based on subject
        subject_advice = {
            "physics": [
                "✓ Check units and convert if necessary",
                "✓ Draw diagrams for vector quantities",
                "✓ Identify the physical principle involved"
            ],
            "chemistry": [
                "✓ Balance chemical equations",
                "✓ Check molecular vs empirical formulas",
                "✓ Verify reaction conditions (temperature, pressure)"
            ],
            "biology": [
                "✓ Look for exceptions to general rules",
                "✓ Distinguish between similar biological processes",
                "✓ Check if the statement applies to all organisms"
            ]
        }
        
        if subject.lower() in subject_advice:
            guidance.append(f"📚 **{subject.title()} Tips:**")
            guidance.extend(subject_advice[subject.lower()])
        
        return "\n".join(guidance)

# Extended patterns for more comprehensive detection
class AdvancedTrickDetector:
    """Advanced trick detection with machine learning-like pattern recognition"""
    
    def __init__(self):
        self.classifier = TrickClassifier()
        self.question_patterns = self._load_question_patterns()
    
    def _load_question_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that indicate specific types of trick questions"""
        return {
            "distractor_heavy": [
                "which of the following is true",
                "which statement is correct",
                "identify the correct option",
                "all of the above except"
            ],
            "calculation_intensive": [
                "calculate the",
                "find the value",
                "determine the",
                "compute the"
            ],
            "conceptual_deep": [
                "why does",
                "what happens when",
                "explain the reason",
                "what is the principle"
            ],
            "exception_based": [
                "always true except",
                "never occurs except",
                "all are correct except",
                "which cannot"
            ]
        }
    
    def predict_trick_probability(self, question: str, options: List[str], subject: str) -> float:
        """Predict probability of question containing tricks"""
        detections = self.classifier.detect_tricks(question, options, subject)
        
        # Base probability from detections
        base_prob = min(sum(d.confidence for d in detections) / 3.0, 1.0)
        
        # Adjust based on question patterns
        pattern_boost = 0.0
        question_lower = question.lower()
        
        for pattern_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    pattern_boost += 0.1
        
        # Adjust based on option similarity
        option_similarity = self._calculate_option_similarity(options)
        similarity_boost = option_similarity * 0.3
        
        final_prob = min(base_prob + pattern_boost + similarity_boost, 1.0)
        return final_prob
    
    def _calculate_option_similarity(self, options: List[str]) -> float:
        """Calculate average similarity between options"""
        if len(options) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(options)):
            for j in range(i + 1, len(options)):
                similarity = self.classifier._calculate_similarity(options[i], options[j])
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

# Example usage and testing
if __name__ == "__main__":
    # Initialize the classifier
    classifier = TrickClassifier()
    advanced_detector = AdvancedTrickDetector()
    
    # Test questions
    test_questions = [
        {
            "question": "A body is moving with constant velocity. The net force acting on it is:",
            "options": ["10 N", "5 N", "0 N", "Cannot be determined"],
            "subject": "physics"
        },
        {
            "question": "Which of the following statements about DNA is NOT correct?",
            "options": [
                "DNA contains genetic information",
                "DNA is double-stranded",
                "DNA contains ribose sugar",
                "DNA has four types of bases"
            ],
            "subject": "biology"
        },
        {
            "question": "At STP, 22.4 L of gas contains how many moles?",
            "options": ["0.5 mol", "1 mol", "2 mol", "22.4 mol"],
            "subject": "chemistry"
        }
    ]
    
    # Test each question
    for i, test_q in enumerate(test_questions, 1):
        print(f"\n{'='*50}")
        print(f"TEST QUESTION {i}")
        print(f"{'='*50}")
        print(f"Question: {test_q['question']}")
        print(f"Options: {test_q['options']}")
        print(f"Subject: {test_q['subject']}")
        
        # Detect tricks
        detections = classifier.detect_tricks(
            test_q['question'], 
            test_q['options'], 
            test_q['subject']
        )
        
        print(f"\nDetected {len(detections)} potential tricks:")
        for detection in detections:
            print(f"• {detection.trick_type.value} (confidence: {detection.confidence:.2f})")
            print(f"  Warning: {detection.warning_message}")
            print(f"  Approach: {detection.suggested_approach}")
        
        # Analysis
        analysis = classifier.analyze_question_difficulty(
            test_q['question'], 
            test_q['options'], 
            test_q['subject']
        )
        print(f"\nDifficulty: {analysis['difficulty']} (score: {analysis['trick_score']:.2f})")
        
        # Trick probability
        prob = advanced_detector.predict_trick_probability(
            test_q['question'], 
            test_q['options'], 
            test_q['subject']
        )
        print(f"Trick Probability: {prob:.2f}")
        
        # Student guidance
        guidance = classifier.generate_student_guidance(
            test_q['question'], 
            test_q['options'], 
            test_q['subject']
        )
        print(f"\nStudent Guidance:\n{guidance}")
_init__(self):
        self.patterns = self._load_trick_patterns()
        self.subject_specific_patterns = self._load_subject_patterns()
        logger.info(f"Initialized TrickClassifier with {len(self.patterns)} patterns")
    
    def _load_trick_patterns(self) -> List[TrickPattern]:
        """Load common trick patterns"""
        patterns = [
            # Unit confusion tricks
            TrickPattern(
                trick_type=TrickType.UNIT_CONFUSION,
                pattern=r'\b(cm|mm|km|mg|kg|g|ml|l|°C|°F|K)\b.*\b(cm|mm|km|mg|kg|g|ml|l|°C|°F|K)\b',
                description="Mixed units in question/options",
                subject_specific=["physics", "chemistry"],
                confidence_keywords=["convert", "unit", "SI", "CGS", "measurement"],
                warning_message="⚠️ Check units carefully! Options may have different units."
            ),
            
            # Sign reversal tricks
            TrickPattern(
                trick_type=TrickType.SIGN_REVERSAL,
                pattern=r'\b(positive|negative|opposite|reverse|inverse)\b',
                description="Questions involving sign changes",
                subject_specific=["physics", "chemistry"],
                confidence_keywords=["charge", "direction", "acceleration", "velocity", "force"],
                warning_message="⚠️ Pay attention to signs! Consider direction/polarity carefully."
            ),
            
            # Magnitude/order of magnitude tricks
            TrickPattern(
                trick_type=TrickType.MAGNITUDE_ORDER,
                pattern=r'\b(10\^|×10|x10|order of magnitude|scientific notation)\b',
                description="Scientific notation or order of magnitude confusion",
                subject_specific=["physics", "chemistry"],
                confidence_keywords=["large", "small", "micro", "nano", "mega", "kilo"],
                warning_message="⚠️ Check powers of 10! Verify the order of magnitude."
            ),
            
            # Word play tricks
            TrickPattern(
                trick_type=TrickType.WORD_PLAY,
                pattern=r'\b(not|except|incorrect|false|cannot|impossible)\b',
                description="Negative phrasing or word tricks",
                subject_specific=["physics", "chemistry", "biology"],
                confidence_keywords=["which", "following", "statement", "option"],
                warning_message="⚠️ Read carefully! Question uses negative phrasing."
            ),
            
            # Exception cases
            TrickPattern(
                trick_type=TrickType.EXCEPTION_CASE,
                pattern=r'\b(always|never|all|none|every|only|except|but)\b',
                description="Absolute statements with exceptions",
                subject_specific=["biology", "chemistry", "physics"],
                confidence_keywords=["rule", "law", "principle", "property"],
                warning_message="⚠️ Look for exceptions! Absolute statements often have special cases."
            ),
            
            # Incomplete information
            TrickPattern(
                trick_type=TrickType.INCOMPLETE_INFO,
                pattern=r'\b(assume|given|provided|additional information needed)\b',
                description="Questions with missing or assumed information",
                subject_specific=["physics", "chemistry"],
                confidence_keywords=["calculate", "find", "determine", "solve"],
                warning_message="⚠️ Check if all required information is given!"
            ),
            
            # Similar concepts confusion
            TrickPattern(
                trick_type=TrickType.SIMILAR_CONCEPTS,
                pattern=r'\b(similar|alike|resemble|confuse|distinguish)\b',
                description="Questions testing similar but different concepts",
                subject_specific=["biology", "chemistry", "physics"],
                confidence_keywords=["difference", "between", "compare", "contrast"],
                warning_message="⚠️ Don't confuse similar concepts! Read definitions carefully."
            ),
            
            # Time context tricks
            TrickPattern(
                trick_type=TrickType.TIME_CONTEXT,
                pattern=r'\b(initial|final|before|after|during|instantaneous|average)\b',
                description="Time-dependent context changes",
                subject_specific=["physics", "chemistry", "biology"],
                confidence_keywords=["time", "rate", "speed", "velocity", "acceleration"],
                warning_message="⚠️ Check the time context! Initial vs final conditions matter."
            )
        ]
        return patterns
    
    def _load_subject_patterns(self) -> Dict[str, List[TrickPattern]]:
        """Load subject-specific trick patterns"""
        subject_patterns = {
            "physics": [
                TrickPattern(
                    trick_type=TrickType.CONCEPTUAL_REVERSAL,
                    pattern=r'\b(work done by|work done against|work done on)\b',
                    description="Work direction confusion",
                    subject_specific=["physics"],
                    confidence_keywords=["work", "energy", "force", "displacement"],
                    warning_message="⚠️ Check work direction! By vs against vs on makes a difference."
                ),
                TrickPattern(
                    trick_type=TrickType.CALCULATION_TRAP,
                    pattern=r'\b(maximum|minimum|range|projectile)\b',
                    description="Projectile motion calculation traps",
                    subject_specific=["physics"],
                    confidence_keywords=["angle", "velocity", "height", "distance"],
                    warning_message="⚠️ Projectile motion: Check angle conditions for max range/height."
                )
            ],
            
            "chemistry": [
                TrickPattern(
                    trick_type=TrickType.CONDITION_REVERSAL,
                    pattern=r'\b(STP|NTP|standard|normal|room temperature)\b',
                    description="Standard condition confusion",
                    subject_specific=["chemistry"],
                    confidence_keywords=["pressure", "temperature", "volume", "mole"],
                    warning_message="⚠️ Check conditions! STP vs NTP vs room temperature differ."
                ),
                TrickPattern(
                    trick_type=TrickType.SIMILAR_CONCEPTS,
                    pattern=r'\b(molecular|empirical|structural|formula)\b',
                    description="Formula type confusion",
                    subject_specific=["chemistry"],
                    confidence_keywords=["formula", "compound", "molecule"],
                    warning_message="⚠️ Distinguish formula types! Molecular ≠ Empirical ≠ Structural."
                )
            ],
            
            "biology": [
                TrickPattern(
                    trick_type=TrickType.EXCEPTION_CASE,
                    pattern=r'\b(all plants|all animals|always|never|only)\b',
                    description="Biological absolute statements",
                    subject_specific=["biology"],
                    confidence_keywords=["plants", "animals", "cells", "organs"],
                    warning_message="⚠️ Biology has exceptions! Avoid absolute generalizations."
                ),
                TrickPattern(
                    trick_type=TrickType.SIMILAR_CONCEPTS,
                    pattern=r'\b(DNA|RNA|protein|enzyme|hormone)\b',
                    description="Biomolecule confusion",
                    subject_specific=["biology"],
                    confidence_keywords=["function", "structure", "synthesis", "metabolism"],
                    warning_message="⚠️ Don't mix biomolecules! Each has specific functions."
                )
            ]
        }
        return subject_patterns
    
    def detect_tricks(self, question: str, options: List[str], subject: str = "general") -> List[TrickDetection]:
        """
        Detect potential tricks in a question
        
        Args:
            question: The question text
            options: List of answer options
            subject: Subject area (physics, chemistry, biology)
            
        Returns:
            List of detected tricks with confidence scores
        """
        detections = []
        full_text = f"{question} {' '.join(options)}"
        
        # Check general patterns
        for pattern in self.patterns:
            detection = self._check_pattern(pattern, full_text, question, options)
            if detection:
                detections.append(detection)
        
        # Check subject-specific patterns
        if subject.lower() in self.subject_specific_patterns:
            for pattern in self.subject_specific_patterns[subject.lower()]:
                detection = self._check_pattern(pattern, full_text, question, options)
                if detection:
                    detections.append(detection)
        
        # Additional heuristic checks
        detections.extend(self._heuristic_checks(question, options, subject))
        
        # Sort by confidence and remove duplicates
        detections = self._deduplicate_detections(detections)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def _check_pattern(self, pattern: TrickPattern, full_text: str, question: str, options: List[str]) -> Optional[TrickDetection]:
        """Check if a specific pattern matches"""
        matches = re.findall(pattern.pattern, full_text, re.IGNORECASE)
        
        if not matches:
            return None
        
        # Calculate confidence based on matches and keywords
        confidence = len(matches) * 0.3
        
        for keyword in pattern.confidence_keywords:
            if keyword.lower() in full_text.lower():
                confidence += 0.2
        
        confidence = min(confidence, 1.0)
        
        if confidence < 0.3:  # Minimum threshold
            return None
        
        return TrickDetection(
            trick_type=pattern.trick_type,
            confidence=confidence,
            matched_patterns=[str(match) for match in matches],
            warning_message=pattern.warning_message,
            suggested_approach=self._get_suggested_approach(pattern.trick_type)
        )
    
    def _heuristic_checks(self, question: str, options: List[str], subject: str) -> List[TrickDetection]:
        """Additional heuristic-based trick detection"""
        detections = []
        
        # Check for very similar options (distractor trap)
        similar_options = self._find_similar_options(options)
        if similar_options:
            detections.append(TrickDetection(
                trick_type=TrickType.DISTRACTOR_OPTIONS,
                confidence=0.7,
                matched_patterns=similar_options,
                warning_message="⚠️ Multiple similar options! Read carefully to spot differences.",
                suggested_approach="Compare options systematically, looking for key differences."
            ))
        
        # Check for calculation complexity
        if self._has_complex_calculation(question):
            detections.append(TrickDetection(
                trick_type=TrickType.CALCULATION_TRAP,
                confidence=0.6,
                matched_patterns=["complex calculation"],
                warning_message="⚠️ Complex calculation! Double-check arithmetic and units.",
                suggested_approach="Break down calculation into steps, verify each step."
            ))
        
        # Check for negative phrasing
        negative_words = ["not", "except", "incorrect", "false", "cannot", "never", "none"]
        if any(word in question.lower() for word in negative_words):
            detections.append(TrickDetection(
                trick_type=TrickType.WORD_PLAY,
                confidence=0.8,
                matched_patterns=["negative phrasing"],
                warning_message="⚠️ Negative question! Look for what does NOT apply.",
                suggested_approach="Identify what the question is asking NOT to find."
            ))
        
        return detections
    
    def _find_similar_options(self, options: List[str]) -> List[str]:
        """Find suspiciously similar options"""
        similar_pairs = []
        
        for i, opt1 in enumerate(options):
            for j, opt2 in enumerate(options[i+1:], i+1):
                # Simple similarity check (can be improved with better algorithms)
                similarity = self._calculate_similarity(opt1, opt2)
                if similarity > 0.7:  # High similarity threshold
                    similar_pairs.append(f"{opt1} ≈ {opt2}")
        
        return similar_pairs
    
    def _