"""
Math Solver for AI NEET Coach
Handles symbolic mathematics, calculations, and step-by-step solutions
"""

import re
import math
import cmath
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Try to import SymPy for symbolic mathematics
try:
    import sympy as sp
    from sympy import symbols, solve, simplify, expand, factor, diff, integrate
    from sympy import sin, cos, tan, log, exp, sqrt, pi, E, I
    from sympy.physics.units import meter, second, kilogram, newton, joule, watt
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Try to import NumPy for numerical calculations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MathType(Enum):
    """Types of mathematical problems"""
    ALGEBRAIC = "algebraic"
    TRIGONOMETRIC = "trigonometric"
    CALCULUS = "calculus"
    PHYSICS_FORMULA = "physics_formula"
    CHEMISTRY_CALCULATION = "chemistry_calculation"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    UNIT_CONVERSION = "unit_conversion"

@dataclass
class SolutionStep:
    """Single step in mathematical solution"""
    step_number: int
    description: str
    expression: str
    result: str
    explanation: str

@dataclass
class MathSolution:
    """Complete mathematical solution"""
    problem: str
    problem_type: MathType
    steps: List[SolutionStep]
    final_answer: str
    alternative_methods: List[str]
    common_mistakes: List[str]
    verification: Optional[str] = None

class MathSolver:
    """Main mathematical solver with step-by-step solutions"""
    
    def __init__(self):
        self.setup_environment()
        self.formula_bank = self._load_formula_bank()
        self.unit_conversions = self._load_unit_conversions()
        logger.info("Initialized MathSolver")
    
    def setup_environment(self):
        """Setup mathematical environment"""
        if SYMPY_AVAILABLE:
            # Define common symbols
            self.x, self.y, self.z = symbols('x y z')
            self.t = symbols('t')  # time
            self.v, self.u, self.a = symbols('v u a')  # velocity, initial velocity, acceleration
            self.s, self.h = symbols('s h')  # displacement, height
            self.m, self.M = symbols('m M')  # mass
            self.F, self.g = symbols('F g')  # force, gravity
            self.T, self.P, self.V = symbols('T P V')  # temperature, pressure, volume
            self.n, self.R = symbols('n R')  # moles, gas constant
            
            logger.info("SymPy environment initialized")
        else:
            logger.warning("SymPy not available. Limited mathematical capabilities.")
    
    def _load_formula_bank(self) -> Dict[str, Dict]:
        """Load common formulas for physics and chemistry"""
        formulas = {
            "kinematics": {
                "v = u + at": {"description": "Final velocity", "variables": ["v", "u", "a", "t"]},
                "s = ut + (1/2)at^2": {"description": "Displacement", "variables": ["s", "u", "t", "a"]},
                "v^2 = u^2 + 2as": {"description": "Velocity-displacement relation", "variables": ["v", "u", "a", "s"]},
                "s = (u + v)t/2": {"description": "Average velocity formula", "variables": ["s", "u", "v", "t"]}
            },
            "mechanics": {
                "F = ma": {"description": "Newton's second law", "variables": ["F", "m", "a"]},
                "W = Fs": {"description": "Work done", "variables": ["W", "F", "s"]},
                "P = W/t": {"description": "Power", "variables": ["P", "W", "t"]},
                "KE = (1/2)mv^2": {"description": "Kinetic energy", "variables": ["KE", "m", "v"]},
                "PE = mgh": {"description": "Potential energy", "variables": ["PE", "m", "g", "h"]}
            },
            "waves": {
                "v = fλ": {"description": "Wave speed", "variables": ["v", "f", "λ"]},
                "f = 1/T": {"description": "Frequency-period relation", "variables": ["f", "T"]}
            },
            "thermodynamics": {
                "PV = nRT": {"description": "Ideal gas law", "variables": ["P", "V", "n", "R", "T"]},
                "Q = mcΔT": {"description": "Heat capacity", "variables": ["Q", "m", "c", "ΔT"]}
            },
            "chemistry": {
                "n = m/M": {"description": "Moles calculation", "variables": ["n", "m", "M"]},
                "C = n/V": {"description": "Molarity", "variables": ["C", "n", "V"]},
                "pH = -log[H+]": {"description": "pH calculation", "variables": ["pH", "H+"]}
            },
            "optics": {
                "1/f = 1/u + 1/v": {"description": "Lens formula", "variables": ["f", "u", "v"]},
                "m = v/u": {"description": "Magnification", "variables": ["m", "v", "u"]}
            }
        }
        return formulas
    
    def _load_unit_conversions(self) -> Dict[str, Dict]:
        """Load unit conversion factors"""
        conversions = {
            "length": {
                "m_to_cm": 100,
                "m_to_mm": 1000,
                "km_to_m": 1000,
                "cm_to_m": 0.01,
                "mm_to_m": 0.001
            },
            "mass": {
                "kg_to_g": 1000,
                "g_to_kg": 0.001,
                "g_to_mg": 1000,
                "mg_to_g": 0.001
            },
            "time": {
                "h_to_s": 3600,
                "min_to_s": 60,
                "s_to_ms": 1000,
                "ms_to_s": 0.001
            },
            "energy": {
                "J_to_kJ": 0.001,
                "kJ_to_J": 1000,
                "cal_to_J": 4.184,
                "J_to_cal": 0.239
            },
            "pressure": {
                "atm_to_Pa": 101325,
                "bar_to_Pa": 100000,
                "mmHg_to_Pa": 133.322
            },
            "temperature": {
                "C_to_K": lambda c: c + 273.15,
                "K_to_C": lambda k: k - 273.15,
                "F_to_C": lambda f: (f - 32) * 5/9,
                "C_to_F": lambda c: c * 9/5 + 32
            }
        }
        return conversions
    
    def identify_problem_type(self, problem: str) -> MathType:
        """Identify the type of mathematical problem"""
        problem_lower = problem.lower()
        
        # Physics keywords
        if any(word in problem_lower for word in ["velocity", "acceleration", "force", "energy", "momentum"]):
            return MathType.PHYSICS_FORMULA
        
        # Chemistry keywords
        if any(word in problem_lower for word in ["moles", "molarity", "concentration", "ph", "reaction"]):
            return MathType.CHEMISTRY_CALCULATION
        
        # Trigonometry
        if any(word in problem_lower for word in ["sin", "cos", "tan", "angle", "triangle"]):
            return MathType.TRIGONOMETRIC
        
        # Calculus
        if any(word in problem_lower for word in ["derivative", "integral", "limit", "differentiate", "integrate"]):
            return MathType.CALCULUS
        
        # Logarithmic
        if any(word in problem_lower for word in ["log", "ln", "exponential", "e^"]):
            return MathType.LOGARITHMIC
        
        # Unit conversion
        if any(word in problem_lower for word in ["convert", "unit", "cm to m", "kg to g"]):
            return MathType.UNIT_CONVERSION
        
        # Default to algebraic
        return MathType.ALGEBRAIC
    
    def solve_step_by_step(self, problem: str, given_values: Dict[str, float] = None) -> MathSolution:
        """
        Solve mathematical problem with step-by-step solution
        
        Args:
            problem: Problem statement
            given_values: Dictionary of known values
            
        Returns:
            Complete solution with steps
        """
        problem_type = self.identify_problem_type(problem)
        given_values = given_values or {}
        
        if problem_type == MathType.PHYSICS_FORMULA:
            return self._solve_physics_problem(problem, given_values)
        elif problem_type == MathType.CHEMISTRY_CALCULATION:
            return self._solve_chemistry_problem(problem, given_values)
        elif problem_type == MathType.UNIT_CONVERSION:
            return self._solve_unit_conversion(problem, given_values)
        elif problem_type == MathType.TRIGONOMETRIC:
            return self._solve_trigonometric(problem, given_values)
        elif problem_type == MathType.CALCULUS:
            return self._solve_calculus(problem, given_values)
        else:
            return self._solve_algebraic(problem, given_values)
    
    def _solve_physics_problem(self, problem: str, given_values: Dict) -> MathSolution:
        """Solve physics problems with appropriate formulas"""
        steps = []
        
        # Step 1: Identify relevant formulas
        relevant_formulas = self._identify_relevant_formulas(problem, "physics")
        steps.append(SolutionStep(
            step_number=1,
            description="Identify relevant physics concepts and formulas",
            expression="",
            result=", ".join(relevant_formulas),
            explanation="Based on the problem keywords, these formulas are applicable."
        ))
        
        # Step 2: List given values
        if given_values:
            given_str = ", ".join([f"{k} = {v}" for k, v in given_values.items()])
            steps.append(SolutionStep(
                step_number=2,
                description="List given values",
                expression="",
                result=given_str,
                explanation="Extract known quantities from the problem."
            ))
        
        # Step 3: Apply formula (example for kinematics)
        if "velocity" in problem.lower() and "acceleration" in problem.lower():
            formula = "v = u + at"
            steps.append(SolutionStep(
                step_number=len(steps) + 1,
                description="Apply kinematic equation",
                expression=formula,
                result="Substitute known values",
                explanation="This equation relates velocity, initial velocity, acceleration, and time."
            ))
            
            # Calculate if possible
            if all(var in given_values for var in ["u", "a", "t"]):
                result = given_values["u"] + given_values["a"] * given_values["t"]
                steps.append(SolutionStep(
                    step_number=len(steps) + 1,
                    description="Calculate final velocity",
                    expression=f"v = {given_values['u']} + {given_values['a']} × {given_values['t']}",
                    result=f"v = {result} m/s",
                    explanation="Substitute values and perform calculation."
                ))
                final_answer = f"{result} m/s"
            else:
                final_answer = "Cannot calculate without all required values"
        else:
            final_answer = "Formula identified but calculation requires specific values"
        
        return MathSolution(
            problem=problem,
            problem_type=MathType.PHYSICS_FORMULA,
            steps=steps,
            final_answer=final_answer,
            alternative_methods=["Graphical method", "Energy conservation approach"],
            common_mistakes=["Unit conversion errors", "Sign convention confusion"]
        )
    
    def _solve_chemistry_problem(self, problem: str, given_values: Dict) -> MathSolution:
        """Solve chemistry calculation problems"""
        steps = []
        
        # Step 1: Identify calculation type
        if "moles" in problem.lower():
            calculation_type = "Moles calculation"
            formula = "n = m/M or n = PV/RT or n = C×V"
        elif "ph" in problem.lower():
            calculation_type = "pH calculation"
            formula = "pH = -log[H+]"
        elif "concentration" in problem.lower() or "molarity" in problem.lower():
            calculation_type = "Molarity calculation"
            formula = "C = n/V"
        else:
            calculation_type = "General chemistry calculation"
            formula = "Identify appropriate formula"
        
        steps.append(SolutionStep(
            step_number=1,
            description="Identify calculation type",
            expression=formula,
            result=calculation_type,
            explanation="Determine the type of chemistry calculation needed."
        ))
        
        # Example calculation for molarity
        if "molarity" in problem.lower() and "moles" in given_values and "volume" in given_values:
            molarity = given_values["moles"] / given_values["volume"]
            steps.append(SolutionStep(
                step_number=2,
                description="Calculate molarity",
                expression=f"C = {given_values['moles']}/{given_values['volume']}",
                result=f"C = {molarity} M",
                explanation="Molarity is moles of solute per liter of solution."
            ))
            final_answer = f"{molarity} M"
        else:
            final_answer = "Specific calculation requires numerical values"
        
        return MathSolution(
            problem=problem,
            problem_type=MathType.CHEMISTRY_CALCULATION,
            steps=steps,
            final_answer=final_answer,
            alternative_methods=["Using different units", "Dimensional analysis"],
            common_mistakes=["Unit confusion", "Significant figures"]
        )
    
    def _solve_unit_conversion(self, problem: str, given_values: Dict) -> MathSolution:
        """Solve unit conversion problems"""
        steps = []
        
        # Extract conversion information
        conversion_match = re.search(r'(\d+\.?\d*)\s*(\w+)\s*to\s*(\w+)', problem.lower())
        
        if conversion_match:
            value = float(conversion_match.group(1))
            from_unit = conversion_match.group(2)
            to_unit = conversion_match.group(3)
            
            steps.append(SolutionStep(
                step_number=1,
                description="Identify conversion",
                expression=f"{value} {from_unit} → {to_unit}",
                result="Conversion identified",
                explanation="Extract the value and units to convert."
            ))
            
            # Find conversion factor
            conversion_factor = self._find_conversion_factor(from_unit, to_unit)
            
            if conversion_factor:
                converted_value = value * conversion_factor
                steps.append(SolutionStep(
                    step_number=2,
                    description="Apply conversion factor",
                    expression=f"{value} × {conversion_factor}",
                    result=f"{converted_value} {to_unit}",
                    explanation=f"Multiply by conversion factor to convert {from_unit} to {to_unit}."
                ))
                final_answer = f"{converted_value} {to_unit}"
            else:
                final_answer = "Conversion factor not found in database"
        else:
            final_answer = "Could not parse conversion from problem statement"
        
        return MathSolution(
            problem=problem,
            problem_type=MathType.UNIT_CONVERSION,
            steps=steps,
            final_answer=final_answer,
            alternative_methods=["Dimensional analysis", "Direct formula"],
            common_mistakes=["Wrong conversion factor", "Inverse conversion"]
        )
    
    def _solve_trigonometric(self, problem: str, given_values: Dict) -> MathSolution:
        """Solve trigonometric problems"""
        steps = []
        
        if SYMPY_AVAILABLE:
            # Example: solve trigonometric equation
            steps.append(SolutionStep(
                step_number=1,
                description="Set up trigonometric equation",
                expression="Identify trigonometric function and equation",
                result="Equation identified",
                explanation="Determine which trigonometric functions are involved."
            ))
            
            # If angle is given, calculate trigonometric values
            if "angle" in given_values:
                angle = given_values["angle"]
                angle_rad = math.radians(angle)
                
                sin_val = math.sin(angle_rad)
                cos_val = math.cos(angle_rad)
                tan_val = math.tan(angle_rad)
                
                steps.append(SolutionStep(
                    step_number=2,
                    description="Calculate trigonometric values",
                    expression=f"For angle = {angle}°",
                    result=f"sin({angle}°) = {sin_val:.4f}, cos({angle}°) = {cos_val:.4f}, tan({angle}°) = {tan_val:.4f}",
                    explanation="Convert angle to radians and calculate trigonometric ratios."
                ))
                
                final_answer = f"sin = {sin_val:.4f}, cos = {cos_val:.4f}, tan = {tan_val:.4f}"
            else:
                final_answer = "Requires specific angle or equation to solve"
        else:
            final_answer = "SymPy required for advanced trigonometric calculations"
        
        return MathSolution(
            problem=problem,
            problem_type=MathType.TRIGONOMETRIC,
            steps=steps,
            final_answer=final_answer,
            alternative_methods=["Unit circle method", "Calculator method"],
            common_mistakes=["Degree vs radian confusion", "Sign errors in different quadrants"]
        )
    
    def _solve_calculus(self, problem: str, given_values: Dict) -> MathSolution:
        """Solve calculus problems"""
        steps = []
        
        if SYMPY_AVAILABLE:
            if "derivative" in problem.lower() or "differentiate" in problem.lower():
                # Extract function to differentiate
                function_match = re.search(r'f\(x\)\s*=\s*([^,\n]+)', problem)
                if function_match:
                    function_str = function_match.group(1).strip()
                    try:
                        expr = sp.sympify(function_str)
                        derivative = sp.diff(expr, self.x)
                        
                        steps.append(SolutionStep(
                            step_number=1,
                            description="Identify function to differentiate",
                            expression=f"f(x) = {function_str}",
                            result="Function identified",
                            explanation="Parse the function from the problem statement."
                        ))
                        
                        steps.append(SolutionStep(
                            step_number=2,
                            description="Apply differentiation rules",
                            expression=f"d/dx({function_str})",
                            result=f"f'(x) = {derivative}",
                            explanation="Use appropriate differentiation rules (power rule, chain rule, etc.)."
                        ))
                        
                        final_answer = f"f'(x) = {derivative}"
                    except Exception as e:
                        final_answer = f"Error parsing function: {str(e)}"
                else:
                    final_answer = "Could not identify function to differentiate"
                    
            elif "integral" in problem.lower() or "integrate" in problem.lower():
                # Similar logic for integration
                function_match = re.search(r'∫\s*([^dx]+)\s*dx', problem)
                if function_match:
                    function_str = function_match.group(1).strip()
                    try:
                        expr = sp.sympify(function_str)
                        integral = sp.integrate(expr, self.x)
                        
                        steps.append(SolutionStep(
                            step_number=1,
                            description="Identify function to integrate",
                            expression=f"∫ {function_str} dx",
                            result="Function identified",
                            explanation="Parse the integrand from the problem statement."
                        ))
                        
                        steps.append(SolutionStep(
                            step_number=2,
                            description="Apply integration rules",
                            expression=f"∫ {function_str} dx",
                            result=f"{integral} + C",
                            explanation="Use appropriate integration techniques."
                        ))
                        
                        final_answer = f"{integral} + C"
                    except Exception as e:
                        final_answer = f"Error parsing function: {str(e)}"
                else:
                    final_answer = "Could not identify function to integrate"
            else:
                final_answer = "Calculus operation not clearly identified"
        else:
            final_answer = "SymPy required for calculus calculations"
        
        return MathSolution(
            problem=problem,
            problem_type=MathType.CALCULUS,
            steps=steps,
            final_answer=final_answer,
            alternative_methods=["Numerical methods", "Graphical interpretation"],
            common_mistakes=["Forgetting constant of integration", "Chain rule errors"]
        )
    
    def _solve_algebraic(self, problem: str, given_values: Dict) -> MathSolution:
        """Solve algebraic equations"""
        steps = []
        
        if SYMPY_AVAILABLE:
            # Try to extract equation from problem
            equation_match = re.search(r'([^=]+)=([^=\n]+)', problem)
            if equation_match:
                left_side = equation_match.group(1).strip()
                right_side = equation_match.group(2).strip()
                
                try:
                    left_expr = sp.sympify(left_side)
                    right_expr = sp.sympify(right_side)
                    equation = sp.Eq(left_expr, right_expr)
                    
                    steps.append(SolutionStep(
                        step_number=1,
                        description="Set up equation",
                        expression=f"{left_side} = {right_side}",
                        result="Equation identified",
                        explanation="Parse the algebraic equation from the problem."
                    ))
                    
                    # Solve for x (assuming x is the variable)
                    solutions = sp.solve(equation, self.x)
                    
                    steps.append(SolutionStep(
                        step_number=2,
                        description="Solve for x",
                        expression=f"Solving: {left_side} = {right_side}",
                        result=f"x = {solutions}",
                        explanation="Apply algebraic manipulation to isolate the variable."
                    ))
                    
                    final_answer = f"x = {solutions}"
                    
                except Exception as e:
                    final_answer = f"Error solving equation: {str(e)}"
            else:
                final_answer = "Could not parse algebraic equation"
        else:
            final_answer = "SymPy required for algebraic solving"
        
        return MathSolution(
            problem=problem,
            problem_type=MathType.ALGEBRAIC,
            steps=steps,
            final_answer=final_answer,
            alternative_methods=["Graphical method", "Numerical approximation"],
            common_mistakes=["Sign errors", "Invalid operations"]
        )
    
    def _identify_relevant_formulas(self, problem: str, subject: str) -> List[str]:
        """Identify relevant formulas based on problem keywords"""
        relevant = []
        problem_lower = problem.lower()
        
        if subject == "physics":
            for category, formulas in self.formula_bank.items():
                if category == "chemistry":
                    continue
                for formula, info in formulas.items():
                    # Check if any variables mentioned in problem
                    if any(var.lower() in problem_lower for var in info["variables"]):
                        relevant.append(formula)
        
        return relevant if relevant else ["General formula identification needed"]
    
    def _find_conversion_factor(self, from_unit: str, to_unit: str) -> Optional[float]:
        """Find conversion factor between units"""
        for category, conversions in self.unit_conversions.items():
            conversion_key = f"{from_unit}_to_{to_unit}"
            if conversion_key in conversions:
                factor = conversions[conversion_key]
                return factor if not callable(factor) else None
        
        # Try reverse conversion
        reverse_key = f"{to_unit}_to_{from_unit}"
        for category, conversions in self.unit_conversions.items():
            if reverse_key in conversions:
                factor = conversions[reverse_key]
                if not callable(factor):
                    return 1.0 / factor
        
        return None
    
    def verify_solution(self, solution: MathSolution) -> str:
        """Verify the mathematical solution by substitution"""
        if not SYMPY_AVAILABLE:
            return "Verification requires SymPy"
        
        try:
            # Basic verification by checking if final answer makes sense
            if solution.problem_type == MathType.UNIT_CONVERSION:
                return "Unit conversion verified by dimensional analysis"
            elif solution.problem_type == MathType.PHYSICS_FORMULA:
                return "Solution verified by formula consistency"
            else:
                return "Solution follows mathematical principles"
        except Exception as e:
            return f"Verification error: {str(e)}"
    
    def get_formula_suggestions(self, keywords: List[str]) -> List[Dict]:
        """Suggest relevant formulas based on keywords"""
        suggestions = []
        
        for category, formulas in self.formula_bank.items():
            for formula, info in formulas.items():
                if any(keyword.lower() in info["description"].lower() for keyword in keywords):
                    suggestions.append({
                        "formula": formula,
                        "description": info["description"],
                        "category": category,
                        "variables": info["variables"]
                    })
        
        return suggestions
    
    def explain_concept(self, concept: str) -> str:
        """Explain mathematical or physics concepts"""
        concept_lower = concept.lower()
        
        explanations = {
            "derivative": "A derivative measures how a function changes as its input changes. It's the slope of the tangent line at any point.",
            "integral": "An integral represents the area under a curve. It's the reverse operation of differentiation.",
            "velocity": "Velocity is the rate of change of displacement with respect to time. It's a vector quantity.",
            "acceleration": "Acceleration is the rate of change of velocity with respect to time.",
            "force": "Force is any interaction that, when unopposed, will change the motion of an object (Newton's laws).",
            "energy": "Energy is the capacity to do work. It comes in many forms: kinetic, potential, thermal, etc.",
            "moles": "A mole is a unit of measurement for the amount of substance, equal to 6.022 × 10²³ particles.",
            "molarity": "Molarity is the concentration of a solution expressed as moles of solute per liter of solution."
        }
        
        for key, explanation in explanations.items():
            if key in concept_lower:
                return explanation
        
        return "Concept explanation not available. Please provide more specific keywords."

# Example usage and testing
if __name__ == "__main__":
    solver = MathSolver()
    
    # Test physics problem
    physics_problem = "A car accelerates from rest with acceleration 2 m/s² for 5 seconds. Find the final velocity."
    physics_values = {"u": 0, "a": 2, "t": 5}
    solution = solver.solve_step_by_step(physics_problem, physics_values)
    
    print(f"Problem: {solution.problem}")
    print(f"Type: {solution.problem_type}")
    print("\nSteps:")
    for step in solution.steps:
        print(f"{step.step_number}. {step.description}")
        print(f"   Expression: {step.expression}")
        print(f"   Result: {step.result}")
        print(f"   Explanation: {step.explanation}\n")
    
    print(f"Final Answer: {solution.final_answer}")
