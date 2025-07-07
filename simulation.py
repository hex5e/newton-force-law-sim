"""
Spring-Mass Simulation
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ============= Interfaces and Protocols =============

class Force(ABC):
    """Abstract base class for forces (OCP - extensible)"""
    @abstractmethod
    def calculate(self, system_state: 'SystemState') -> np.ndarray:
        """Calculate forces on all masses"""
        pass


class Integrator(ABC):
    """Abstract base class for integration methods (OCP)"""
    @abstractmethod
    def step(self, system_state: 'SystemState', forces: np.ndarray, dt: float) -> 'SystemState':
        """Perform one integration step"""
        pass


class CollisionDetector(ABC):
    """Abstract base class for collision detection (OCP)"""
    @abstractmethod
    def detect_collisions(self, system_state: 'SystemState') -> List['Collision']:
        """Detect all collisions in the system"""
        pass


class CollisionHandler(ABC):
    """Abstract base class for collision handling (OCP)"""
    @abstractmethod
    def handle_collision(self, collision: 'Collision', system_state: 'SystemState') -> 'SystemState':
        """Handle a single collision"""
        pass


class SystemInitializer(ABC):
    """Abstract base class for system initialization (OCP)"""
    @abstractmethod
    def initialize(self) -> 'SystemState':
        """Initialize and return system state"""
        pass


class Visualizer(ABC):
    """Abstract base class for visualization (SRP - separate from physics)"""
    @abstractmethod
    def visualize(self, history: 'SimulationHistory'):
        """Visualize simulation results"""
        pass


# ============= Data Classes (SRP - just data) =============

@dataclass
class Mass:
    """Represents a point mass"""
    id: int
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)


@dataclass
class Spring:
    """Represents a spring connection"""
    mass1_id: int
    mass2_id: int
    k: float
    rest_length: float
    
    def potential_energy(self, positions: np.ndarray) -> float:
        """Calculate spring potential energy"""
        p1 = positions[self.mass1_id]
        p2 = positions[self.mass2_id]
        extension = np.linalg.norm(p2 - p1) - self.rest_length
        return 0.5 * self.k * extension**2


@dataclass
class SystemState:
    """Immutable system state (SRP - just holds state)"""
    masses: List[Mass]
    springs: List[Spring]
    time: float
    
    @property
    def positions(self) -> np.ndarray:
        """Get all positions as numpy array"""
        return np.array([m.position for m in self.masses])
    
    @property
    def velocities(self) -> np.ndarray:
        """Get all velocities as numpy array"""
        return np.array([m.velocity for m in self.masses])
    
    def with_updated_dynamics(self, positions: np.ndarray, velocities: np.ndarray, dt: float) -> 'SystemState':
        """Create new state with updated positions/velocities"""
        new_masses = []
        for i, mass in enumerate(self.masses):
            new_masses.append(Mass(
                id=mass.id,
                mass=mass.mass,
                position=positions[i].copy(),
                velocity=velocities[i].copy()
            ))
        return SystemState(new_masses, self.springs, self.time + dt)


@dataclass
class Collision:
    """Represents a collision event"""
    mass_id: int
    spring: Spring
    distance: float


@dataclass
class SimulationHistory:
    """Stores simulation history (SRP - just storage)"""
    states: List[SystemState]
    
    def add_state(self, state: SystemState):
        self.states.append(state)
    
    @property
    def times(self) -> np.ndarray:
        return np.array([s.time for s in self.states])
    
    @property
    def energies(self) -> np.ndarray:
        return np.array([self._calculate_energy(s) for s in self.states])
    
    def _calculate_energy(self, state: SystemState) -> float:
        kinetic = sum(m.kinetic_energy() for m in state.masses)
        potential = sum(s.potential_energy(state.positions) for s in state.springs)
        return kinetic + potential


# ============= Concrete Implementations =============

class SpringForce(Force):
    """Spring force implementation (SRP - just spring forces)"""
    
    def calculate(self, system_state: SystemState) -> np.ndarray:
        forces = np.zeros((len(system_state.masses), 2))
        
        for spring in system_state.springs:
            # Get positions
            p1 = system_state.positions[spring.mass1_id]
            p2 = system_state.positions[spring.mass2_id]
            
            # Calculate spring force
            r = p2 - p1
            length = np.linalg.norm(r)
            if length > 0:
                r_hat = r / length
                force_magnitude = spring.k * (length - spring.rest_length)
                
                # Apply equal and opposite forces
                forces[spring.mass1_id] += force_magnitude * r_hat
                forces[spring.mass2_id] -= force_magnitude * r_hat
        
        return forces


class ThermalForce(Force):
    """Random thermal forces (OCP - easy to add new force types)"""
    
    def __init__(self, temperature: float):
        self.temperature = temperature
    
    def calculate(self, system_state: SystemState) -> np.ndarray:
        n_masses = len(system_state.masses)
        return np.random.normal(0, np.sqrt(self.temperature), (n_masses, 2))


class CompositeForce(Force):
    """Combines multiple forces (Composite pattern)"""
    
    def __init__(self, forces: List[Force]):
        self.forces = forces
    
    def calculate(self, system_state: SystemState) -> np.ndarray:
        total = np.zeros((len(system_state.masses), 2))
        for force in self.forces:
            total += force.calculate(system_state)
        return total


class RK4Integrator(Integrator):
    """Runge-Kutta 4th order integration (SRP)"""
    
    def step(self, system_state: SystemState, forces: np.ndarray, dt: float) -> SystemState:
        # Get current state
        x0 = system_state.positions
        v0 = system_state.velocities
        masses = np.array([m.mass for m in system_state.masses])
        
        # RK4 implementation
        a0 = forces / masses[:, np.newaxis]
        
        # k1
        k1_x = v0 * dt
        k1_v = a0 * dt
        
        # k2 (we'd need force recalculation for full accuracy)
        k2_x = (v0 + k1_v/2) * dt
        k2_v = a0 * dt  # Simplified - assumes constant force
        
        # k3
        k3_x = (v0 + k2_v/2) * dt
        k3_v = a0 * dt
        
        # k4
        k4_x = (v0 + k3_v) * dt
        k4_v = a0 * dt
        
        # Combine
        new_x = x0 + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        new_v = v0 + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        
        return system_state.with_updated_dynamics(new_x, new_v, dt)


class LineSegmentCollisionDetector(CollisionDetector):
    """Detects mass-spring collisions (SRP)"""
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
    
    def detect_collisions(self, system_state: SystemState) -> List[Collision]:
        collisions = []
        
        for mass in system_state.masses:
            for spring in system_state.springs:
                if mass.id in [spring.mass1_id, spring.mass2_id]:
                    continue
                
                distance = self._point_to_line_distance(
                    mass.position,
                    system_state.masses[spring.mass1_id].position,
                    system_state.masses[spring.mass2_id].position
                )
                
                if distance < self.threshold:
                    collisions.append(Collision(mass.id, spring, distance))
        
        return collisions
    
    def _point_to_line_distance(self, point: np.ndarray, 
                               line_start: np.ndarray, 
                               line_end: np.ndarray) -> float:
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            return np.linalg.norm(point - line_start)
        
        t = np.clip(np.dot(point - line_start, line_vec) / line_length**2, 0, 1)
        closest = line_start + t * line_vec
        
        return np.linalg.norm(point - closest)


class ElasticCollisionHandler(CollisionHandler):
    """Handles elastic collisions (SRP)"""
    
    def handle_collision(self, collision: Collision, system_state: SystemState) -> SystemState:
        # Get affected mass IDs
        affected_ids = {collision.mass_id, collision.spring.mass1_id, collision.spring.mass2_id}
        
        # Create new velocities
        new_velocities = system_state.velocities.copy()
        for mass_id in affected_ids:
            new_velocities[mass_id] *= -1
        
        return system_state.with_updated_dynamics(
            system_state.positions, new_velocities, 0
        )


class HexagonalInitializer(SystemInitializer):
    """Initializes hexagonal configuration (SRP)"""
    
    def __init__(self, n_masses: int = 6, radius: float = 1.0, 
                 mass: float = 1.0, k: float = 1.0, rest_length: float = 1.0):
        self.n_masses = n_masses
        self.radius = radius
        self.mass = mass
        self.k = k
        self.rest_length = rest_length
    
    def initialize(self) -> SystemState:
        masses = []
        springs = []
        
        # Create masses at vertices
        angles = np.linspace(0, 2*np.pi, self.n_masses + 1)[:-1]
        for i in range(self.n_masses):
            x = self.radius * np.cos(angles[i])
            y = self.radius * np.sin(angles[i])
            masses.append(Mass(
                id=i,
                mass=self.mass,
                position=np.array([x, y]),
                velocity=np.zeros(2)
            ))
        
        # Add perturbation
        masses[-1].position += np.array([0.2, -0.2])
        
        # Create springs
        for i in range(self.n_masses):
            next_i = (i + 1) % self.n_masses
            springs.append(Spring(i, next_i, self.k, self.rest_length))
        
        return SystemState(masses, springs, 0.0)


class MatplotlibVisualizer(Visualizer):
    """Matplotlib-based visualization (SRP, DIP - depends on abstraction)"""
    
    def visualize(self, history: SimulationHistory):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Setup axes
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal')
        ax1.set_title('Spring-Mass System')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Total Energy')
        ax2.set_title('Energy Conservation')
        ax2.grid(True, alpha=0.3)
        
        # Plot energy
        ax2.plot(history.times, history.energies)
        
        # Animation
        line, = ax1.plot([], [], 'b-', linewidth=2)
        dots, = ax1.plot([], [], 'ro', markersize=10)
        
        def init():
            line.set_data([], [])
            dots.set_data([], [])
            return line, dots
        
        def update(frame):
            if frame >= len(history.states):
                return line, dots
            
            state = history.states[frame]
            positions = state.positions
            
            # Create closed loop
            x_coords = list(positions[:, 0]) + [positions[0, 0]]
            y_coords = list(positions[:, 1]) + [positions[0, 1]]
            
            line.set_data(x_coords, y_coords)
            dots.set_data(positions[:, 0], positions[:, 1])
            
            return line, dots
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(history.states),
            init_func=init, interval=50, blit=True
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim


# ============= Main Simulator (uses DI) =============

class PhysicsSimulator:
    """Main simulator class using dependency injection (DIP)"""
    
    def __init__(self, 
                 initializer: SystemInitializer,
                 force: Force,
                 integrator: Integrator,
                 collision_detector: Optional[CollisionDetector] = None,
                 collision_handler: Optional[CollisionHandler] = None,
                 dt: float = 0.01):
        self.initializer = initializer
        self.force = force
        self.integrator = integrator
        self.collision_detector = collision_detector
        self.collision_handler = collision_handler
        self.dt = dt
        self.history = SimulationHistory([])
    
    def simulate(self, duration: float) -> SimulationHistory:
        """Run simulation"""
        state = self.initializer.initialize()
        self.history.add_state(state)
        
        steps = int(duration / self.dt)
        for _ in range(steps):
            # Handle collisions
            if self.collision_detector and self.collision_handler:
                collisions = self.collision_detector.detect_collisions(state)
                for collision in collisions:
                    state = self.collision_handler.handle_collision(collision, state)
            
            # Calculate forces and integrate
            forces = self.force.calculate(state)
            state = self.integrator.step(state, forces, self.dt)
            
            self.history.add_state(state)
        
        return self.history


# ============= Example Usage =============

def main():
    """Example of how to use the SOLID design"""
    # Create components (easy to swap implementations)
    initializer = HexagonalInitializer()
    
    # Compose forces
    spring_force = SpringForce()
    # thermal_force = ThermalForce(temperature=0.1)
    # force = CompositeForce([spring_force, thermal_force])
    force = spring_force
    
    integrator = RK4Integrator()
    collision_detector = LineSegmentCollisionDetector()
    collision_handler = ElasticCollisionHandler()
    
    # Create simulator with dependency injection
    simulator = PhysicsSimulator(
        initializer=initializer,
        force=force,
        integrator=integrator,
        collision_detector=collision_detector,
        collision_handler=collision_handler,
        dt=0.01
    )
    
    # Run simulation
    history = simulator.simulate(duration=10.0)
    
    # Visualize separately (SRP)
    visualizer = MatplotlibVisualizer()
    visualizer.visualize(history)


if __name__ == "__main__":
    main()